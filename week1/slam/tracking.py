from enum import Enum
from typing import Optional, List, Tuple
import numpy as np
import cv2

from slam.camera import Camera
from slam.frame import Frame
from slam.map import Map
from slam.map_point import MapPoint
from slam.feature_tracker import FeatureTracker
from slam.initializer import Initializer
from slam.mapping import LocalMapping

class TrackingState(Enum):
    NOT_INITIALIZED = 1
    OK = 2
    LOST = 3

class Tracking:
    def __init__(
        self,
        camera: Camera,
        slam_map: Map,
        feature_tracker: FeatureTracker,
        initializer: Initializer,
        local_mapping: LocalMapping,
        min_inliers_pnp: int = 10,
        min_frames_between_kfs: int = 4,
        max_frames_between_kfs: int = 25
    ):
        self.camera = camera
        self.slam_map = slam_map
        self.feature_tracker = feature_tracker
        self.initializer = initializer
        self.local_mapping = local_mapping

        self.min_inliers_pnp = min_inliers_pnp
        self.min_frames_between_kfs = min_frames_between_kfs
        self.max_frames_between_kfs = max_frames_between_kfs

        self.state = TrackingState.NOT_INITIALIZED
        self.init_ref_frame: Optional[Frame] = None
        self.last_frame: Optional[Frame] = None
        self.curr_frame: Optional[Frame] = None
        self.last_keyframe: Optional[Frame] = None

        self.velocity = np.eye(4, dtype=np.float64)
        self.has_velocity = False
        self.num_inliers = 0
        self.frames_since_kf = 0

    def process_frame(self, frame_id: int, ts: float, img: np.ndarray) -> Frame:
        self.curr_frame = Frame(frame_id, ts, img, self.camera)

        if self.state == TrackingState.NOT_INITIALIZED:
            self._init_pipeline()
        elif self.state == TrackingState.OK:
            self._track()
        elif self.state == TrackingState.LOST:
            self._relocalize()

        self.slam_map.add_frame(self.curr_frame)
        if self.curr_frame.has_pose:
            if self.last_frame is not None and not self.last_frame.is_keyframe:
                self.last_frame.image = None
                self.last_frame.gray = None
            self.last_frame = self.curr_frame

        return self.curr_frame

    def _init_pipeline(self):
        kps, des = self.feature_tracker.detect_and_compute(self.curr_frame.image)
        self.curr_frame.set_features(kps, des)

        if self.init_ref_frame is None:
            if self.curr_frame.num_features >= 50:
                self.init_ref_frame = self.curr_frame
            return

        matches = self.feature_tracker.match_features(
            self.init_ref_frame.des, self.curr_frame.des, ratio_thresh=0.80
        )
        ok = self.initializer.try_initialize(
            self.init_ref_frame, self.curr_frame, matches, self.slam_map
        )

        if ok:
            self.state = TrackingState.OK
            self.last_keyframe = self.curr_frame
            self.num_inliers = self.curr_frame.count_tracked_points()
            self.frames_since_kf = 0
            self.has_velocity = False
        elif self.curr_frame.id - self.init_ref_frame.id > 30:
            self.init_ref_frame = self.curr_frame

    def _track(self):
        self.frames_since_kf += 1

        last_3d, last_2d, last_mps = [], [], []
        for idx, mp in enumerate(self.last_frame.map_points):
            if mp and not mp.is_bad and not self.last_frame.outliers[idx]:
                last_3d.append(mp.point_3d)
                last_2d.append(self.last_frame.kps_pts[idx])
                last_mps.append(mp)

        tracked_3d, tracked_2d, tracked_mps = [], [], []
        if last_2d:
            pts_prev = np.array(last_2d, dtype=np.float32)
            pts_curr, valid = self.feature_tracker.track_klt(
                self.last_frame.gray, self.curr_frame.gray, pts_prev, max_fb_err=2.0
            )
            for i, ok in enumerate(valid):
                if ok:
                    tracked_3d.append(last_3d[i])
                    tracked_2d.append(pts_curr[i])
                    tracked_mps.append(last_mps[i])

        if len(tracked_3d) < 50 and self.last_keyframe and self.last_keyframe.des is not None:
            kps_orb, des_orb = self.feature_tracker.detect_and_compute(self.curr_frame.image)
            self.curr_frame.set_features(kps_orb, des_orb)
            seen_ids = {mp.id for mp in tracked_mps}

            matches = self.feature_tracker.match_features(
                self.last_keyframe.des, des_orb, ratio_thresh=0.80
            )
            for m_kf, m_curr, _ in matches:
                mp = self.last_keyframe.map_points[m_kf]
                if mp and not mp.is_bad and not self.last_keyframe.outliers[m_kf]:
                    if mp.id not in seen_ids:
                        tracked_3d.append(mp.point_3d)
                        tracked_2d.append(self.curr_frame.kps_pts[m_curr])
                        tracked_mps.append(mp)
                        seen_ids.add(mp.id)

        if len(tracked_3d) < self.min_inliers_pnp:
            self.state = TrackingState.LOST
            self.has_velocity = False
            return

        pts_3d = np.array(tracked_3d, dtype=np.float64)
        pts_2d = np.array(tracked_2d, dtype=np.float32)

        rvec_init, tvec_init = None, None
        use_guess = False
        if self.has_velocity and self.last_frame.has_pose:
            T_pred = self.velocity @ self.last_frame.get_pose()
            rvec_init, _ = cv2.Rodrigues(T_pred[:3, :3])
            tvec_init = T_pred[:3, 3].reshape(3, 1)
            use_guess = True

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, self.camera.K, None,
            rvec=rvec_init, tvec=tvec_init, useExtrinsicGuess=use_guess,
            iterationsCount=300, reprojectionError=3.0, confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not ok or inliers is None or len(inliers) < self.min_inliers_pnp:
            self.state = TrackingState.LOST
            self.has_velocity = False
            return

        inliers = inliers.flatten()
        cv2.solvePnPRefineLM(pts_3d[inliers], pts_2d[inliers], self.camera.K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        self.curr_frame.set_R_t(R, tvec)

        if self.last_frame.has_pose:
            self.velocity = self.curr_frame.get_pose() @ self.last_frame.get_inverse_pose()
            self.has_velocity = True

        inlier_kps = [cv2.KeyPoint(float(pts_2d[i, 0]), float(pts_2d[i, 1]), 31.0) for i in inliers]
        inlier_mps = [tracked_mps[i] for i in inliers]
        inlier_des = np.array([
            mp.descriptor if mp.descriptor is not None and len(mp.descriptor) == 32
            else np.zeros(32, dtype=np.uint8) for mp in inlier_mps
        ], dtype=np.uint8)

        self.curr_frame.set_features(inlier_kps, inlier_des, inlier_mps)
        for idx, mp in enumerate(inlier_mps):
            mp.add_observation(self.curr_frame.id, idx)
            mp.num_found += 1

        self.num_inliers = len(inliers)
        if self._should_keyframe():
            self._insert_keyframe()

    def _should_keyframe(self) -> bool:
        if self.last_keyframe is None:
            return True
        if self.frames_since_kf < self.min_frames_between_kfs:
            return False
        if self.frames_since_kf >= self.max_frames_between_kfs:
            return True

        curr_pts = self.curr_frame.count_tracked_points()
        last_pts = max(self.last_keyframe.count_tracked_points(), 1)
        if (curr_pts / last_pts) < 0.60:
            return True

        pos_curr = self.curr_frame.get_camera_center()
        pos_last = self.last_keyframe.get_camera_center()
        if np.linalg.norm(pos_curr - pos_last) > 0.08:
            return True

        R_rel = self.curr_frame.get_rotation() @ self.last_keyframe.get_rotation().T
        angle = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)))
        return angle > 7.0

    def _insert_keyframe(self):
        new_kps, new_des = self.feature_tracker.detect_new_features(
            self.curr_frame.gray, self.curr_frame.kps_pts, min_distance=15, max_new=1000
        )
        if new_kps:
            self.curr_frame.add_features(new_kps, new_des)

        self.slam_map.add_keyframe(self.curr_frame)
        self.last_keyframe = self.curr_frame
        self.frames_since_kf = 0
        self.local_mapping.process_new_keyframe(self.curr_frame, self.slam_map)

    def _relocalize(self):
        if not self.slam_map.keyframes:
            return
        if self.curr_frame.num_features < 50:
            kps, des = self.feature_tracker.detect_and_compute(self.curr_frame.image)
            self.curr_frame.set_features(kps, des)

        if len(self.curr_frame.des) == 0:
            return

        for kf in reversed(list(self.slam_map.keyframes.values())[-5:]):
            if kf.des is None or len(kf.des) < 20:
                continue
            matches = self.feature_tracker.match_features(kf.des, self.curr_frame.des, ratio_thresh=0.80)
            if len(matches) < self.min_inliers_pnp:
                continue

            pts_3d, pts_2d, pairs = [], [], []
            for m_kf, m_curr, _ in matches:
                mp = kf.map_points[m_kf]
                if mp and not mp.is_bad and not kf.outliers[m_kf]:
                    pts_3d.append(mp.point_3d)
                    pts_2d.append(self.curr_frame.kps_pts[m_curr])
                    pairs.append((m_curr, mp))

            if len(pts_3d) >= self.min_inliers_pnp:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    np.array(pts_3d, dtype=np.float64), np.array(pts_2d, dtype=np.float32),
                    self.camera.K, None, iterationsCount=200, reprojectionError=3.5, confidence=0.99
                )
                if ok and inliers is not None and len(inliers) >= self.min_inliers_pnp:
                    R, _ = cv2.Rodrigues(rvec)
                    self.curr_frame.set_R_t(R, tvec)
                    for idx in inliers.flatten():
                        m_curr, mp = pairs[idx]
                        self.curr_frame.associate_map_point(m_curr, mp)
                    self.state = TrackingState.OK
                    self.num_inliers = len(inliers)
                    self.last_keyframe = kf
                    self.frames_since_kf = 0
                    return
