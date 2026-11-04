"""
Visual Odometry and Tracking engine for Monocular Visual SLAM.
Features hybrid Pyramidal KLT Optical Flow for fast, continuous frame-to-frame tracking,
ORB descriptor matching for relocalization, and robust PnP-RANSAC pose estimation.
"""

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
from slam.relocalization import Relocalizer


class TrackingState(Enum):
    SYSTEM_NOT_READY = 0
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
        self.fast_detector = cv2.FastFeatureDetector_create(threshold=20)
        self.relocalizer = Relocalizer(self.camera, self.feature_tracker, min_inliers_pnp=self.min_inliers_pnp)

        self.state = TrackingState.NOT_INITIALIZED

        # Frames history
        self.init_reference_frame: Optional[Frame] = None
        self.last_frame: Optional[Frame] = None
        self.current_frame: Optional[Frame] = None
        self.last_keyframe: Optional[Frame] = None

        # Motion model (relative pose T_curr_prev = T_cw_curr @ T_cw_prev^-1)
        self.velocity = np.eye(4, dtype=np.float64)
        self.has_velocity = False

        # Stats
        self.num_inliers = 0
        self.frames_since_kf = 0

    def process_frame(
        self,
        frame_id: int,
        timestamp: float,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None
    ) -> Frame:
        """
        Processes an incoming video frame.
        Executes hybrid KLT optical flow, PnP tracking, or initialization.
        """
        self.current_frame = Frame(frame_id, timestamp, image, self.camera, depth=depth)

        if self.state == TrackingState.NOT_INITIALIZED:
            self._handle_initialization()
        elif self.state == TrackingState.OK:
            self._track_frame()
        elif self.state == TrackingState.LOST:
            self._handle_lost()

        self.slam_map.add_frame(self.current_frame)
        if self.current_frame.has_pose:
            self.last_frame = self.current_frame

        return self.current_frame

    def _handle_initialization(self) -> None:
        """Attempts initialization: RGB-D direct metric seeding if depth available, else monocular."""
        kps, des = self.feature_tracker.detect_and_compute(self.current_frame.image)
        self.current_frame.set_features(kps, des)

        # 1. RGB-D Direct Metric Initialization
        if self.current_frame.depth is not None:
            self.current_frame.set_pose(np.eye(4, dtype=np.float64))
            h, w = self.current_frame.depth.shape[:2]
            fx, fy = self.camera.fx, self.camera.fy
            cx, cy = self.camera.cx, self.camera.cy

            valid_count = 0
            for i, kp in enumerate(kps):
                u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
                if 0 <= u < w and 0 <= v < h:
                    z = float(self.current_frame.depth[v, u])
                    if 0.3 <= z <= 3.8 and np.isfinite(z):
                        x = (u - cx) * z / fx
                        y = (v - cy) * z / fy
                        pt_3d = np.array([x, y, z], dtype=np.float64)

                        color = self.current_frame.image[v, u]
                        rgb = color[::-1] if len(color) == 3 else np.array([200, 200, 200])
                        des_i = des[i] if des is not None and len(des) > i else None

                        mp = MapPoint(pt_3d, first_frame_id=self.current_frame.id, color=rgb, descriptor=des_i)
                        mp.add_observation(self.current_frame.id, i)
                        self.slam_map.add_map_point(mp)
                        self.current_frame.associate_map_point(i, mp)
                        valid_count += 1

            if valid_count >= 35:
                self.slam_map.add_keyframe(self.current_frame)
                self.state = TrackingState.OK
                self.last_keyframe = self.current_frame
                self.num_inliers = valid_count
                self.frames_since_kf = 0
                self.has_velocity = False
                print(f"[SLAM Initializer] Success! RGB-D metric map initialized with {valid_count} MapPoints on frame {self.current_frame.id}.")
                return

        # 2. Monocular two-view fallback
        if self.init_reference_frame is None:
            if self.current_frame.num_features >= 50:
                self.init_reference_frame = self.current_frame
            return

        matches = self.feature_tracker.match_features(
            self.init_reference_frame.des,
            self.current_frame.des,
            ratio_thresh=0.80
        )

        success = self.initializer.try_initialize(
            self.init_reference_frame,
            self.current_frame,
            matches,
            self.slam_map
        )

        if success:
            self.state = TrackingState.OK
            self.last_keyframe = self.current_frame
            self.num_inliers = self.current_frame.count_tracked_points()
            self.frames_since_kf = 0
            self.has_velocity = False
        else:
            if self.current_frame.id - self.init_reference_frame.id > 30:
                self.init_reference_frame = self.current_frame

    def _track_frame(self) -> None:
        """
        Tracks current frame using Pyramidal KLT Optical Flow on MapPoints,
        followed by PnP-RANSAC pose estimation.
        """
        self.frames_since_kf += 1

        # 1. Gather all active MapPoint observations from last frame
        last_pts_3d = []
        last_pts_2d = []
        last_mps = []

        for idx, mp in enumerate(self.last_frame.map_points):
            if mp is not None and not mp.is_bad and not self.last_frame.outliers[idx]:
                last_pts_3d.append(mp.point_3d)
                last_pts_2d.append(self.last_frame.kps_pts[idx])
                last_mps.append(mp)

        tracked_3d = []
        tracked_2d = []
        tracked_mps = []

        # 2. Pyramidal KLT Optical Flow tracking
        if len(last_pts_2d) > 0:
            last_pts_arr = np.array(last_pts_2d, dtype=np.float32)
            curr_pts_arr, valid_klt = self.feature_tracker.track_klt(
                self.last_frame.gray,
                self.current_frame.gray,
                last_pts_arr,
                max_fb_err=2.0
            )

            for i in range(len(last_pts_2d)):
                if valid_klt[i]:
                    tracked_3d.append(last_pts_3d[i])
                    tracked_2d.append(curr_pts_arr[i])
                    tracked_mps.append(last_mps[i])

        # 3. Fallback: If KLT points are low, supplement with ORB descriptor matching across recent keyframes
        if len(tracked_3d) < 50 and len(self.slam_map.keyframes) > 0:
            kps_orb, des_orb = self.feature_tracker.detect_and_compute(self.current_frame.image)
            self.current_frame.set_features(kps_orb, des_orb)

            existing_ids = {mp.id for mp in tracked_mps}
            recent_kfs = list(self.slam_map.keyframes.values())[-4:]
            for ref_kf in reversed(recent_kfs):
                if ref_kf.des is not None and len(ref_kf.des) > 0:
                    orb_matches = self.feature_tracker.match_features(ref_kf.des, des_orb, ratio_thresh=0.82)
                    for m_ref, m_curr, _ in orb_matches:
                        mp = ref_kf.map_points[m_ref]
                        if mp is not None and not mp.is_bad and not ref_kf.outliers[m_ref]:
                            if mp.id not in existing_ids:
                                tracked_3d.append(mp.point_3d)
                                tracked_2d.append(self.current_frame.kps_pts[m_curr])
                                tracked_mps.append(mp)
                                existing_ids.add(mp.id)
                if len(tracked_3d) >= 60:
                    break

        if len(tracked_3d) < self.min_inliers_pnp:
            self._try_relocalization()
            return

        pts_3d = np.array(tracked_3d, dtype=np.float64)
        pts_2d = np.array(tracked_2d, dtype=np.float32)

        # 4. Motion model extrinsic guess
        use_extrinsic_guess = False
        rvec_init = np.zeros((3, 1), dtype=np.float64)
        tvec_init = np.zeros((3, 1), dtype=np.float64)

        if self.has_velocity and self.last_frame.has_pose:
            T_pred = self.velocity @ self.last_frame.get_pose()
            R_pred = T_pred[:3, :3]
            t_pred = T_pred[:3, 3].reshape(3, 1)
            rvec_init, _ = cv2.Rodrigues(R_pred)
            tvec_init = t_pred
            use_extrinsic_guess = True

        # 5. PnP-RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d,
            pts_2d,
            self.camera.K,
            None,
            rvec=rvec_init if use_extrinsic_guess else None,
            tvec=tvec_init if use_extrinsic_guess else None,
            useExtrinsicGuess=use_extrinsic_guess,
            iterationsCount=300,
            reprojectionError=3.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success or inliers is None or len(inliers) < self.min_inliers_pnp:
            # Fallback retry with slightly relaxed threshold before declaring tracking lost
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d,
                pts_2d,
                self.camera.K,
                None,
                iterationsCount=300,
                reprojectionError=4.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        if not success or inliers is None or len(inliers) < self.min_inliers_pnp:
            self._try_relocalization()
            return

        inliers = inliers.flatten()
        cv2.solvePnPRefineLM(
            pts_3d[inliers],
            pts_2d[inliers],
            self.camera.K,
            None,
            rvec,
            tvec
        )

        R, _ = cv2.Rodrigues(rvec)
        self.current_frame.set_R_t(R, tvec)

        # Update velocity model
        if self.last_frame.has_pose:
            T_curr = self.current_frame.get_pose()
            T_last_inv = self.last_frame.get_inverse_pose()
            self.velocity = T_curr @ T_last_inv
            self.has_velocity = True

        # Register tracked inliers in current frame
        inlier_kps = [cv2.KeyPoint(x=float(pts_2d[i, 0]), y=float(pts_2d[i, 1]), size=31.0) for i in inliers]
        inlier_mps = [tracked_mps[i] for i in inliers]

        # Use the MapPoints' existing representative descriptors (safe, zero-overhead, no dropped points)
        inlier_des_list = []
        for mp in inlier_mps:
            if mp.descriptor is not None and len(mp.descriptor) == 32:
                inlier_des_list.append(mp.descriptor)
            else:
                inlier_des_list.append(np.zeros(32, dtype=np.uint8))

        inlier_des = np.array(inlier_des_list, dtype=np.uint8) if inlier_des_list else np.empty((0, 32), dtype=np.uint8)
        self.current_frame.set_features(inlier_kps, inlier_des, inlier_mps)

        # Link observations to MapPoints
        for idx, mp in enumerate(inlier_mps):
            mp.add_observation(self.current_frame.id, idx)
            mp.num_found += 1

        self.num_inliers = len(inliers)

        # Project local map points to replenish tracks and prevent track starvation
        self._track_local_map(R, tvec, inlier_mps)

        # 6. Check KeyFrame insertion
        if self._should_insert_keyframe():
            self._insert_keyframe()

    def _should_insert_keyframe(self) -> bool:
        """Determines if current frame should become a KeyFrame."""
        if self.last_keyframe is None:
            return True

        if self.frames_since_kf < self.min_frames_between_kfs:
            return False

        if self.frames_since_kf >= self.max_frames_between_kfs:
            return True

        curr_pts = self.current_frame.count_tracked_points()
        last_kf_pts = max(self.last_keyframe.count_tracked_points(), 1)
        ratio = curr_pts / last_kf_pts

        if ratio < 0.60:
            return True

        pos_curr = self.current_frame.get_camera_center()
        pos_last_kf = self.last_keyframe.get_camera_center()
        dist = np.linalg.norm(pos_curr - pos_last_kf)

        if dist > 0.08:
            return True

        # Significant rotation
        R_rel = self.current_frame.get_rotation() @ self.last_keyframe.get_rotation().T
        angle = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)))
        if angle > 7.0:
            return True

        return False

    def _insert_keyframe(self) -> None:
        """Adds current frame as KeyFrame, replenishes features, and executes local mapping."""
        # Detect new features in unrepresented image regions
        new_kps, new_des = self.feature_tracker.detect_new_features(
            self.current_frame.gray,
            self.current_frame.kps_pts,
            min_distance=15,
            max_new=1000
        )
        if new_kps:
            self.current_frame.add_features(new_kps, new_des)

        # In RGB-D mode: instantiate true metric 3D landmarks for unassociated keypoints
        if self.current_frame.depth is not None and self.current_frame.has_pose:
            T_wc = self.current_frame.get_inverse_pose()
            R_wc = T_wc[:3, :3]
            t_wc = T_wc[:3, 3]
            h, w = self.current_frame.depth.shape[:2]
            fx, fy = self.camera.fx, self.camera.fy
            cx, cy = self.camera.cx, self.camera.cy

            for idx, mp in enumerate(self.current_frame.map_points):
                if mp is None:
                    pt = self.current_frame.kps_pts[idx]
                    u, v = int(round(pt[0])), int(round(pt[1]))
                    if 0 <= u < w and 0 <= v < h:
                        z = float(self.current_frame.depth[v, u])
                        if 0.3 <= z <= 3.8 and np.isfinite(z):
                            x_c = (u - cx) * z / fx
                            y_c = (v - cy) * z / fy
                            pt_c = np.array([x_c, y_c, z], dtype=np.float64)
                            pt_w = R_wc @ pt_c + t_wc

                            color = self.current_frame.image[v, u]
                            rgb = color[::-1] if len(color) == 3 else np.array([200, 200, 200])
                            des_i = self.current_frame.des[idx] if len(self.current_frame.des) > idx else None

                            new_mp = MapPoint(pt_w, first_frame_id=self.current_frame.id, color=rgb, descriptor=des_i)
                            new_mp.add_observation(self.current_frame.id, idx)
                            self.slam_map.add_map_point(new_mp)
                            self.current_frame.associate_map_point(idx, new_mp)

        self.slam_map.add_keyframe(self.current_frame)
        self.last_keyframe = self.current_frame
        self.frames_since_kf = 0

        # Trigger local mapping
        self.local_mapping.process_new_keyframe(self.current_frame, self.slam_map)

    def _try_relocalization(self) -> None:
        """Attempts relocalization against keyframes."""
        self.state = TrackingState.LOST
        self.has_velocity = False

    def _handle_lost(self) -> None:
        """
        Global relocalization searching across all keyframes in the map.
        Restores tracking when camera re-enters known territory.
        """
        if len(self.slam_map.keyframes) == 0:
            return

        # 1. Fast Bag of Visual Words relocalization
        vocab = self.local_mapping.loop_closing.vocab
        index = self.local_mapping.loop_closing.index
        if self.relocalizer.relocalize(self.current_frame, self.slam_map, vocab, index):
            self.state = TrackingState.OK
            self.num_inliers = self.current_frame.count_tracked_points()
            self.last_keyframe = self.slam_map.keyframes.get(max(self.slam_map.keyframes.keys()))
            self.frames_since_kf = 0
            print(f"[Relocalize] Successfully recovered pose at frame {self.current_frame.id} via visual vocabulary!")
            return

        # Ensure current frame has ORB features extracted
        if self.current_frame.num_features < 50:
            kps, des = self.feature_tracker.detect_and_compute(self.current_frame.image)
            self.current_frame.set_features(kps, des)

        if len(self.current_frame.des) == 0:
            return

        # Search keyframes, prioritizing those with highest point counts
        kfs_sorted = sorted(
            self.slam_map.keyframes.values(),
            key=lambda kf: kf.count_tracked_points(),
            reverse=True
        )

        for kf in kfs_sorted:
            if kf.des is None or len(kf.des) < 20:
                continue

            matches = self.feature_tracker.match_features(
                kf.des,
                self.current_frame.des,
                ratio_thresh=0.85
            )

            if len(matches) < self.min_inliers_pnp:
                continue

            pts_3d = []
            pts_2d = []
            match_pairs = []

            for m_kf, m_curr, _ in matches:
                mp = kf.map_points[m_kf]
                if mp is not None and not mp.is_bad and not kf.outliers[m_kf]:
                    pts_3d.append(mp.point_3d)
                    pts_2d.append(self.current_frame.kps_pts[m_curr])
                    match_pairs.append((m_curr, mp))

            if len(pts_3d) >= self.min_inliers_pnp:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    np.array(pts_3d, dtype=np.float64),
                    np.array(pts_2d, dtype=np.float32),
                    self.camera.K,
                    None,
                    iterationsCount=300,
                    reprojectionError=4.0,
                    confidence=0.99
                )

                if success and inliers is not None and len(inliers) >= self.min_inliers_pnp:
                    R, _ = cv2.Rodrigues(rvec)
                    self.current_frame.set_R_t(R, tvec)

                    inlier_indices = inliers.flatten()
                    for idx in inlier_indices:
                        m_curr, mp = match_pairs[idx]
                        self.current_frame.associate_map_point(m_curr, mp)

                    self.state = TrackingState.OK
                    self.num_inliers = len(inlier_indices)
                    self.last_keyframe = kf
                    self.frames_since_kf = 0
                    print(f"[SLAM Relocalization] Successfully relocalized at frame {self.current_frame.id} with {len(inlier_indices)} inliers against KeyFrame {kf.id}!")
                    self.local_mapping.triangulate_with_keyframe(kf, self.current_frame, self.slam_map)
                    self.slam_map.add_keyframe(self.current_frame)
                    self._track_local_map(R, tvec, [match_pairs[idx][1] for idx in inlier_indices])
                    return

    def _track_local_map(self, R: np.ndarray, tvec: np.ndarray, inlier_mps: List[MapPoint]) -> None:
        """
        Projects local 3D MapPoints from recent keyframes into the current frame.
        Maintains a dense set of tracked inliers (80-120+ points) to prevent track starvation.
        """
        if len(self.slam_map.keyframes) < 2:
            return

        existing_mp_ids = {mp.id for mp in inlier_mps}
        recent_kfs = list(self.slam_map.keyframes.values())[-6:]
        local_mps = [
            mp for kf in recent_kfs for mp in kf.map_points
            if mp is not None and not mp.is_bad and mp.id not in existing_mp_ids
        ]
        if not local_mps:
            return

        pts_w = np.array([mp.point_3d for mp in local_mps], dtype=np.float64)
        pts_c = (R @ pts_w.T + tvec.reshape(3, 1)).T
        valid_depth = (pts_c[:, 2] > 0.2) & (pts_c[:, 2] < 5.0)
        if not np.any(valid_depth):
            return

        sub_indices = np.where(valid_depth)[0]
        uvs, val_mask = self.camera.project(pts_c[sub_indices])
        h, w = self.current_frame.gray.shape[:2]

        new_kps_add = []
        new_des_add = []
        new_mps_add = []

        for s_idx, is_val in enumerate(val_mask):
            if not is_val:
                continue
            u, v = uvs[s_idx]
            if 5 <= u < w - 5 and 5 <= v < h - 5:
                mp = local_mps[sub_indices[s_idx]]
                new_kps_add.append(cv2.KeyPoint(x=float(u), y=float(v), size=31.0))
                desc = mp.descriptor if (mp.descriptor is not None and len(mp.descriptor) == 32) else np.zeros(32, dtype=np.uint8)
                new_des_add.append(desc)
                new_mps_add.append(mp)
                existing_mp_ids.add(mp.id)
                if len(existing_mp_ids) >= 120:
                    break

        if new_kps_add:
            des_arr = np.array(new_des_add, dtype=np.uint8)
            start_idx = self.current_frame.num_features
            self.current_frame.add_features(new_kps_add, des_arr, new_mps_add)
            for offset, mp in enumerate(new_mps_add):
                mp.add_observation(self.current_frame.id, start_idx + offset)
                mp.num_found += 1

        self.num_inliers = self.current_frame.count_tracked_points()
