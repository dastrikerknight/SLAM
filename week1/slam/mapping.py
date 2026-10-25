from typing import List, Tuple
import numpy as np
import cv2

from slam.camera import Camera
from slam.frame import Frame
from slam.map import Map
from slam.map_point import MapPoint
from slam.feature_tracker import FeatureTracker
from slam.optimizer import LocalBundleAdjustment

class LocalMapping:
    def __init__(
        self,
        camera: Camera,
        feature_tracker: FeatureTracker,
        min_parallax_deg: float = 1.2,
        max_reprojection_err: float = 2.5
    ):
        self.camera = camera
        self.feature_tracker = feature_tracker
        self.min_parallax_deg = min_parallax_deg
        self.max_reprojection_err = max_reprojection_err
        self.optimizer = LocalBundleAdjustment(window_size=5, max_iterations=10, max_points=120)
        self.kf_counter = 0

    def process_new_keyframe(self, curr_kf: Frame, slam_map: Map) -> int:
        self.kf_counter += 1
        kfs = list(slam_map.keyframes.values())
        if len(kfs) < 2:
            return 0

        # Triangulate with previous 1-2 keyframes using standard 2D feature matching
        new_pts = self.triangulate_with_keyframe(kfs[-2], curr_kf, slam_map)
        if len(kfs) >= 3:
            new_pts += self.triangulate_with_keyframe(kfs[-3], curr_kf, slam_map)

        slam_map.cull_map_points(min_observations=2)

        # Local BA every 3 keyframes
        if len(slam_map.keyframes) >= 3 and (self.kf_counter % 3 == 0):
            self.optimizer.optimize(slam_map, self.camera)

        return new_pts

    def triangulate_with_keyframe(self, kf1: Frame, kf2: Frame, slam_map: Map) -> int:
        unmatched_1 = [i for i, mp in enumerate(kf1.map_points) if mp is None or mp.is_bad]
        unmatched_2 = [i for i, mp in enumerate(kf2.map_points) if mp is None or mp.is_bad]

        if len(unmatched_1) < 10 or len(unmatched_2) < 10:
            return 0

        des1 = kf1.des[unmatched_1]
        des2 = kf2.des[unmatched_2]
        matches = self.feature_tracker.match_features(des1, des2, ratio_thresh=0.80)
        if not matches:
            return 0

        m_idx1 = [unmatched_1[m[0]] for m in matches]
        m_idx2 = [unmatched_2[m[1]] for m in matches]

        pts1 = kf1.kps_pts[m_idx1]
        pts2 = kf2.kps_pts[m_idx2]

        R1, t1 = kf1.get_rotation(), kf1.get_translation()
        R2, t2 = kf2.get_rotation(), kf2.get_translation()

        P1 = self.camera.K @ np.hstack((R1, t1))
        P2 = self.camera.K @ np.hstack((R2, t2))

        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = (pts4d[:3] / (pts4d[3] + 1e-12)).T

        c1 = kf1.get_camera_center()
        c2 = kf2.get_camera_center()

        new_count = 0
        for i, pt_w in enumerate(pts3d):
            pt_c1 = R1 @ pt_w + t1.flatten()
            pt_c2 = R2 @ pt_w + t2.flatten()

            if pt_c1[2] < 0.2 or pt_c1[2] > 5.0 or pt_c2[2] < 0.2 or pt_c2[2] > 5.0:
                continue
            if not np.all(np.isfinite(pt_w)):
                continue

            ray1 = pt_w - c1
            ray2 = pt_w - c2
            norm1, norm2 = np.linalg.norm(ray1), np.linalg.norm(ray2)
            if norm1 < 1e-4 or norm2 < 1e-4:
                continue

            cos_p = np.clip(np.dot(ray1, ray2) / (norm1 * norm2), -1.0, 1.0)
            if np.degrees(np.arccos(cos_p)) < self.min_parallax_deg:
                continue

            uv1, ok1 = self.camera.project(pt_c1.reshape(1, 3))
            uv2, ok2 = self.camera.project(pt_c2.reshape(1, 3))
            if not ok1 or not ok2:
                continue

            err1 = np.linalg.norm(uv1[0] - pts1[i])
            err2 = np.linalg.norm(uv2[0] - pts2[i])
            if err1 > self.max_reprojection_err or err2 > self.max_reprojection_err:
                continue

            orig_1, orig_2 = m_idx1[i], m_idx2[i]
            d1, d2 = kf1.des[orig_1], kf2.des[orig_2]

            u, v = int(round(pts2[i][0])), int(round(pts2[i][1]))
            if 0 <= v < kf2.image.shape[0] and 0 <= u < kf2.image.shape[1]:
                px = kf2.image[v, u]
                color = np.array(px[::-1], dtype=np.uint8) if len(kf2.image.shape) == 3 else np.array([px, px, px], dtype=np.uint8)
            else:
                color = np.array([255, 255, 0], dtype=np.uint8)

            mp = MapPoint(pt_w, first_frame_id=kf1.id, color=color, descriptor=d2)
            mp.update_descriptor([d1, d2])

            slam_map.add_map_point(mp)
            kf1.associate_map_point(orig_1, mp)
            kf2.associate_map_point(orig_2, mp)
            new_count += 1

        return new_count
