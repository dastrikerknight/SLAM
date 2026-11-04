"""
Two-view initialization for Monocular Visual SLAM.
Computes relative pose and initial 3D MapPoints using Essential Matrix decomposition,
cheirality verification, parallax check, and triangulation.
"""

from typing import Tuple, Optional, List
import numpy as np
import cv2
from slam.camera import Camera
from slam.frame import Frame
from slam.map import Map
from slam.map_point import MapPoint


class Initializer:
    def __init__(
        self,
        min_matches: int = 50,
        min_inliers: int = 35,
        min_parallax_deg: float = 2.5,
        max_reprojection_err: float = 2.5
    ):
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.min_parallax_deg = min_parallax_deg
        self.max_reprojection_err = max_reprojection_err

    def try_initialize(
        self,
        frame0: Frame,
        frame1: Frame,
        matches: List[Tuple[int, int, float]],
        slam_map: Map
    ) -> bool:
        """
        Attempts two-view initialization between frame0 (reference) and frame1 (current).
        If successful, populates slam_map with keyframes and initial 3D MapPoints.
        """
        if len(matches) < self.min_matches:
            return False

        # Extract matched 2D coordinates
        idx0 = np.array([m[0] for m in matches], dtype=int)
        idx1 = np.array([m[1] for m in matches], dtype=int)

        pts0 = frame0.kps_pts[idx0]
        pts1 = frame1.kps_pts[idx1]

        camera = frame0.camera

        # 1. Estimate Essential Matrix with RANSAC
        E, inlier_mask = cv2.findEssentialMat(
            pts0, pts1,
            camera.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.5
        )

        if E is None or inlier_mask is None:
            return False

        inlier_mask = inlier_mask.ravel().astype(bool)
        if np.sum(inlier_mask) < self.min_inliers:
            return False

        # 2. Decompose Essential Matrix and resolve cheirality (points in front of both cameras)
        num_valid, R, t, pose_mask = cv2.recoverPose(
            E, pts0, pts1, camera.K,
            mask=inlier_mask.astype(np.uint8)
        )

        if num_valid < self.min_inliers:
            return False

        pose_mask = pose_mask.ravel().astype(bool)
        inliers_idx = np.where(pose_mask)[0]

        if len(inliers_idx) < self.min_inliers:
            return False

        # 3. Parallax Check: Ensure camera actually moved (not pure rotation or stationary)
        pts0_in = pts0[inliers_idx]
        pts1_in = pts1[inliers_idx]

        # Bearing rays in camera frame
        rays0 = camera.unproject(pts0_in, depth=1.0)
        rays0 /= np.linalg.norm(rays0, axis=1, keepdims=True)

        rays1 = camera.unproject(pts1_in, depth=1.0)
        rays1 /= np.linalg.norm(rays1, axis=1, keepdims=True)

        # Rotate rays1 to camera 0 frame
        rays1_rot = (R.T @ rays1.T).T

        # Cosine of parallax angles
        cos_angles = np.sum(rays0 * rays1_rot, axis=1)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(cos_angles))
        median_parallax = float(np.median(angles_deg))

        if median_parallax < self.min_parallax_deg:
            # Baseline is too small to accurately triangulate depths
            return False

        # 4. Triangulate initial 3D MapPoints
        P0 = camera.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P1 = camera.K @ np.hstack((R, t))

        pts4d = cv2.triangulatePoints(P0, P1, pts0_in.T, pts1_in.T)
        pts3d = (pts4d[:3] / (pts4d[3] + 1e-12)).T

        # Metric scale normalization: rescale so median depth is 1.2m
        pos_z_mask = (pts3d[:, 2] > 0.1) & (pts3d[:, 2] < 100.0)
        if np.sum(pos_z_mask) < self.min_inliers:
            return False

        raw_median_z = float(np.median(pts3d[pos_z_mask, 2]))
        scale_factor = 1.2 / max(raw_median_z, 1e-3)
        t_scaled = t.flatten() * scale_factor
        pts3d_scaled = pts3d * scale_factor

        # 5. Check positive depth and reprojection error
        valid_points = []
        for i, pt_w in enumerate(pts3d_scaled):
            z0 = pt_w[2]
            if z0 < 0.2 or z0 > 5.0 or not np.isfinite(z0):
                continue

            # In frame 1 coordinates
            pt_c1 = R @ pt_w + t_scaled
            z1 = pt_c1[2]
            if z1 < 0.2 or z1 > 5.0 or not np.isfinite(z1):
                continue

            # Reprojection in frame 0
            uv0, ok0 = camera.project(pt_w)
            if not ok0 or np.linalg.norm(uv0[0] - pts0_in[i]) > self.max_reprojection_err:
                continue

            # Reprojection in frame 1
            uv1, ok1 = camera.project(pt_c1)
            if not ok1 or np.linalg.norm(uv1[0] - pts1_in[i]) > self.max_reprojection_err:
                continue

            orig_m_idx = inliers_idx[i]
            kp0_idx = idx0[orig_m_idx]
            kp1_idx = idx1[orig_m_idx]

            valid_points.append((pt_w, kp0_idx, kp1_idx))

        if len(valid_points) < self.min_inliers:
            return False

        # 6. Set KeyFrame poses
        frame0.set_pose(np.eye(4, dtype=np.float64))
        T1 = np.eye(4, dtype=np.float64)
        T1[:3, :3] = R
        T1[:3, 3] = t_scaled
        frame1.set_pose(T1)

        slam_map.add_keyframe(frame0)
        slam_map.add_keyframe(frame1)
        slam_map.add_frame(frame0)
        slam_map.add_frame(frame1)

        # 7. Create MapPoints and link observations
        for pt_w, kp0_idx, kp1_idx in valid_points:
            desc0 = frame0.des[kp0_idx]
            desc1 = frame1.des[kp1_idx]

            # Sample color from frame0 image
            u, v = int(round(frame0.kps_pts[kp0_idx][0])), int(round(frame0.kps_pts[kp0_idx][1]))
            if 0 <= v < frame0.image.shape[0] and 0 <= u < frame0.image.shape[1]:
                pixel_val = frame0.image[v, u]
                if len(frame0.image.shape) == 3:
                    color = np.array(pixel_val[::-1], dtype=np.uint8)  # BGR to RGB
                else:
                    color = np.array([pixel_val, pixel_val, pixel_val], dtype=np.uint8)
            else:
                color = np.array([0, 255, 0], dtype=np.uint8)

            map_point = MapPoint(pt_w, first_frame_id=frame0.id, color=color, descriptor=desc0)
            map_point.update_descriptor([desc0, desc1])

            slam_map.add_map_point(map_point)
            frame0.associate_map_point(kp0_idx, map_point)
            frame1.associate_map_point(kp1_idx, map_point)

        print(f"[SLAM Initializer] Success! Initialized map with {len(valid_points)} MapPoints. Parallax: {median_parallax:.2f} deg.")
        return True
