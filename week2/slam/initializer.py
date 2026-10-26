from typing import Tuple, List
import numpy as np
import cv2
from slam.frame import Frame
from slam.map import Map
from slam.map_point import MapPoint

class Initializer:
    def __init__(self, min_matches: int = 50, min_inliers: int = 35,
                 min_parallax_deg: float = 2.5, max_reprojection_err: float = 2.5):
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.min_parallax_deg = min_parallax_deg
        self.max_reprojection_err = max_reprojection_err

    def try_initialize(self, f0: Frame, f1: Frame, matches: List[Tuple[int, int, float]],
                       slam_map: Map) -> bool:
        if len(matches) < self.min_matches:
            return False

        idx0 = np.array([m[0] for m in matches], dtype=int)
        idx1 = np.array([m[1] for m in matches], dtype=int)
        pts0 = f0.kps_pts[idx0]
        pts1 = f1.kps_pts[idx1]

        E, inlier_mask = cv2.findEssentialMat(
            pts0, pts1, f0.camera.K, method=cv2.RANSAC, prob=0.999, threshold=1.5
        )
        if E is None or inlier_mask is None:
            return False

        inlier_mask = inlier_mask.ravel().astype(bool)
        if np.sum(inlier_mask) < self.min_inliers:
            return False

        valid, R, t, pose_mask = cv2.recoverPose(
            E, pts0, pts1, f0.camera.K, mask=inlier_mask.astype(np.uint8)
        )
        if valid < self.min_inliers:
            return False

        inliers = np.where(pose_mask.ravel().astype(bool))[0]
        if len(inliers) < self.min_inliers:
            return False

        pts0_in, pts1_in = pts0[inliers], pts1[inliers]
        rays0 = f0.camera.unproject(pts0_in, 1.0)
        rays0 /= np.linalg.norm(rays0, axis=1, keepdims=True)
        rays1 = f1.camera.unproject(pts1_in, 1.0)
        rays1 /= np.linalg.norm(rays1, axis=1, keepdims=True)

        cos_par = np.clip(np.sum(rays0 * (R.T @ rays1.T).T, axis=1), -1.0, 1.0)
        median_parallax = float(np.median(np.degrees(np.arccos(cos_par))))
        if median_parallax < self.min_parallax_deg:
            return False

        P0 = f0.camera.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P1 = f0.camera.K @ np.hstack((R, t))
        pts4d = cv2.triangulatePoints(P0, P1, pts0_in.T, pts1_in.T)
        pts3d = (pts4d[:3] / (pts4d[3] + 1e-12)).T

        pos_z = (pts3d[:, 2] > 0.1) & (pts3d[:, 2] < 100.0)
        if np.sum(pos_z) < self.min_inliers:
            return False

        scale = 1.2 / max(float(np.median(pts3d[pos_z, 2])), 1e-3)
        t_scaled = t.flatten() * scale
        pts3d *= scale

        valid_pts = []
        for i, pt_w in enumerate(pts3d):
            if pt_w[2] < 0.2 or pt_w[2] > 5.0 or not np.isfinite(pt_w[2]):
                continue
            pt_c1 = R @ pt_w + t_scaled
            if pt_c1[2] < 0.2 or pt_c1[2] > 5.0 or not np.isfinite(pt_c1[2]):
                continue

            uv0, ok0 = f0.camera.project(pt_w)
            if not ok0 or np.linalg.norm(uv0[0] - pts0_in[i]) > self.max_reprojection_err:
                continue
            uv1, ok1 = f0.camera.project(pt_c1)
            if not ok1 or np.linalg.norm(uv1[0] - pts1_in[i]) > self.max_reprojection_err:
                continue

            valid_pts.append((pt_w, idx0[inliers[i]], idx1[inliers[i]]))

        if len(valid_pts) < self.min_inliers:
            return False

        f0.set_pose(np.eye(4, dtype=np.float64))
        T1 = np.eye(4, dtype=np.float64)
        T1[:3, :3] = R
        T1[:3, 3] = t_scaled
        f1.set_pose(T1)

        slam_map.add_keyframe(f0)
        slam_map.add_keyframe(f1)
        slam_map.add_frame(f0)
        slam_map.add_frame(f1)

        for pt_w, k0, k1 in valid_pts:
            d0, d1 = f0.des[k0], f1.des[k1]
            u, v = int(round(f0.kps_pts[k0][0])), int(round(f0.kps_pts[k0][1]))
            if 0 <= v < f0.image.shape[0] and 0 <= u < f0.image.shape[1]:
                col = np.array(f0.image[v, u][::-1], dtype=np.uint8) if f0.image.ndim == 3 else np.array([f0.image[v, u]] * 3, dtype=np.uint8)
            else:
                col = np.array([0, 255, 0], dtype=np.uint8)

            mp = MapPoint(pt_w, first_frame_id=f0.id, color=col, descriptor=d0)
            mp.update_descriptor([d0, d1])
            slam_map.add_map_point(mp)
            f0.associate_map_point(k0, mp)
            f1.associate_map_point(k1, mp)

        print(f"[init] map created: {len(valid_pts)} points, parallax {median_parallax:.1f} deg")
        return True
