"""
Bundle Adjustment for Monocular Visual SLAM using SciPy's Trust Region Reflective least squares.
Optimizes camera poses and selected high-quality 3D map points within a local sliding window.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from slam.camera import Camera
from slam.frame import Frame
from slam.map import Map
from slam.map_point import MapPoint


class LocalBundleAdjustment:
    def __init__(self, window_size: int = 5, max_iterations: int = 10, max_points: int = 100):
        self.window_size = window_size
        self.max_iterations = max_iterations
        self.max_points = max_points

    def optimize(self, slam_map: Map, camera: Camera) -> int:
        """
        Executes fast local Bundle Adjustment over recent and covisible keyframes and their top MapPoints.
        Returns number of optimized keyframes.
        """
        all_kf_ids = sorted(list(slam_map.keyframes.keys()))
        if len(all_kf_ids) < 2:
            return 0

        # Select window of recent keyframes
        window_kf_ids = all_kf_ids[-self.window_size:]
        keyframes = [slam_map.keyframes[kid] for kid in window_kf_ids]

        # First keyframe in window (oldest) is fixed (anchor) to remove gauge freedom
        fixed_kf_id = window_kf_ids[0]
        free_kf_ids = window_kf_ids[1:]

        kf_id_to_idx = {kid: i for i, kid in enumerate(free_kf_ids)}

        # Find candidate 3D MapPoints observed by these keyframes
        point_obs_count: Dict[int, int] = {}
        candidate_obs: List[Tuple[int, int, np.ndarray]] = []

        for kf in keyframes:
            for kp_idx, mp in enumerate(kf.map_points):
                if mp is not None and not mp.is_bad and not kf.outliers[kp_idx]:
                    point_obs_count[mp.id] = point_obs_count.get(mp.id, 0) + 1
                    uv = kf.kps_pts[kp_idx]
                    candidate_obs.append((kf.id, mp.id, uv))

        if len(candidate_obs) < 15 or len(point_obs_count) < 10:
            return 0

        # Sort points by observation count and select top max_points for speed
        sorted_pt_ids = sorted(point_obs_count.keys(), key=lambda pid: point_obs_count[pid], reverse=True)
        selected_pt_ids = set(sorted_pt_ids[:self.max_points])

        # Filter observations to only include selected points
        observations = [obs for obs in candidate_obs if obs[1] in selected_pt_ids]
        points_list = [slam_map.map_points[pid] for pid in selected_pt_ids if pid in slam_map.map_points]
        pt_id_to_idx = {mp.id: i for i, mp in enumerate(points_list)}

        num_cameras = len(free_kf_ids)
        num_points = len(points_list)

        if num_cameras == 0 or num_points == 0:
            return 0

        # Build parameter vector x0
        camera_params = np.empty((num_cameras, 6), dtype=np.float64)
        for i, kid in enumerate(free_kf_ids):
            kf = slam_map.keyframes[kid]
            R = kf.get_rotation()
            rvec, _ = cv2.Rodrigues(R)
            tvec = kf.get_translation().flatten()
            camera_params[i, :3] = rvec.flatten()
            camera_params[i, 3:] = tvec

        point_params = np.array([mp.point_3d for mp in points_list], dtype=np.float64)
        x0 = np.hstack((camera_params.ravel(), point_params.ravel()))

        # Sparsity pattern
        num_obs = len(observations)
        n_residuals = num_obs * 2
        n_params = len(x0)
        A = lil_matrix((n_residuals, n_params), dtype=int)

        obs_indices = []
        for i, (kid, pid, uv) in enumerate(observations):
            r_idx = 2 * i
            pt_idx = pt_id_to_idx[pid]

            pt_param_start = num_cameras * 6 + pt_idx * 3
            A[r_idx:r_idx + 2, pt_param_start:pt_param_start + 3] = 1

            if kid in kf_id_to_idx:
                cam_idx = kf_id_to_idx[kid]
                cam_param_start = cam_idx * 6
                A[r_idx:r_idx + 2, cam_param_start:cam_param_start + 6] = 1

            obs_indices.append((kid, pid, uv))

        fixed_kf = slam_map.keyframes[fixed_kf_id]
        fixed_R = fixed_kf.get_rotation()
        fixed_t = fixed_kf.get_translation().flatten()

        def residual_fun(params):
            cam_p = params[:num_cameras * 6].reshape((num_cameras, 6))
            pt_p = params[num_cameras * 6:].reshape((num_points, 3))

            residuals = np.empty(num_obs * 2, dtype=np.float64)

            for idx, (kid, pid, uv_obs) in enumerate(obs_indices):
                pt_w = pt_p[pt_id_to_idx[pid]]

                if kid == fixed_kf_id:
                    R_c = fixed_R
                    t_c = fixed_t
                else:
                    c_idx = kf_id_to_idx[kid]
                    rvec = cam_p[c_idx, :3]
                    t_c = cam_p[c_idx, 3:]
                    R_c, _ = cv2.Rodrigues(rvec)

                pt_c = R_c @ pt_w + t_c
                z = pt_c[2]

                if z > 1e-4:
                    u = camera.fx * (pt_c[0] / z) + camera.cx
                    v = camera.fy * (pt_c[1] / z) + camera.cy
                    residuals[2 * idx] = u - uv_obs[0]
                    residuals[2 * idx + 1] = v - uv_obs[1]
                else:
                    residuals[2 * idx] = 100.0
                    residuals[2 * idx + 1] = 100.0

            return residuals

        try:
            res = least_squares(
                residual_fun,
                x0,
                jac_sparsity=A,
                verbose=0,
                x_scale='jac',
                ftol=1e-3,
                method='trf',
                loss='huber',
                f_scale=2.0,
                max_nfev=self.max_iterations
            )

            opt_cams = res.x[:num_cameras * 6].reshape((num_cameras, 6))
            opt_pts = res.x[num_cameras * 6:].reshape((num_points, 3))

            for i, kid in enumerate(free_kf_ids):
                rvec = opt_cams[i, :3]
                tvec = opt_cams[i, 3:]
                R, _ = cv2.Rodrigues(rvec)
                slam_map.keyframes[kid].set_R_t(R, tvec)

            for i, mp in enumerate(points_list):
                new_pt = opt_pts[i]
                if np.all(np.isfinite(new_pt)):
                    mp.set_position(new_pt)

            return len(free_kf_ids)
        except Exception:
            return 0
