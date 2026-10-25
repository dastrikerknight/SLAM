from typing import Dict, List, Tuple
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from slam.camera import Camera
from slam.map import Map

class LocalBundleAdjustment:
    def __init__(self, window_size: int = 5, max_iterations: int = 10, max_points: int = 120):
        self.window_size = window_size
        self.max_iterations = max_iterations
        self.max_points = max_points

    def optimize(self, slam_map: Map, camera: Camera) -> int:
        all_kf_ids = sorted(list(slam_map.keyframes.keys()))
        if len(all_kf_ids) < 2:
            return 0

        window_kf_ids = all_kf_ids[-self.window_size:]
        keyframes = [slam_map.keyframes[kid] for kid in window_kf_ids]

        # anchor first keyframe to remove gauge freedom
        fixed_kf_id = window_kf_ids[0]
        free_kf_ids = window_kf_ids[1:]
        kf_idx_map = {kid: i for i, kid in enumerate(free_kf_ids)}

        point_obs: Dict[int, int] = {}
        candidate_obs: List[Tuple[int, int, np.ndarray]] = []

        for kf in keyframes:
            for kp_idx, mp in enumerate(kf.map_points):
                if mp and not mp.is_bad and not kf.outliers[kp_idx]:
                    point_obs[mp.id] = point_obs.get(mp.id, 0) + 1
                    candidate_obs.append((kf.id, mp.id, kf.kps_pts[kp_idx]))

        if len(candidate_obs) < 15 or len(point_obs) < 10:
            return 0

        top_pts = set(sorted(point_obs.keys(), key=lambda p: point_obs[p], reverse=True)[:self.max_points])
        obs = [o for o in candidate_obs if o[1] in top_pts]
        points = [slam_map.map_points[p] for p in top_pts if p in slam_map.map_points]
        pt_idx_map = {mp.id: i for i, mp in enumerate(points)}

        num_cams = len(free_kf_ids)
        num_pts = len(points)
        if num_cams == 0 or num_pts == 0:
            return 0

        cam_params = np.empty((num_cams, 6), dtype=np.float64)
        for i, kid in enumerate(free_kf_ids):
            kf = slam_map.keyframes[kid]
            rvec, _ = cv2.Rodrigues(kf.get_rotation())
            cam_params[i, :3] = rvec.flatten()
            cam_params[i, 3:] = kf.get_translation().flatten()

        pt_params = np.array([mp.point_3d for mp in points], dtype=np.float64)
        x0 = np.hstack((cam_params.ravel(), pt_params.ravel()))

        n_obs = len(obs)
        A = lil_matrix((n_obs * 2, len(x0)), dtype=int)
        for i, (kid, pid, _) in enumerate(obs):
            r = 2 * i
            pt_idx = pt_idx_map[pid]
            p_start = num_cams * 6 + pt_idx * 3
            A[r:r + 2, p_start:p_start + 3] = 1
            if kid in kf_idx_map:
                c_start = kf_idx_map[kid] * 6
                A[r:r + 2, c_start:c_start + 6] = 1

        fixed_kf = slam_map.keyframes[fixed_kf_id]
        fixed_R = fixed_kf.get_rotation()
        fixed_t = fixed_kf.get_translation().flatten()

        def residual_fn(params):
            c_p = params[:num_cams * 6].reshape((num_cams, 6))
            p_p = params[num_cams * 6:].reshape((num_pts, 3))
            res = np.empty(n_obs * 2, dtype=np.float64)

            for idx, (kid, pid, uv_obs) in enumerate(obs):
                pw = p_p[pt_idx_map[pid]]
                if kid == fixed_kf_id:
                    Rc, tc = fixed_R, fixed_t
                else:
                    c = kf_idx_map[kid]
                    Rc, _ = cv2.Rodrigues(c_p[c, :3])
                    tc = c_p[c, 3:]

                pc = Rc @ pw + tc
                z = pc[2]
                if z > 1e-4:
                    res[2 * idx] = camera.fx * (pc[0] / z) + camera.cx - uv_obs[0]
                    res[2 * idx + 1] = camera.fy * (pc[1] / z) + camera.cy - uv_obs[1]
                else:
                    res[2 * idx] = res[2 * idx + 1] = 100.0
            return res

        try:
            sol = least_squares(
                residual_fn, x0, jac_sparsity=A,
                method="trf", loss="huber", f_scale=2.0,
                max_nfev=self.max_iterations, ftol=1e-3
            )
            opt_c = sol.x[:num_cams * 6].reshape((num_cams, 6))
            opt_p = sol.x[num_cams * 6:].reshape((num_pts, 3))

            for i, kid in enumerate(free_kf_ids):
                R, _ = cv2.Rodrigues(opt_c[i, :3])
                slam_map.keyframes[kid].set_R_t(R, opt_c[i, 3:])

            for i, mp in enumerate(points):
                if np.all(np.isfinite(opt_p[i])):
                    mp.set_position(opt_p[i])
            return len(free_kf_ids)
        except Exception:
            return 0
