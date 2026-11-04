from typing import Dict, List, Tuple
import numpy as np
import cv2
from scipy.optimize import least_squares

class PoseGraphOptimizer:
    def __init__(self, max_iterations: int = 25):
        self.max_iterations = max_iterations

    def optimize(
        self,
        keyframe_poses: Dict[int, np.ndarray],
        loop_edges: List[Tuple[int, int, np.ndarray, float]]
    ) -> Dict[int, np.ndarray]:
        kf_ids = sorted(keyframe_poses.keys())
        if len(kf_ids) < 3:
            return keyframe_poses

        id_to_idx = {kid: i for i, kid in enumerate(kf_ids)}
        n_kfs = len(kf_ids)

        # Build sequential odometry edges
        edges = []
        for i in range(n_kfs - 1):
            id_a, id_b = kf_ids[i], kf_ids[i + 1]
            Ta, Tb = keyframe_poses[id_a], keyframe_poses[id_b]
            # Relative motion from a to b: T_ba = Tb @ inv(Ta)
            T_ba = Tb @ np.linalg.inv(Ta)
            edges.append((i, i + 1, T_ba, 1.0))

        # Add loop closure edges
        for id_a, id_b, T_meas, weight in loop_edges:
            if id_a in id_to_idx and id_b in id_to_idx:
                edges.append((id_to_idx[id_a], id_to_idx[id_b], T_meas, weight))

        # Flatten initial state: keyframe 0 is fixed at origin, optimize kf 1..N-1
        # Each pose has 6 DOF: 3 translation, 3 rodrigues rotation
        params0 = np.zeros((n_kfs - 1) * 6, dtype=np.float64)
        for i in range(1, n_kfs):
            T = keyframe_poses[kf_ids[i]]
            rvec, _ = cv2.Rodrigues(T[:3, :3])
            params0[(i - 1) * 6 : (i - 1) * 6 + 3] = T[:3, 3]
            params0[(i - 1) * 6 + 3 : (i - 1) * 6 + 6] = rvec.ravel()

        T0 = keyframe_poses[kf_ids[0]]

        def unpack_pose(idx: int, p: np.ndarray) -> np.ndarray:
            if idx == 0:
                return T0
            sub = p[(idx - 1) * 6 : (idx - 1) * 6 + 6]
            T = np.eye(4, dtype=np.float64)
            T[:3, 3] = sub[:3]
            T[:3, :3], _ = cv2.Rodrigues(sub[3:6])
            return T

        def residual_fn(p: np.ndarray) -> np.ndarray:
            res_list = []
            for idx_a, idx_b, T_meas, weight in edges:
                Ta = unpack_pose(idx_a, p)
                Tb = unpack_pose(idx_b, p)
                # Error: T_err = T_meas^-1 @ (Tb @ inv(Ta))
                T_est = Tb @ np.linalg.inv(Ta)
                T_err = np.linalg.inv(T_meas) @ T_est

                t_err = T_err[:3, 3]
                rvec_err, _ = cv2.Rodrigues(T_err[:3, :3])
                w = np.sqrt(weight)
                res_list.extend([
                    t_err[0] * w, t_err[1] * w, t_err[2] * w,
                    rvec_err[0, 0] * w * 2.0, rvec_err[1, 0] * w * 2.0, rvec_err[2, 0] * w * 2.0
                ])
            return np.array(res_list, dtype=np.float64)

        res = least_squares(
            residual_fn,
            params0,
            method="trf",
            loss="cauchy",
            f_scale=0.1,
            max_nfev=self.max_iterations
        )

        optimized = {kf_ids[0]: T0.copy()}
        for i in range(1, n_kfs):
            optimized[kf_ids[i]] = unpack_pose(i, res.x)

        return optimized
