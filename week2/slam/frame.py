from typing import List, Optional
import numpy as np
import cv2
from slam.camera import Camera
from slam.map_point import MapPoint

class Frame:
    def __init__(self, frame_id: int, ts: float, img: np.ndarray, cam: Camera,
                 kps: Optional[List[cv2.KeyPoint]] = None, des: Optional[np.ndarray] = None):
        self.id = frame_id
        self.timestamp = ts
        self.camera = cam
        self.image = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        self.kps = list(kps) if kps else []
        self.des = des if des is not None else np.empty((0, 32), dtype=np.uint8)
        self.num_features = len(self.kps)

        if self.kps:
            self.kps_pts = np.array([kp.pt for kp in self.kps], dtype=np.float32)
            self.kps_undistorted = self.camera.undistort_points(self.kps_pts)
        else:
            self.kps_pts = np.empty((0, 2), dtype=np.float32)
            self.kps_undistorted = np.empty((0, 2), dtype=np.float32)

        self.T_cw = np.eye(4, dtype=np.float64)
        self.has_pose = False
        self.is_keyframe = False

        self.map_points: List[Optional[MapPoint]] = [None] * self.num_features
        self.outliers = np.zeros(self.num_features, dtype=bool)

    def set_pose(self, T_cw: np.ndarray):
        self.T_cw = np.asarray(T_cw, dtype=np.float64)
        self.has_pose = True

    def set_R_t(self, R: np.ndarray, t: np.ndarray):
        self.T_cw[:3, :3] = R
        self.T_cw[:3, 3] = t.flatten()
        self.has_pose = True

    def get_pose(self) -> np.ndarray:
        return self.T_cw

    def get_inverse_pose(self) -> np.ndarray:
        R = self.T_cw[:3, :3]
        t = self.T_cw[:3, 3]
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = R.T
        T_wc[:3, 3] = -R.T @ t
        return T_wc

    def get_camera_center(self) -> np.ndarray:
        return -self.T_cw[:3, :3].T @ self.T_cw[:3, 3]

    def get_rotation(self) -> np.ndarray:
        return self.T_cw[:3, :3]

    def get_translation(self) -> np.ndarray:
        return self.T_cw[:3, 3].reshape(3, 1)

    def associate_map_point(self, idx: int, mp: MapPoint):
        if 0 <= idx < self.num_features:
            self.map_points[idx] = mp
            mp.add_observation(self.id, idx)

    def set_features(self, kps: List[cv2.KeyPoint], des: np.ndarray,
                     mps: Optional[List[Optional[MapPoint]]] = None):
        self.kps = list(kps) if kps else []
        self.des = des if des is not None else np.empty((0, 32), dtype=np.uint8)
        self.num_features = len(self.kps)

        if self.kps:
            self.kps_pts = np.array([kp.pt for kp in self.kps], dtype=np.float32)
            self.kps_undistorted = self.camera.undistort_points(self.kps_pts)
        else:
            self.kps_pts = np.empty((0, 2), dtype=np.float32)
            self.kps_undistorted = np.empty((0, 2), dtype=np.float32)

        self.map_points = list(mps) if (mps and len(mps) == self.num_features) else [None] * self.num_features
        self.outliers = np.zeros(self.num_features, dtype=bool)

    def add_features(self, new_kps: List[cv2.KeyPoint], new_des: np.ndarray,
                     new_mps: Optional[List[Optional[MapPoint]]] = None):
        if not new_kps:
            return
        n = len(new_kps)
        self.kps.extend(new_kps)
        self.des = new_des if len(self.des) == 0 else np.vstack((self.des, new_des))

        pts = np.array([kp.pt for kp in new_kps], dtype=np.float32)
        undist = self.camera.undistort_points(pts)

        if len(self.kps_pts) == 0:
            self.kps_pts, self.kps_undistorted = pts, undist
        else:
            self.kps_pts = np.vstack((self.kps_pts, pts))
            self.kps_undistorted = np.vstack((self.kps_undistorted, undist))

        self.num_features = len(self.kps)
        self.map_points.extend(new_mps if (new_mps and len(new_mps) == n) else [None] * n)
        self.outliers = np.concatenate((self.outliers, np.zeros(n, dtype=bool)))

    def count_tracked_points(self) -> int:
        return sum(1 for idx, mp in enumerate(self.map_points) if mp and not mp.is_bad and not self.outliers[idx])

    def __repr__(self) -> str:
        pos = self.get_camera_center()
        return f"Frame(id={self.id}, tracked={self.count_tracked_points()}, pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}])"
