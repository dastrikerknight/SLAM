"""Single video frame with 2D keypoints, pose, and landmark associations."""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2
from slam.camera import Camera
from slam.map_point import MapPoint


class Frame:
    def __init__(
        self,
        frame_id: int,
        timestamp: float,
        image: np.ndarray,
        camera: Camera,
        kps: Optional[List[cv2.KeyPoint]] = None,
        des: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None
    ):
        self.id = frame_id
        self.timestamp = timestamp
        self.camera = camera
        self.image = image
        self.depth = depth

        # grayscale image for optical flow
        if len(image.shape) == 3:
            self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = image

        self.kps = list(kps) if kps is not None else []
        self.des = des if des is not None else np.empty((0, 32), dtype=np.uint8)
        self.num_features = len(self.kps)

        # cache Nx2 pixel coordinates for fast vector ops
        if self.kps:
            self.kps_pts = np.array([kp.pt for kp in self.kps], dtype=np.float32)
            self.kps_undistorted = self.camera.undistort_points(self.kps_pts)
        else:
            self.kps_pts = np.empty((0, 2), dtype=np.float32)
            self.kps_undistorted = np.empty((0, 2), dtype=np.float32)

        # 4x4 T_cw (world to camera)
        self.T_cw = np.eye(4, dtype=np.float64)
        self.has_pose = False

        # feature index -> MapPoint association
        self.map_points: List[Optional[MapPoint]] = [None] * self.num_features
        self.outliers = np.zeros(self.num_features, dtype=bool)
        self.is_keyframe = False

    @property
    def pose(self) -> np.ndarray:
        return self.T_cw

    @pose.setter
    def pose(self, T_cw: np.ndarray) -> None:
        self.set_pose(T_cw)

    def set_pose(self, T_cw: np.ndarray) -> None:
        self.T_cw = np.asarray(T_cw, dtype=np.float64)
        self.has_pose = True

    def set_R_t(self, R: np.ndarray, t: np.ndarray) -> None:
        self.T_cw = np.eye(4, dtype=np.float64)
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
        R = self.T_cw[:3, :3]
        t = self.T_cw[:3, 3]
        return -R.T @ t

    def get_rotation(self) -> np.ndarray:
        return self.T_cw[:3, :3]

    def get_translation(self) -> np.ndarray:
        return self.T_cw[:3, 3:4]

    def set_features(
        self,
        kps: List[cv2.KeyPoint],
        des: Optional[np.ndarray],
        map_points: Optional[List[Optional[MapPoint]]] = None
    ) -> None:
        self.kps = list(kps)
        self.des = des if des is not None else np.empty((0, 32), dtype=np.uint8)
        self.num_features = len(self.kps)
        if self.kps:
            self.kps_pts = np.array([kp.pt for kp in self.kps], dtype=np.float32)
            self.kps_undistorted = self.camera.undistort_points(self.kps_pts)
        else:
            self.kps_pts = np.empty((0, 2), dtype=np.float32)
            self.kps_undistorted = np.empty((0, 2), dtype=np.float32)

        if map_points is not None:
            self.map_points = list(map_points)
        else:
            self.map_points = [None] * self.num_features

        self.outliers = np.zeros(self.num_features, dtype=bool)

    def add_features(
        self,
        kps: List[cv2.KeyPoint],
        des: Optional[np.ndarray],
        map_points: Optional[List[Optional[MapPoint]]] = None
    ) -> None:
        if not kps:
            return

        n_new = len(kps)
        self.kps.extend(kps)
        if des is not None and len(des) > 0:
            if self.des is not None and len(self.des) > 0:
                self.des = np.vstack((self.des, des))
            else:
                self.des = des

        new_pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        new_undist = self.camera.undistort_points(new_pts)

        if len(self.kps_pts) > 0:
            self.kps_pts = np.vstack((self.kps_pts, new_pts))
            self.kps_undistorted = np.vstack((self.kps_undistorted, new_undist))
        else:
            self.kps_pts = new_pts
            self.kps_undistorted = new_undist

        if map_points is not None:
            self.map_points.extend(map_points)
        else:
            self.map_points.extend([None] * n_new)

        self.outliers = np.concatenate([self.outliers, np.zeros(n_new, dtype=bool)])
        self.num_features = len(self.kps)

    def associate_map_point(self, idx: int, map_point: MapPoint) -> None:
        if 0 <= idx < self.num_features:
            self.map_points[idx] = map_point
            self.outliers[idx] = False

    def remove_map_point(self, idx: int) -> None:
        if 0 <= idx < self.num_features:
            self.map_points[idx] = None

    def set_outlier(self, idx: int, is_outlier: bool = True) -> None:
        if 0 <= idx < self.num_features:
            self.outliers[idx] = is_outlier

    def get_matched_3d_2d(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        pts_3d = []
        pts_2d = []
        indices = []

        for idx, mp in enumerate(self.map_points):
            if mp is not None and not mp.is_bad and not self.outliers[idx]:
                pts_3d.append(mp.point_3d)
                pts_2d.append(self.kps_pts[idx])
                indices.append(idx)

        if not pts_3d:
            return np.empty((0, 3)), np.empty((0, 2)), []

        return np.array(pts_3d, dtype=np.float64), np.array(pts_2d, dtype=np.float32), indices

    def count_tracked_points(self) -> int:
        return sum(1 for idx, mp in enumerate(self.map_points) if mp is not None and not mp.is_bad and not self.outliers[idx])

    def is_visible_in_image(self, pt_3d: np.ndarray, margin: int = 10) -> Tuple[bool, Optional[np.ndarray]]:
        R = self.T_cw[:3, :3]
        t = self.T_cw[:3, 3]
        pt_c = R @ pt_3d + t

        if pt_c[2] <= 0.1:
            return False, None

        uv, valid = self.camera.project(pt_c.reshape(1, 3))
        if not valid[0]:
            return False, None

        u, v = uv[0]
        h, w = self.image.shape[:2]
        if margin <= u < (w - margin) and margin <= v < (h - margin):
            return True, uv[0]

        return False, None

    def __repr__(self) -> str:
        pts = self.count_tracked_points()
        return f"Frame(id={self.id}, ts={self.timestamp:.3f}, kps={self.num_features}, tracked={pts}, kf={self.is_keyframe})"
