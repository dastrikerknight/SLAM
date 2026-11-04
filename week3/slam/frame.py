"""
Frame representation for Monocular Visual SLAM.
Maintains 2D keypoints, ORB descriptors, camera pose, grayscale image, and 2D-3D MapPoint associations.
"""

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

        # Store grayscale image for optical flow
        if len(image.shape) == 3:
            self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = image

        self.kps = list(kps) if kps is not None else []
        self.des = des if des is not None else np.empty((0, 32), dtype=np.uint8)
        self.num_features = len(self.kps)

        # Extract (N, 2) numpy array of pixel coordinates
        if self.kps:
            self.kps_pts = np.array([kp.pt for kp in self.kps], dtype=np.float32)
            self.kps_undistorted = self.camera.undistort_points(self.kps_pts)
        else:
            self.kps_pts = np.empty((0, 2), dtype=np.float32)
            self.kps_undistorted = np.empty((0, 2), dtype=np.float32)

        # 4x4 Transformation matrix T_cw (World to Camera)
        self.T_cw = np.eye(4, dtype=np.float64)
        self.has_pose = False

        # Associations: 1-to-1 matching with keypoint index -> MapPoint or None
        self.map_points: List[Optional[MapPoint]] = [None] * self.num_features

        # Outlier mask
        self.outliers = np.zeros(self.num_features, dtype=bool)

        self.is_keyframe = False

    @property
    def pose(self) -> np.ndarray:
        return self.T_cw

    @pose.setter
    def pose(self, T_cw: np.ndarray) -> None:
        self.set_pose(T_cw)

    def set_pose(self, T_cw: np.ndarray) -> None:
        """Set 4x4 transformation matrix (world to camera)."""
        self.T_cw = np.asarray(T_cw, dtype=np.float64)
        self.has_pose = True

    def set_R_t(self, R: np.ndarray, t: np.ndarray) -> None:
        """Set pose from 3x3 rotation and 3x1 translation."""
        self.T_cw = np.eye(4, dtype=np.float64)
        self.T_cw[:3, :3] = R
        self.T_cw[:3, 3] = t.flatten()
        self.has_pose = True

    def get_pose(self) -> np.ndarray:
        """Returns 4x4 matrix T_cw (World to Camera)."""
        return self.T_cw

    def get_inverse_pose(self) -> np.ndarray:
        """Returns 4x4 matrix T_wc (Camera to World)."""
        R = self.T_cw[:3, :3]
        t = self.T_cw[:3, 3]
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = R.T
        T_wc[:3, 3] = -R.T @ t
        return T_wc

    def get_camera_center(self) -> np.ndarray:
        """Returns camera center in world coordinates (3,)."""
        R = self.T_cw[:3, :3]
        t = self.T_cw[:3, 3]
        return -R.T @ t

    def get_rotation(self) -> np.ndarray:
        """Returns 3x3 rotation matrix R_cw."""
        return self.T_cw[:3, :3]

    def get_translation(self) -> np.ndarray:
        """Returns 3x1 translation vector t_cw."""
        return self.T_cw[:3, 3].reshape(3, 1)

    def associate_map_point(self, kp_idx: int, map_point: MapPoint) -> None:
        """Link a keypoint in this frame with a 3D MapPoint."""
        if 0 <= kp_idx < self.num_features:
            self.map_points[kp_idx] = map_point
            map_point.add_observation(self.id, kp_idx)

    def set_features(
        self,
        kps: List[cv2.KeyPoint],
        des: np.ndarray,
        map_points: Optional[List[Optional[MapPoint]]] = None
    ) -> None:
        """Sets or replaces the features in this frame."""
        self.kps = list(kps) if kps is not None else []
        self.des = des if des is not None else np.empty((0, 32), dtype=np.uint8)
        self.num_features = len(self.kps)

        if self.kps:
            self.kps_pts = np.array([kp.pt for kp in self.kps], dtype=np.float32)
            self.kps_undistorted = self.camera.undistort_points(self.kps_pts)
        else:
            self.kps_pts = np.empty((0, 2), dtype=np.float32)
            self.kps_undistorted = np.empty((0, 2), dtype=np.float32)

        if map_points is not None and len(map_points) == self.num_features:
            self.map_points = list(map_points)
        else:
            self.map_points = [None] * self.num_features

        self.outliers = np.zeros(self.num_features, dtype=bool)

    def add_features(
        self,
        new_kps: List[cv2.KeyPoint],
        new_des: np.ndarray,
        new_map_points: Optional[List[Optional[MapPoint]]] = None
    ) -> None:
        """Appends new features to this frame."""
        if not new_kps:
            return

        new_kps = list(new_kps)
        n_new = len(new_kps)
        self.kps.extend(new_kps)
        if len(self.des) == 0:
            self.des = new_des
        else:
            self.des = np.vstack((self.des, new_des))

        new_pts = np.array([kp.pt for kp in new_kps], dtype=np.float32)
        new_undist = self.camera.undistort_points(new_pts)

        if len(self.kps_pts) == 0:
            self.kps_pts = new_pts
            self.kps_undistorted = new_undist
        else:
            self.kps_pts = np.vstack((self.kps_pts, new_pts))
            self.kps_undistorted = np.vstack((self.kps_undistorted, new_undist))

        self.num_features = len(self.kps)

        if new_map_points is not None and len(new_map_points) == n_new:
            self.map_points.extend(new_map_points)
        else:
            self.map_points.extend([None] * n_new)

        self.outliers = np.concatenate((self.outliers, np.zeros(n_new, dtype=bool)))

        # Cap feature count at maximum 1500 to prevent unbounded memory growth
        max_feat_cap = 1500
        if self.num_features > max_feat_cap:
            self.kps = self.kps[:max_feat_cap]
            self.des = self.des[:max_feat_cap]
            self.kps_pts = self.kps_pts[:max_feat_cap]
            self.kps_undistorted = self.kps_undistorted[:max_feat_cap]
            self.map_points = self.map_points[:max_feat_cap]
            self.outliers = self.outliers[:max_feat_cap]
            self.num_features = max_feat_cap

    def get_matched_3d_2d(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Gathers all keypoints that have valid associated 3D MapPoints.
        Returns:
            points_3d (M, 3): world 3D coordinates.
            points_2d (M, 2): 2D pixel coordinates in this frame.
            kp_indices: list of corresponding keypoint indices in this frame.
        """
        pts_3d = []
        pts_2d = []
        kp_indices = []

        for idx, mp in enumerate(self.map_points):
            if mp is not None and not mp.is_bad and not self.outliers[idx]:
                pts_3d.append(mp.point_3d)
                pts_2d.append(self.kps_pts[idx])
                kp_indices.append(idx)

        if pts_3d:
            return np.array(pts_3d, dtype=np.float64), np.array(pts_2d, dtype=np.float32), kp_indices
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float32), []

    def count_tracked_points(self) -> int:
        """Counts how many valid map points are currently observed."""
        count = 0
        for idx, mp in enumerate(self.map_points):
            if mp is not None and not mp.is_bad and not self.outliers[idx]:
                count += 1
        return count

    def project_point(self, point_3d_world: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Projects a 3D point in world coordinates to pixel coordinates in this frame."""
        pt_w = np.asarray(point_3d_world, dtype=np.float64).reshape(3, 1)
        R = self.T_cw[:3, :3]
        t = self.T_cw[:3, 3].reshape(3, 1)
        pt_cam = (R @ pt_w + t).flatten()

        uv, valid = self.camera.project(pt_cam.reshape(1, 3))
        return uv[0], bool(valid[0])

    def __repr__(self) -> str:
        cam_pos = self.get_camera_center()
        return f"Frame(id={self.id}, features={self.num_features}, tracked={self.count_tracked_points()}, pos=({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}))"
