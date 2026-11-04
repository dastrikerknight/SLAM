"""
Camera model for Monocular Visual SLAM.
Handles intrinsic parameters, projection, unprojection, and distortion correction.
"""

from typing import Tuple, Optional
import numpy as np
import cv2


class Camera:
    """
    Pinhole camera model with optional radial-tangential distortion parameters.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        k1: float = 0.0,
        k2: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0,
        k3: float = 0.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = width
        self.height = height

        # Intrinsic matrix K (3x3)
        self.K = np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0]
        ], dtype=np.float64)

        self.K_inv = np.linalg.inv(self.K)

        # Distortion coefficients (k1, k2, p1, p2, k3)
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        self.has_distortion = np.any(np.abs(self.dist_coeffs) > 1e-7)

        # Precompute undistortion rectification maps if resolution is known
        self._map1 = None
        self._map2 = None
        if self.has_distortion and self.width is not None and self.height is not None:
            self._init_undistort_maps(self.width, self.height)

    def _init_undistort_maps(self, width: int, height: int) -> None:
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, None, self.K, (width, height), cv2.CV_16SC2
        )

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Undistorts an entire image using precomputed rectification maps."""
        if not self.has_distortion:
            return image
        h, w = image.shape[:2]
        if self._map1 is None or (self.width != w or self.height != h):
            self.width = w
            self.height = h
            self._init_undistort_maps(w, h)
        return cv2.remap(image, self._map1, self._map2, cv2.INTER_LINEAR)

    @classmethod
    def from_intrinsics_matrix(
        cls,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> 'Camera':
        """Construct Camera from 3x3 intrinsic matrix."""
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        if dist_coeffs is not None and len(dist_coeffs) >= 4:
            k1 = dist_coeffs[0]
            k2 = dist_coeffs[1]
            p1 = dist_coeffs[2]
            p2 = dist_coeffs[3]
            k3 = dist_coeffs[4] if len(dist_coeffs) > 4 else 0.0
            return cls(fx, fy, cx, cy, k1, k2, p1, p2, k3, width, height)
        return cls(fx, fy, cx, cy, width=width, height=height)

    def project(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects 3D points in camera coordinates (N, 3) to 2D image coordinates (N, 2).
        Returns:
            uv (N, 2): pixel coordinates.
            valid_mask (N,): boolean mask of points strictly in front of camera (Z > 0).
        """
        points_3d = np.asarray(points_3d, dtype=np.float64)
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, 3)

        z = points_3d[:, 2]
        valid_mask = z > 1e-4

        # Avoid division by zero
        safe_z = np.where(valid_mask, z, 1.0)
        x = points_3d[:, 0] / safe_z
        y = points_3d[:, 1] / safe_z

        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        uv = np.column_stack((u, v))

        if self.width is not None and self.height is not None:
            in_bounds = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
            valid_mask = valid_mask & in_bounds

        return uv, valid_mask

    def unproject(self, uv: np.ndarray, depth: float = 1.0) -> np.ndarray:
        """
        Unprojects 2D pixel coordinates (N, 2) to 3D rays in camera coordinates (N, 3).
        """
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim == 1:
            uv = uv.reshape(1, 2)

        x = (uv[:, 0] - self.cx) / self.fx
        y = (uv[:, 1] - self.cy) / self.fy
        z = np.ones_like(x)
        rays = np.column_stack((x, y, z)) * depth
        return rays

    def undistort_points(self, points_2d: np.ndarray) -> np.ndarray:
        """
        Undistorts 2D points using camera distortion coefficients.
        """
        if not self.has_distortion:
            return points_2d

        pts = np.asarray(points_2d, dtype=np.float32).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(pts, self.K, self.dist_coeffs, P=self.K)
        return undistorted.reshape(-1, 2)

    def is_in_image(self, uv: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """
        Checks if 2D points are within camera image boundaries.
        """
        if self.width is None or self.height is None:
            return np.ones(len(uv), dtype=bool)

        u = uv[:, 0]
        v = uv[:, 1]
        return (u >= margin) & (u < self.width - margin) & (v >= margin) & (v < self.height - margin)

    def __repr__(self) -> str:
        return f"Camera(fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}, resolution={self.width}x{self.height})"
