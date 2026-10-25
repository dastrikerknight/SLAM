from typing import Tuple, Optional
import numpy as np
import cv2

class Camera:
    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 k1: float = 0.0, k2: float = 0.0, p1: float = 0.0, p2: float = 0.0, k3: float = 0.0,
                 width: Optional[int] = None, height: Optional[int] = None):
        self.fx, self.fy = float(fx), float(fy)
        self.cx, self.cy = float(cx), float(cy)
        self.width, self.height = width, height

        self.K = np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0]
        ], dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)

        self.dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        self.has_distortion = np.any(np.abs(self.dist_coeffs) > 1e-7)

        self._map1 = None
        self._map2 = None
        if self.has_distortion and self.width and self.height:
            self._init_maps(self.width, self.height)

    def _init_maps(self, w: int, h: int):
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, None, self.K, (w, h), cv2.CV_16SC2
        )

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        if not self.has_distortion:
            return img
        h, w = img.shape[:2]
        if self._map1 is None or self.width != w or self.height != h:
            self.width, self.height = w, h
            self._init_maps(w, h)
        return cv2.remap(img, self._map1, self._map2, cv2.INTER_LINEAR)

    def project(self, pts_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(pts_3d, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)

        z = pts[:, 2]
        valid = z > 1e-4

        # avoid zero division on invalid/behind-camera points
        safe_z = np.where(valid, z, 1.0)
        u = self.fx * (pts[:, 0] / safe_z) + self.cx
        v = self.fy * (pts[:, 1] / safe_z) + self.cy
        uv = np.column_stack((u, v))

        if self.width and self.height:
            valid &= (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)

        return uv, valid

    def unproject(self, uv: np.ndarray, depth: float = 1.0) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim == 1:
            uv = uv.reshape(1, 2)
        x = (uv[:, 0] - self.cx) / self.fx
        y = (uv[:, 1] - self.cy) / self.fy
        return np.column_stack((x, y, np.ones_like(x))) * depth

    def undistort_points(self, pts_2d: np.ndarray) -> np.ndarray:
        if not self.has_distortion:
            return pts_2d
        p = np.asarray(pts_2d, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.undistortPoints(p, self.K, self.dist_coeffs, P=self.K).reshape(-1, 2)

    def __repr__(self) -> str:
        return f"Camera({self.fx:.1f}, {self.fy:.1f}, {self.cx:.1f}, {self.cy:.1f})"
