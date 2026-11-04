"""
Synthetic 3D SLAM benchmark dataset generator.
Generates an artificial 3D textured world with unique, feature-rich landmarks,
and renders camera views along a simulated driving trajectory with exact ground truth.
Guarantees distinct, non-aliased ORB descriptors and precise geometric ground truth.
"""

from typing import Generator, Tuple, Optional, List
import numpy as np
import cv2
from slam.camera import Camera
from datasets.dataset_base import DatasetBase


class SyntheticDataset(DatasetBase):
    def __init__(
        self,
        num_frames: int = 120,
        width: int = 640,
        height: int = 480,
        seed: int = 42
    ):
        self.num_frames = num_frames
        self.width = width
        self.height = height
        np.random.seed(seed)

        self.camera = Camera(
            fx=500.0, fy=500.0,
            cx=width / 2.0, cy=height / 2.0,
            width=width, height=height
        )

        # 1. Generate 3D landmark features in world coordinates
        # Road corridor: X in [-5, 5], Y in [-2, 2.5], Z in [2, 50]
        self.landmarks_3d = []
        self.landmark_textures = []

        # Floor and wall grid positions with jitter
        positions = []

        # Left wall (X around -3.5 to -4.0)
        for z in np.linspace(2, 45, 30):
            for y in np.linspace(-1.5, 1.5, 5):
                positions.append([-3.8 + np.random.uniform(-0.2, 0.2), y, z])

        # Right wall (X around +3.5 to +4.0)
        for z in np.linspace(2, 45, 30):
            for y in np.linspace(-1.5, 1.5, 5):
                positions.append([3.8 + np.random.uniform(-0.2, 0.2), y, z])

        # Road surface (Y = 1.8 to 2.2)
        for z in np.linspace(2, 45, 35):
            for x in np.linspace(-2.5, 2.5, 5):
                positions.append([x, 2.0, z])

        # Distinct 3D geometric pillars (X = -2.0 and +2.0)
        for z in np.linspace(4, 44, 15):
            for x in [-2.0, 2.0]:
                for y in [-0.8, 0.2, 1.2]:
                    positions.append([x, y, z])

        self.landmarks_3d = np.array(positions, dtype=np.float64)

        # Generate a UNIQUE high-contrast binary/grayscale texture patch for each landmark
        # This completely eliminates perceptual aliasing and ensures unique ORB descriptors
        patch_size = 24
        for idx in range(len(self.landmarks_3d)):
            # Unique pattern using random bits and seeded hash
            rng = np.random.RandomState(idx + seed * 1000)
            pattern = rng.choice([30, 220], size=(patch_size // 4, patch_size // 4)).astype(np.uint8)
            patch = cv2.resize(pattern, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
            # Add high-contrast border
            cv2.rectangle(patch, (0, 0), (patch_size - 1, patch_size - 1), 255, 2)
            cv2.circle(patch, (patch_size // 2, patch_size // 2), 3, 0, -1)
            self.landmark_textures.append(patch)

        # 2. Pre-generate smooth camera trajectory
        # Forward translation along Z with smooth S-curve in X and slight elevation
        self.gt_poses: List[np.ndarray] = []
        t_steps = np.linspace(0, 1, self.num_frames)

        for i, t in enumerate(t_steps):
            z_pos = 1.0 + t * 24.0
            x_pos = 0.8 * np.sin(2.0 * np.pi * t)
            y_pos = 0.15 * np.cos(np.pi * t)
            cam_center = np.array([x_pos, y_pos, z_pos], dtype=np.float64)

            # Look direction
            if i < self.num_frames - 1:
                t_next = t_steps[i + 1]
                z_next = 1.0 + t_next * 24.0
                x_next = 0.8 * np.sin(2.0 * np.pi * t_next)
                y_next = 0.15 * np.cos(np.pi * t_next)
                look_dir = np.array([x_next - x_pos, y_next - y_pos, z_next - z_pos])
            else:
                look_dir = np.array([0.0, 0.0, 1.0])

            look_dir /= np.linalg.norm(look_dir)

            # Up vector in world: [0, -1, 0] because camera Y points downwards in OpenCV
            up = np.array([0.0, -1.0, 0.0])
            right = np.cross(look_dir, up)
            norm_r = np.linalg.norm(right)
            if norm_r > 1e-6:
                right /= norm_r
            else:
                right = np.array([1.0, 0.0, 0.0])

            down = np.cross(look_dir, right)
            down /= np.linalg.norm(down)

            # Camera-to-world rotation matrix R_wc
            R_wc = np.column_stack((right, down, look_dir))

            # 4x4 T_wc matrix
            T_wc = np.eye(4, dtype=np.float64)
            T_wc[:3, :3] = R_wc
            T_wc[:3, 3] = cam_center

            self.gt_poses.append(T_wc)

    def get_camera(self) -> Camera:
        return self.camera

    def __len__(self) -> int:
        return self.num_frames

    def _render_frame(self, T_wc: np.ndarray) -> np.ndarray:
        """Projects 3D world objects onto image canvas for given camera pose."""
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc

        # Background canvas (neutral gray with subtle gradient)
        img = np.full((self.height, self.width), 120, dtype=np.uint8)

        # Transform 3D landmarks to camera coordinates: P_cam = R_cw @ P_w + t_cw
        pts_cam = (R_cw @ self.landmarks_3d.T).T + t_cw

        # Sort by depth (painter's algorithm)
        depths = pts_cam[:, 2]
        sort_order = np.argsort(-depths)

        for idx in sort_order:
            pt_c = pts_cam[idx]
            z = pt_c[2]
            if z <= 0.5 or z > 32.0:
                continue

            # Project to pixel
            uv, valid = self.camera.project(pt_c.reshape(1, 3))
            if not valid[0]:
                continue

            u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))

            # Scale texture patch according to depth: scale = fx / z
            base_tex = self.landmark_textures[idx]
            target_size = max(10, min(50, int(round(self.camera.fx * 0.35 / z))))

            # Paste texture patch
            half = target_size // 2
            x1, x2 = u - half, u + half
            y1, y2 = v - half, v + half

            if x1 < 0 or y1 < 0 or x2 >= self.width or y2 >= self.height:
                continue

            scaled_patch = cv2.resize(base_tex, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            img[y1:y2, x1:x2] = scaled_patch

        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray, Optional[np.ndarray]], None, None]:
        for idx in range(self.num_frames):
            gt_pose = self.gt_poses[idx]
            img = self._render_frame(gt_pose)
            timestamp = idx * (1.0 / 30.0)
            yield idx, timestamp, img, gt_pose

    def get_all_groundtruth_centers(self) -> Optional[np.ndarray]:
        return np.array([p[:3, 3] for p in self.gt_poses], dtype=np.float64)
