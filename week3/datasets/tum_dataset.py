"""
TUM RGB-D Dataset loader for Monocular Visual SLAM.
Loads monocular RGB frames, parses ground truth trajectories, and synchronizes timestamps.
"""

import os
from typing import Generator, Tuple, Optional, List
import numpy as np
import cv2
from slam.camera import Camera
from datasets.dataset_base import DatasetBase


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Converts [qx, qy, qz, qw] quaternion to 3x3 rotation matrix."""
    qx, qy, qz, qw = q
    # Normalize
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-6:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n

    R = np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw),       2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw),       1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw),       2.0 * (qy * qz + qx * qw),       1.0 - 2.0 * (qx * qx + qy * qy)]
    ], dtype=np.float64)
    return R


class TUMDataset(DatasetBase):
    def __init__(
        self,
        sequence_path: str,
        freiburg_version: int = 1,
        is_rgbd: bool = True
    ):
        self.sequence_path = os.path.abspath(sequence_path)
        self.is_rgbd = is_rgbd

        # Intrinsics for TUM Freiburg series
        if freiburg_version == 1:
            self.camera = Camera(
                fx=517.3, fy=516.5, cx=318.6, cy=255.3,
                k1=0.2624, k2=-0.9531, p1=-0.0054, p2=0.0026, k3=1.1633,
                width=640, height=480
            )
        elif freiburg_version == 2:
            self.camera = Camera(
                fx=520.9, fy=521.0, cx=325.1, cy=249.7,
                k1=0.2312, k2=-0.7849, p1=-0.0033, p2=-0.0001, k3=0.9172,
                width=640, height=480
            )
        else:
            self.camera = Camera(
                fx=535.4, fy=539.2, cx=320.1, cy=247.6,
                width=640, height=480
            )

        # 1. Parse rgb.txt
        rgb_file = os.path.join(self.sequence_path, "rgb.txt")
        if not os.path.exists(rgb_file):
            raise FileNotFoundError(f"TUM rgb.txt not found in {self.sequence_path}")

        self.rgb_entries: List[Tuple[float, str]] = []  # (timestamp, relative_path)
        with open(rgb_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    ts = float(parts[0])
                    img_rel = parts[1]
                    self.rgb_entries.append((ts, img_rel))

        if len(self.rgb_entries) == 0:
            raise ValueError(f"No valid entries found in {rgb_file}")

        # 2. Parse groundtruth.txt (if available)
        gt_file = os.path.join(self.sequence_path, "groundtruth.txt")
        self.gt_poses: List[Optional[np.ndarray]] = [None] * len(self.rgb_entries)

        if os.path.exists(gt_file):
            gt_data = []
            with open(gt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = [float(x) for x in line.split()]
                    if len(parts) >= 8:
                        ts = parts[0]
                        t = np.array(parts[1:4], dtype=np.float64)
                        q = np.array(parts[4:8], dtype=np.float64)  # qx, qy, qz, qw
                        R = quaternion_to_rotation_matrix(q)
                        T = np.eye(4, dtype=np.float64)
                        T[:3, :3] = R
                        T[:3, 3] = t
                        gt_data.append((ts, T))

            # Associate closest GT pose within 0.02s
            if gt_data:
                gt_timestamps = np.array([x[0] for x in gt_data])
                for idx, (rgb_ts, _) in enumerate(self.rgb_entries):
                    diffs = np.abs(gt_timestamps - rgb_ts)
                    closest_idx = int(np.argmin(diffs))
                    if diffs[closest_idx] < 0.02:
                        self.gt_poses[idx] = gt_data[closest_idx][1]

        # 3. Parse depth.txt (if available)
        depth_file = os.path.join(self.sequence_path, "depth.txt")
        self.depth_entries: List[Optional[str]] = [None] * len(self.rgb_entries)
        if os.path.exists(depth_file):
            depth_data = []
            with open(depth_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        depth_data.append((float(parts[0]), parts[1]))
            if depth_data:
                depth_timestamps = np.array([x[0] for x in depth_data])
                depth_paths = [x[1] for x in depth_data]
                for idx, (rgb_ts, _) in enumerate(self.rgb_entries):
                    diffs = np.abs(depth_timestamps - rgb_ts)
                    closest_idx = int(np.argmin(diffs))
                    if diffs[closest_idx] < 0.05:
                        self.depth_entries[idx] = depth_paths[closest_idx]

        # In RGB-D mode, synchronize by keeping only frames with valid depth
        if self.is_rgbd and any(d is not None for d in self.depth_entries):
            synced_indices = [i for i, d in enumerate(self.depth_entries) if d is not None]
            self.rgb_entries = [self.rgb_entries[i] for i in synced_indices]
            self.depth_entries = [self.depth_entries[i] for i in synced_indices]
            self.gt_poses = [self.gt_poses[i] for i in synced_indices]

    def get_camera(self) -> Camera:
        return self.camera

    def __len__(self) -> int:
        return len(self.rgb_entries)

    def __iter__(self):
        for idx, (ts, rel_path) in enumerate(self.rgb_entries):
            full_path = os.path.join(self.sequence_path, rel_path)
            img = cv2.imread(full_path)
            if img is None:
                continue

            gt_pose = self.gt_poses[idx]

            if self.is_rgbd:
                depth_m = None
                if idx < len(self.depth_entries) and self.depth_entries[idx] is not None:
                    depth_full = os.path.join(self.sequence_path, self.depth_entries[idx])
                    depth_raw = cv2.imread(depth_full, cv2.IMREAD_UNCHANGED)
                    if depth_raw is not None:
                        depth_m = depth_raw.astype(np.float32) / 5000.0
                yield idx, ts, img, depth_m, gt_pose
            else:
                yield idx, ts, img, gt_pose

    def get_all_groundtruth_centers(self) -> Optional[np.ndarray]:
        valid_poses = [p for p in self.gt_poses if p is not None]
        if not valid_poses:
            return None
        return np.array([p[:3, 3] for p in valid_poses], dtype=np.float64)
