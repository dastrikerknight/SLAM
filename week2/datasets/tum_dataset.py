import os
from typing import Generator, Tuple, Optional, List
import numpy as np
import cv2
from slam.camera import Camera
from datasets.dataset_base import DatasetBase

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-6:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n

    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw),       2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw),       1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw),       2.0 * (qy * qz + qx * qw),       1.0 - 2.0 * (qx * qx + qy * qy)]
    ], dtype=np.float64)

class TUMDataset(DatasetBase):
    def __init__(self, sequence_path: str, freiburg_version: int = 1):
        self.sequence_path = os.path.abspath(sequence_path)

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

        rgb_file = os.path.join(self.sequence_path, "rgb.txt")
        if not os.path.exists(rgb_file):
            raise FileNotFoundError(f"TUM rgb.txt not found in {self.sequence_path}")

        self.rgb_entries: List[Tuple[float, str]] = []
        with open(rgb_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    self.rgb_entries.append((float(parts[0]), parts[1]))

        if not self.rgb_entries:
            raise ValueError(f"No valid entries found in {rgb_file}")

        gt_file = os.path.join(self.sequence_path, "groundtruth.txt")
        self.gt_poses: List[Optional[np.ndarray]] = [None] * len(self.rgb_entries)

        if os.path.exists(gt_file):
            gt_data = []
            with open(gt_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = [float(x) for x in line.split()]
                    if len(parts) >= 8:
                        ts = parts[0]
                        t = np.array(parts[1:4], dtype=np.float64)
                        q = np.array(parts[4:8], dtype=np.float64)
                        T = np.eye(4, dtype=np.float64)
                        T[:3, :3] = quaternion_to_rotation_matrix(q)
                        T[:3, 3] = t
                        gt_data.append((ts, T))

            if gt_data:
                gt_timestamps = np.array([x[0] for x in gt_data])
                for idx, (rgb_ts, _) in enumerate(self.rgb_entries):
                    diffs = np.abs(gt_timestamps - rgb_ts)
                    closest_idx = int(np.argmin(diffs))
                    if diffs[closest_idx] < 0.02:
                        self.gt_poses[idx] = gt_data[closest_idx][1]

    def get_camera(self) -> Camera:
        return self.camera

    def __len__(self) -> int:
        return len(self.rgb_entries)

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray, Optional[np.ndarray]], None, None]:
        for idx, (ts, rel_path) in enumerate(self.rgb_entries):
            img = cv2.imread(os.path.join(self.sequence_path, rel_path))
            if img is not None:
                yield idx, ts, img, self.gt_poses[idx]

    def get_all_groundtruth_centers(self) -> Optional[np.ndarray]:
        valid_poses = [p for p in self.gt_poses if p is not None]
        return np.array([p[:3, 3] for p in valid_poses], dtype=np.float64) if valid_poses else None
