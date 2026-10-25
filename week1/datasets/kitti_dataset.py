import os
import glob
from typing import Generator, Tuple, Optional, List
import numpy as np
import cv2
from slam.camera import Camera
from datasets.dataset_base import DatasetBase

class KITTIDataset(DatasetBase):
    def __init__(self, sequence_path: str, cam_id: str = "image_0", poses_file: Optional[str] = None):
        self.sequence_path = os.path.abspath(sequence_path)
        self.cam_id = cam_id

        img_dir = os.path.join(self.sequence_path, self.cam_id)
        if not os.path.exists(img_dir):
            for alt in ["image_2", "image_0", "image_1", "image_3"]:
                candidate = os.path.join(self.sequence_path, alt)
                if os.path.exists(candidate):
                    img_dir = candidate
                    self.cam_id = alt
                    break

        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"KITTI image folder not found in {self.sequence_path}")

        extensions = ["*.png", "*.jpg", "*.jpeg"]
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(glob.glob(os.path.join(img_dir, ext)))
        self.image_files.sort()

        if not self.image_files:
            raise FileNotFoundError(f"No image files found in {img_dir}")

        sample_img = cv2.imread(self.image_files[0])
        h, w = sample_img.shape[:2]

        calib_file = os.path.join(self.sequence_path, "calib.txt")
        if not os.path.exists(calib_file):
            calib_file = os.path.join(os.path.dirname(self.sequence_path), "calib.txt")

        if os.path.exists(calib_file):
            self.camera = self._parse_calib(calib_file, self.cam_id, w, h)
        else:
            self.camera = Camera(fx=718.856, fy=718.856, cx=607.1928, cy=185.2157, width=w, height=h)

        self.gt_poses: List[Optional[np.ndarray]] = [None] * len(self.image_files)
        seq_name = os.path.basename(os.path.normpath(self.sequence_path))
        parent_dir = os.path.dirname(self.sequence_path)

        candidates = [
            poses_file,
            os.path.join(parent_dir, "poses", f"{seq_name}.txt"),
            os.path.join(os.path.dirname(parent_dir), "poses", f"{seq_name}.txt"),
            os.path.join(self.sequence_path, f"{seq_name}.txt"),
            os.path.join(self.sequence_path, "poses.txt")
        ]
        found = next((p for p in candidates if p and os.path.exists(p)), None)
        if found:
            self._load_poses(found)

    def _parse_calib(self, calib_path: str, cam_id: str, width: int, height: int) -> Camera:
        cam_key = "P0"
        if "1" in cam_id: cam_key = "P1"
        elif "2" in cam_id: cam_key = "P2"
        elif "3" in cam_id: cam_key = "P3"

        with open(calib_path, "r") as f:
            for line in f:
                if line.startswith(cam_key + ":") or line.startswith(cam_key):
                    parts = line.strip().split()
                    vals = [float(x) for x in parts[1:] if x != ""]
                    P = np.array(vals, dtype=np.float64).reshape(3, 4)
                    return Camera(fx=P[0, 0], fy=P[1, 1], cx=P[0, 2], cy=P[1, 2], width=width, height=height)

        return Camera(fx=718.856, fy=718.856, cx=607.1928, cy=185.2157, width=width, height=height)

    def _load_poses(self, pose_file: str):
        poses = []
        with open(pose_file, "r") as f:
            for line in f:
                vals = [float(x) for x in line.strip().split()]
                if len(vals) == 12:
                    P = np.array(vals, dtype=np.float64).reshape(3, 4)
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :] = P
                    poses.append(T)

        for i in range(min(len(poses), len(self.gt_poses))):
            self.gt_poses[i] = poses[i]

    def get_camera(self) -> Camera:
        return self.camera

    def __len__(self) -> int:
        return len(self.image_files)

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray, Optional[np.ndarray]], None, None]:
        for idx, file_path in enumerate(self.image_files):
            img = cv2.imread(file_path)
            yield idx, idx * 0.1, img, self.gt_poses[idx]

    def get_all_groundtruth_centers(self) -> Optional[np.ndarray]:
        valid = [p for p in self.gt_poses if p is not None]
        return np.array([p[:3, 3] for p in valid], dtype=np.float64) if valid else None
