"""
KITTI Odometry Dataset loader for Monocular Visual SLAM.
Parses calibration parameters, image sequences, and ground truth poses.
"""

import os
import glob
from typing import Generator, Tuple, Optional, List
import numpy as np
import cv2
from slam.camera import Camera
from datasets.dataset_base import DatasetBase


class KITTIDataset(DatasetBase):
    def __init__(
        self,
        sequence_path: str,
        cam_id: str = "image_0",
        poses_file: Optional[str] = None
    ):
        self.sequence_path = os.path.abspath(sequence_path)
        self.cam_id = cam_id

        # 1. Resolve image directory
        img_dir = os.path.join(self.sequence_path, self.cam_id)
        if not os.path.exists(img_dir):
            # Try alternate camera folders (e.g., image_2 for color)
            for alt in ["image_2", "image_0", "image_1", "image_3"]:
                candidate = os.path.join(self.sequence_path, alt)
                if os.path.exists(candidate):
                    img_dir = candidate
                    self.cam_id = alt
                    break

        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"KITTI image folder not found in {self.sequence_path}")

        # Sorted list of image file paths
        extensions = ["*.png", "*.jpg", "*.jpeg"]
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(glob.glob(os.path.join(img_dir, ext)))
        self.image_files.sort()

        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No image files found in {img_dir}")

        # Determine image dimensions from first frame
        sample_img = cv2.imread(self.image_files[0])
        h, w = sample_img.shape[:2]

        # 2. Parse Calibration
        calib_file = os.path.join(self.sequence_path, "calib.txt")
        if not os.path.exists(calib_file):
            # Check parent directory
            calib_file = os.path.join(os.path.dirname(self.sequence_path), "calib.txt")

        if os.path.exists(calib_file):
            self.camera = self._parse_calib(calib_file, self.cam_id, w, h)
        else:
            # Standard default KITTI camera matrix
            print("[KITTI] Warning: calib.txt not found. Using default KITTI calibration.")
            self.camera = Camera(fx=718.856, fy=718.856, cx=607.1928, cy=185.2157, width=w, height=h)

        # 3. Parse Ground Truth Poses
        self.gt_poses: List[Optional[np.ndarray]] = [None] * len(self.image_files)
        seq_name = os.path.basename(os.path.normpath(self.sequence_path))

        candidate_pose_files = []
        if poses_file and os.path.exists(poses_file):
            candidate_pose_files.append(poses_file)
        # Check standard KITTI paths: dataset_root/poses/00.txt
        parent_dir = os.path.dirname(self.sequence_path)
        candidate_pose_files.append(os.path.join(parent_dir, "poses", f"{seq_name}.txt"))
        candidate_pose_files.append(os.path.join(os.path.dirname(parent_dir), "poses", f"{seq_name}.txt"))
        candidate_pose_files.append(os.path.join(self.sequence_path, f"{seq_name}.txt"))
        candidate_pose_files.append(os.path.join(self.sequence_path, "poses.txt"))

        found_pose_file = None
        for pf in candidate_pose_files:
            if os.path.exists(pf):
                found_pose_file = pf
                break

        if found_pose_file:
            print(f"[KITTI] Loading ground truth poses from: {found_pose_file}")
            self._load_poses(found_pose_file)
        else:
            print("[KITTI] Ground truth poses not found (evaluation against GT will be skipped).")

    def _parse_calib(self, calib_path: str, cam_id: str, width: int, height: int) -> Camera:
        """Parses KITTI calib.txt to extract camera projection matrix P."""
        cam_key = "P0"
        if "1" in cam_id:
            cam_key = "P1"
        elif "2" in cam_id:
            cam_key = "P2"
        elif "3" in cam_id:
            cam_key = "P3"

        with open(calib_path, 'r') as f:
            for line in f:
                if line.startswith(cam_key + ":") or line.startswith(cam_key):
                    parts = line.strip().split()
                    vals = [float(x) for x in parts[1:] if x != '']
                    P = np.array(vals, dtype=np.float64).reshape(3, 4)
                    fx = P[0, 0]
                    fy = P[1, 1]
                    cx = P[0, 2]
                    cy = P[1, 2]
                    return Camera(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)

        # Fallback
        return Camera(fx=718.856, fy=718.856, cx=607.1928, cy=185.2157, width=width, height=height)

    def _load_poses(self, pose_file: str) -> None:
        """Loads KITTI ground truth poses (12 floats per line, forming 3x4 T_wc matrix)."""
        poses = []
        with open(pose_file, 'r') as f:
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
            timestamp = idx * 0.1  # KITTI runs at ~10 Hz
            gt_pose = self.gt_poses[idx]
            yield idx, timestamp, img, gt_pose

    def get_all_groundtruth_centers(self) -> Optional[np.ndarray]:
        valid_poses = [p for p in self.gt_poses if p is not None]
        if len(valid_poses) == 0:
            return None
        centers = [p[:3, 3] for p in valid_poses]
        return np.array(centers, dtype=np.float64)
