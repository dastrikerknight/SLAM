import os
import glob
import re
from typing import Generator, Tuple, Optional, List
import numpy as np
import cv2
from slam.camera import Camera
from datasets.dataset_base import DatasetBase

def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class ImageFolderDataset(DatasetBase):
    def __init__(
        self,
        folder_path: str,
        fps: float = 30.0,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        max_frames: Optional[int] = None
    ):
        self.folder_path = os.path.abspath(folder_path)
        self.fps = fps
        self.max_frames = max_frames

        if not os.path.isdir(self.folder_path):
            raise NotADirectoryError(f"Directory not found: {self.folder_path}")

        extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.PNG", "*.JPG", "*.JPEG"]
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(glob.glob(os.path.join(self.folder_path, ext)))

        if not self.image_files:
            for sub in ["images", "image_0", "image_2", "rgb", "data", "frames"]:
                sub_dir = os.path.join(self.folder_path, sub)
                if os.path.isdir(sub_dir):
                    for ext in extensions:
                        self.image_files.extend(glob.glob(os.path.join(sub_dir, ext)))
                    if self.image_files:
                        break

        self.image_files.sort(key=natural_sort_key)
        if not self.image_files:
            raise FileNotFoundError(f"No image files found in {self.folder_path}")

        if self.max_frames:
            self.image_files = self.image_files[:self.max_frames]

        sample_img = cv2.imread(self.image_files[0])
        if sample_img is None:
            raise ValueError(f"Unable to read first image: {self.image_files[0]}")
        h, w = sample_img.shape[:2]
        self.width = w
        self.height = h

        calib_file = os.path.join(self.folder_path, "calib.txt")
        if os.path.exists(calib_file) and (fx is None or fy is None):
            self.camera = self._parse_calib(calib_file, w, h)
        else:
            if fx is None or fy is None:
                diag = np.sqrt(w**2 + h**2)
                self.fx = self.fy = diag * 0.85
            else:
                self.fx, self.fy = float(fx), float(fy)

            self.cx = float(cx) if cx is not None else w / 2.0
            self.cy = float(cy) if cy is not None else h / 2.0
            self.camera = Camera(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, width=w, height=h)

    def _parse_calib(self, calib_path: str, w: int, h: int) -> Camera:
        try:
            with open(calib_path, "r") as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                vals = []
                for line in lines:
                    for token in line.replace(",", " ").split():
                        try:
                            vals.append(float(token))
                        except ValueError:
                            pass
                if len(vals) >= 4:
                    return Camera(fx=vals[0], fy=vals[1], cx=vals[2], cy=vals[3], width=w, height=h)
        except Exception:
            pass
        diag = np.sqrt(w**2 + h**2)
        return Camera(fx=diag * 0.85, fy=diag * 0.85, cx=w / 2.0, cy=h / 2.0, width=w, height=h)

    def get_camera(self) -> Camera:
        return self.camera

    def __len__(self) -> int:
        return len(self.image_files)

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray, Optional[np.ndarray]], None, None]:
        for idx, file_path in enumerate(self.image_files):
            img = cv2.imread(file_path)
            if img is not None:
                yield idx, idx * (1.0 / self.fps), img, None
