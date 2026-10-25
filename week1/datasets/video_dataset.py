from typing import Generator, Tuple, Optional, Union
import numpy as np
import cv2
from slam.camera import Camera
from datasets.dataset_base import DatasetBase

class VideoDataset(DatasetBase):
    def __init__(
        self,
        video_source: Union[str, int],
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        max_frames: Optional[int] = None
    ):
        self.video_source = video_source
        self.max_frames = max_frames

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.video_source}")

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 30.0

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            self.total_frames = 1000
        if self.max_frames:
            self.total_frames = min(self.total_frames, self.max_frames)

        cap.release()

        if fx is None or fy is None:
            diag = np.sqrt(self.width**2 + self.height**2)
            self.fx = self.fy = diag * 0.8
        else:
            self.fx, self.fy = float(fx), float(fy)

        self.cx = float(cx) if cx is not None else self.width / 2.0
        self.cy = float(cy) if cy is not None else self.height / 2.0

        self.camera = Camera(
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            width=self.width, height=self.height
        )

    def get_camera(self) -> Camera:
        return self.camera

    def __len__(self) -> int:
        return self.total_frames

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray, Optional[np.ndarray]], None, None]:
        cap = cv2.VideoCapture(self.video_source)
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            yield frame_id, frame_id / self.fps, frame, None
            frame_id += 1

            if self.max_frames and frame_id >= self.max_frames:
                break

        cap.release()
