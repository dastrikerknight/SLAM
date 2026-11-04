"""
Base interface for visual SLAM datasets.
"""

from abc import ABC, abstractmethod
from typing import Generator, Tuple, Optional
import numpy as np
from slam.camera import Camera


class DatasetBase(ABC):
    @abstractmethod
    def get_camera(self) -> Camera:
        """Returns the camera calibration model for this dataset."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns total number of frames in the dataset."""
        pass

    @abstractmethod
    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray, Optional[np.ndarray]], None, None]:
        """
        Yields (frame_id, timestamp, image, groundtruth_pose_4x4_or_None).
        groundtruth_pose is the camera-to-world (T_wc) 4x4 matrix.
        """
        pass

    def get_all_groundtruth_centers(self) -> Optional[np.ndarray]:
        """Returns (N, 3) ground truth camera positions if available."""
        return None
