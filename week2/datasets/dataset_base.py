from abc import ABC, abstractmethod
from typing import Generator, Tuple, Optional
import numpy as np
from slam.camera import Camera

class DatasetBase(ABC):
    @abstractmethod
    def get_camera(self) -> Camera:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray, Optional[np.ndarray]], None, None]:
        pass

    def get_all_groundtruth_centers(self) -> Optional[np.ndarray]:
        return None
