from typing import Dict, Optional, List
import numpy as np
import cv2

class MapPoint:
    _counter = 0

    def __init__(self, pt_3d: np.ndarray, first_frame_id: int,
                 color: Optional[np.ndarray] = None, descriptor: Optional[np.ndarray] = None):
        MapPoint._counter += 1
        self.id = MapPoint._counter
        self.point_3d = np.asarray(pt_3d, dtype=np.float64).flatten()
        self.first_frame_id = first_frame_id
        self.color = color if color is not None else np.array([200, 200, 200], dtype=np.uint8)

        self.observations: Dict[int, int] = {}
        self.num_visible = 1
        self.num_found = 1
        self.is_bad = False
        self.descriptor = descriptor

    @property
    def x(self) -> float: return float(self.point_3d[0])
    @property
    def y(self) -> float: return float(self.point_3d[1])
    @property
    def z(self) -> float: return float(self.point_3d[2])

    def set_position(self, pos: np.ndarray):
        self.point_3d = np.asarray(pos, dtype=np.float64).flatten()

    def add_observation(self, frame_id: int, kp_idx: int):
        self.observations[frame_id] = kp_idx

    def remove_observation(self, frame_id: int):
        self.observations.pop(frame_id, None)
        if len(self.observations) < 2:
            self.is_bad = True

    def get_num_observations(self) -> int:
        return len(self.observations)

    def mark_bad(self):
        self.is_bad = True
        self.observations.clear()

    def update_descriptor(self, des_list: List[np.ndarray]):
        if not des_list:
            return
        if len(des_list) == 1:
            self.descriptor = des_list[0]
            return

        mat = np.vstack(des_list)
        n = len(des_list)
        dists = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                d = cv2.norm(mat[i], mat[j], cv2.NORM_HAMMING)
                dists[i, j] = dists[j, i] = d
        self.descriptor = des_list[int(np.argmin(np.median(dists, axis=1)))]

    def __repr__(self) -> str:
        return f"MapPoint(id={self.id}, obs={len(self.observations)})"
