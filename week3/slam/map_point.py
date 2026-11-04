"""
3D MapPoint representation for Visual Monocular SLAM.
Tracks 3D world coordinates, viewing directions, observations across keyframes,
and representative ORB descriptors.
"""

from typing import Dict, Optional, List
import numpy as np
import cv2


class MapPoint:
    _id_counter = 0

    def __init__(
        self,
        point_3d: np.ndarray,
        first_frame_id: int = 0,
        color: Optional[np.ndarray] = None,
        descriptor: Optional[np.ndarray] = None
    ):
        MapPoint._id_counter += 1
        self.id = MapPoint._id_counter
        self.point_3d = np.asarray(point_3d, dtype=np.float64).flatten()

        # Defensive check for argument order
        if isinstance(first_frame_id, np.ndarray) and descriptor is None:
            descriptor = first_frame_id
            first_frame_id = 0

        self.first_frame_id = int(first_frame_id) if isinstance(first_frame_id, (int, np.integer)) else 0
        self.color = color if color is not None else np.array([200, 200, 200], dtype=np.uint8)

        # Observations mapping: frame_id -> keypoint_index
        self.observations: Dict[int, int] = {}

        # Tracking statistics
        self.num_visible: int = 1
        self.num_found: int = 1
        self.is_bad: bool = False

        # Representative descriptor and viewing direction
        self.descriptor: Optional[np.ndarray] = descriptor
        self.normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Min and max distances from which the point can be reliably observed
        self.min_distance: float = 0.0
        self.max_distance: float = 1e6

    @property
    def x(self) -> float:
        return self.point_3d[0]

    @property
    def y(self) -> float:
        return self.point_3d[1]

    @property
    def z(self) -> float:
        return self.point_3d[2]

    def set_position(self, pos_3d: np.ndarray) -> None:
        """Update 3D world position."""
        self.point_3d = np.asarray(pos_3d, dtype=np.float64).flatten()

    def add_observation(self, frame_id: int, keypoint_idx: int) -> None:
        """Record observation of this map point in a keyframe."""
        self.observations[frame_id] = keypoint_idx

    def remove_observation(self, frame_id: int) -> None:
        """Remove observation from a keyframe."""
        if frame_id in self.observations:
            del self.observations[frame_id]
            if len(self.observations) < 2:
                # Monocular points need at least 2 observations to be constrained
                self.is_bad = True

    def get_num_observations(self) -> int:
        return len(self.observations)

    def mark_bad(self) -> None:
        self.is_bad = True
        self.observations.clear()

    def update_descriptor(self, all_descriptors: List[np.ndarray]) -> None:
        """
        Computes the representative descriptor by selecting the one with
        the minimum median distance to all other observed descriptors.
        """
        if not all_descriptors:
            return

        if len(all_descriptors) == 1:
            self.descriptor = all_descriptors[0]
            return

        desc_mat = np.vstack(all_descriptors)
        # Compute pairwise Hamming distances
        n = len(all_descriptors)
        distances = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                dist = cv2.norm(desc_mat[i], desc_mat[j], cv2.NORM_HAMMING)
                distances[i, j] = dist
                distances[j, i] = dist

        median_dists = np.median(distances, axis=1)
        best_idx = int(np.argmin(median_dists))
        self.descriptor = all_descriptors[best_idx]

    def update_normal_and_depth(self, keyframes: Dict[int, 'Frame']) -> None:
        """
        Updates mean viewing direction normal and valid depth scale bounds
        based on observing keyframes.
        """
        if not self.observations:
            return

        normal = np.zeros(3, dtype=np.float64)
        n = 0
        distances = []

        for frame_id in self.observations:
            if frame_id in keyframes:
                kf = keyframes[frame_id]
                cam_center = kf.get_camera_center()
                ray = self.point_3d - cam_center
                dist = np.linalg.norm(ray)
                if dist > 1e-4:
                    normal += ray / dist
                    distances.append(dist)
                    n += 1

        if n > 0:
            norm_val = np.linalg.norm(normal)
            if norm_val > 1e-6:
                self.normal = normal / norm_val
            if distances:
                mean_dist = float(np.mean(distances))
                self.min_distance = mean_dist * 0.5
                self.max_distance = mean_dist * 2.0

    def __repr__(self) -> str:
        return f"MapPoint(id={self.id}, pos=({self.x:.2f}, {self.y:.2f}, {self.z:.2f}), obs={len(self.observations)})"
