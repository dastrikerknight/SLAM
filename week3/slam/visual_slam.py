"""
VisualSLAM top-level orchestrator for Monocular Visual SLAM.
Integrates feature tracking, two-view initialization, local mapping, and bundle adjustment.
"""

from typing import Optional, List, Tuple
import numpy as np
from slam.camera import Camera
from slam.map import Map
from slam.frame import Frame
from slam.feature_tracker import FeatureTracker
from slam.initializer import Initializer
from slam.mapping import LocalMapping
from slam.tracking import Tracking, TrackingState


class VisualSLAM:
    def __init__(
        self,
        camera: Camera,
        num_features: int = 1500,
        min_inliers_pnp: int = 10,
        min_parallax_deg: float = 2.5
    ):
        self.raw_camera = camera
        # When camera has distortion, frames are rectified at ingestion,
        # so internal geometry operates on an ideal pinhole model with zero distortion.
        if camera.has_distortion:
            self.camera = Camera(camera.fx, camera.fy, camera.cx, camera.cy, width=camera.width, height=camera.height)
        else:
            self.camera = camera

        self.map = Map()
        self.feature_tracker = FeatureTracker(num_features=num_features)
        self.initializer = Initializer(
            min_matches=50,
            min_inliers=35,
            min_parallax_deg=min_parallax_deg,
            max_reprojection_err=2.5
        )
        self.local_mapping = LocalMapping(
            camera=self.camera,
            feature_tracker=self.feature_tracker,
            min_parallax_deg=1.2,
            max_reprojection_err=2.5
        )
        self.tracking = Tracking(
            camera=self.camera,
            slam_map=self.map,
            feature_tracker=self.feature_tracker,
            initializer=self.initializer,
            local_mapping=self.local_mapping,
            min_inliers_pnp=min_inliers_pnp
        )

    def process_frame(
        self,
        frame_id: int,
        timestamp: float,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None
    ) -> Frame:
        """
        Feeds a single image frame (and optional depth map) to the SLAM pipeline.
        Returns the processed Frame with pose and point associations.
        """
        if self.raw_camera.has_distortion:
            image = self.raw_camera.undistort_image(image)
        return self.tracking.process_frame(frame_id, timestamp, image, depth=depth)

    @property
    def state(self) -> TrackingState:
        return self.tracking.state

    @property
    def num_inliers(self) -> int:
        return self.tracking.num_inliers

    def get_trajectory_centers(self) -> np.ndarray:
        """Returns (N, 3) estimated camera positions in world coordinates."""
        return self.map.get_trajectory_centers()

    def get_all_map_points(self) -> np.ndarray:
        """Returns (M, 3) active 3D MapPoint world coordinates."""
        return self.map.get_all_points_3d()

    def get_all_map_colors(self) -> np.ndarray:
        """Returns (M, 3) RGB colors of MapPoints."""
        return self.map.get_all_points_colors()

    def reset(self) -> None:
        """Resets the SLAM system."""
        self.map.clear()
        self.tracking.state = TrackingState.NOT_INITIALIZED
        self.tracking.init_reference_frame = None
        self.tracking.last_frame = None
        self.tracking.current_frame = None
        self.tracking.last_keyframe = None
        self.tracking.has_velocity = False
