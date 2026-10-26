from typing import Optional
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
        # If camera has distortion, undistort at frame ingestion and track with ideal pinhole
        self.camera = Camera(camera.fx, camera.fy, camera.cx, camera.cy, width=camera.width, height=camera.height) if camera.has_distortion else camera

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

    def process_frame(self, frame_id: int, ts: float, img: np.ndarray) -> Frame:
        if self.raw_camera.has_distortion:
            img = self.raw_camera.undistort_image(img)
        return self.tracking.process_frame(frame_id, ts, img)

    @property
    def state(self) -> TrackingState:
        return self.tracking.state

    @property
    def num_inliers(self) -> int:
        return self.tracking.num_inliers

    def get_trajectory_centers(self) -> np.ndarray:
        return self.map.get_trajectory_centers()

    def get_all_map_points(self) -> np.ndarray:
        return self.map.get_all_points_3d()

    def reset(self):
        self.map.clear()
        self.tracking.state = TrackingState.NOT_INITIALIZED
        self.tracking.init_ref_frame = None
        self.tracking.last_frame = None
        self.tracking.curr_frame = None
        self.tracking.last_keyframe = None
        self.tracking.has_velocity = False
