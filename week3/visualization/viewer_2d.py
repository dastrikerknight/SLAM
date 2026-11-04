"""
Real-time 2D Visualizer for Monocular Visual SLAM using OpenCV.
Features a live Camera HUD (tracked keypoints, status, statistics)
and a Top-Down Bird's-Eye-View Trajectory Canvas.
"""

from typing import Optional, List
import time
import numpy as np
import cv2
from slam.frame import Frame
from slam.tracking import TrackingState


class Viewer2D:
    def __init__(
        self,
        canvas_size: int = 600,
        scale: float = 12.0,
        center_offset: Optional[np.ndarray] = None
    ):
        self.canvas_size = canvas_size
        self.scale = scale
        self.center_offset = center_offset if center_offset is not None else np.array([canvas_size // 2, canvas_size // 2])

        self.last_time = time.time()
        self.fps = 0.0

        # Trajectory history (X, Z coordinates in world)
        self.estimated_traj_2d: List[Tuple[int, int]] = []
        self.gt_traj_2d: List[Tuple[int, int]] = []

    def draw(
        self,
        frame: Frame,
        state: TrackingState,
        num_map_points: int,
        num_keyframes: int,
        gt_pose: Optional[np.ndarray] = None,
        wait_ms: int = 1
    ) -> int:
        """
        Draws HUD on camera frame and updates the top-down trajectory canvas.
        Returns key code pressed (e.g. 27 for ESC, ord('q') for quit).
        """
        now = time.time()
        dt = now - self.last_time
        if dt > 0.001:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_time = now

        # 1. Draw Camera HUD
        hud_img = self._render_camera_hud(frame, state, num_map_points, num_keyframes)

        # 2. Draw Top-down Trajectory
        traj_img = self._render_trajectory(frame, gt_pose)

        # Show windows
        cv2.imshow("Monocular Visual SLAM - Camera View", hud_img)
        cv2.imshow("Monocular Visual SLAM - Trajectory (Top-Down)", traj_img)

        key = cv2.waitKey(wait_ms) & 0xFF
        return key

    def _render_camera_hud(
        self,
        frame: Frame,
        state: TrackingState,
        num_map_points: int,
        num_keyframes: int
    ) -> np.ndarray:
        """Draws keypoints, status text, and diagnostic metrics on the camera frame."""
        vis = frame.image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # Draw keypoints
        for idx in range(frame.num_features):
            pt = (int(round(frame.kps_pts[idx, 0])), int(round(frame.kps_pts[idx, 1])))
            mp = frame.map_points[idx]
            is_outlier = frame.outliers[idx]

            if mp is not None and not mp.is_bad and not is_outlier:
                # Tracked 3D MapPoint (Bright Green)
                cv2.circle(vis, pt, 3, (0, 255, 0), -1)
            elif is_outlier:
                # Outlier (Red)
                cv2.drawMarker(vis, pt, (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=4, thickness=1)
            else:
                # Candidate / Untracked 2D feature (Cyan)
                cv2.circle(vis, pt, 2, (255, 200, 0), 1)

        # Status Banner
        h, w = vis.shape[:2]
        banner_h = 45
        cv2.rectangle(vis, (0, 0), (w, banner_h), (20, 20, 20), -1)

        # State text and color
        if state == TrackingState.OK:
            state_text = "STATE: TRACKING OK"
            state_color = (0, 255, 0)
        elif state == TrackingState.NOT_INITIALIZED:
            state_text = "STATE: INITIALIZING"
            state_color = (0, 220, 255)
        elif state == TrackingState.LOST:
            state_text = "STATE: TRACKING LOST"
            state_color = (0, 0, 255)
        else:
            state_text = "STATE: STARTING"
            state_color = (200, 200, 200)

        cv2.putText(vis, state_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)

        # Statistics
        inliers_count = frame.count_tracked_points()
        stats_str = f"Frame: {frame.id} | Inliers: {inliers_count} | MapPoints: {num_map_points} | KFs: {num_keyframes} | FPS: {self.fps:.1f}"
        cv2.putText(vis, stats_str, (260, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1)

        return vis

    def _render_trajectory(
        self,
        frame: Frame,
        gt_pose: Optional[np.ndarray]
    ) -> np.ndarray:
        """Renders top-down 2D camera path vs ground truth path."""
        canvas = np.full((self.canvas_size, self.canvas_size, 3), 25, dtype=np.uint8)

        # Draw grid lines
        grid_step = 50
        for i in range(0, self.canvas_size, grid_step):
            cv2.line(canvas, (i, 0), (i, self.canvas_size), (40, 40, 40), 1)
            cv2.line(canvas, (0, i), (self.canvas_size, i), (40, 40, 40), 1)

        # Center reference crosshair
        cx, cz = int(self.center_offset[0]), int(self.center_offset[1])
        cv2.drawMarker(canvas, (cx, cz), (80, 80, 80), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=1)

        # Ground Truth update
        if gt_pose is not None:
            gt_x = gt_pose[0, 3]
            gt_z = gt_pose[2, 3]
            px_gt = int(cx + gt_x * self.scale)
            pz_gt = int(cz - gt_z * self.scale)
            if 0 <= px_gt < self.canvas_size and 0 <= pz_gt < self.canvas_size:
                self.gt_traj_2d.append((px_gt, pz_gt))

        # Estimated Camera Center update
        curr_est_pt = None
        if frame.has_pose:
            cam_pos = frame.get_camera_center()
            est_x = cam_pos[0]
            est_z = cam_pos[2]
            px_est = int(cx + est_x * self.scale)
            pz_est = int(cz - est_z * self.scale)
            if 0 <= px_est < self.canvas_size and 0 <= pz_est < self.canvas_size:
                self.estimated_traj_2d.append((px_est, pz_est))
                curr_est_pt = (px_est, pz_est)

        # Draw Ground Truth Path (Green)
        for i in range(1, len(self.gt_traj_2d)):
            cv2.line(canvas, self.gt_traj_2d[i - 1], self.gt_traj_2d[i], (0, 180, 0), 2)

        # Draw Estimated Path (Cyan/Blue)
        for i in range(1, len(self.estimated_traj_2d)):
            cv2.line(canvas, self.estimated_traj_2d[i - 1], self.estimated_traj_2d[i], (255, 120, 0), 2)

        # Draw Current Camera orientation arrow
        if curr_est_pt is not None and frame.has_pose:
            R = frame.get_rotation()
            # Forward direction in world is R^T @ [0, 0, 1]
            fwd = R.T @ np.array([0.0, 0.0, 1.0])
            arrow_end = (
                int(curr_est_pt[0] + fwd[0] * 18.0),
                int(curr_est_pt[1] - fwd[2] * 18.0)
            )
            cv2.arrowedLine(canvas, curr_est_pt, arrow_end, (0, 0, 255), 2, tipLength=0.35)
            cv2.circle(canvas, curr_est_pt, 4, (0, 255, 255), -1)

        # Legend
        cv2.putText(canvas, "Top-Down View (X-Z plane)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(canvas, "- Estimated Path", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)
        if gt_pose is not None:
            cv2.putText(canvas, "- Ground Truth", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1)

        return canvas

    def close(self) -> None:
        cv2.destroyAllWindows()
