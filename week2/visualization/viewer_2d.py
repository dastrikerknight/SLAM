from typing import Optional, List, Tuple
import time
import numpy as np
import cv2
from slam.frame import Frame
from slam.tracking import TrackingState

class Viewer2D:
    def __init__(self, canvas_size: int = 600, scale: float = 12.0, center_offset: Optional[np.ndarray] = None):
        self.canvas_size = canvas_size
        self.scale = scale
        self.center_offset = center_offset if center_offset is not None else np.array([canvas_size // 2, canvas_size // 2])
        self.last_time = time.time()
        self.fps = 0.0

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
        now = time.time()
        dt = now - self.last_time
        if dt > 0.001:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_time = now

        hud_img = self._render_camera_hud(frame, state, num_map_points, num_keyframes)
        traj_img = self._render_trajectory(frame, gt_pose)

        cv2.imshow("Monocular Visual SLAM - Camera View", hud_img)
        cv2.imshow("Monocular Visual SLAM - Trajectory (Top-Down)", traj_img)
        return cv2.waitKey(wait_ms) & 0xFF

    def _render_camera_hud(self, frame: Frame, state: TrackingState, num_map_points: int, num_keyframes: int) -> np.ndarray:
        vis = frame.image.copy() if frame.image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        for idx in range(frame.num_features):
            pt = (int(round(frame.kps_pts[idx, 0])), int(round(frame.kps_pts[idx, 1])))
            mp = frame.map_points[idx]
            is_outlier = frame.outliers[idx]

            if mp is not None and not mp.is_bad and not is_outlier:
                cv2.circle(vis, pt, 3, (0, 255, 0), -1)
            elif is_outlier:
                cv2.drawMarker(vis, pt, (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=4, thickness=1)
            else:
                cv2.circle(vis, pt, 2, (255, 200, 0), 1)

        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, 45), (20, 20, 20), -1)

        if state == TrackingState.OK:
            state_text, state_color = "STATE: TRACKING OK", (0, 255, 0)
        elif state == TrackingState.NOT_INITIALIZED:
            state_text, state_color = "STATE: INITIALIZING", (0, 220, 255)
        elif state == TrackingState.LOST:
            state_text, state_color = "STATE: TRACKING LOST", (0, 0, 255)
        else:
            state_text, state_color = "STATE: STARTING", (200, 200, 200)

        cv2.putText(vis, state_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)
        stats = f"Frame: {frame.id} | Inliers: {frame.count_tracked_points()} | Points: {num_map_points} | KFs: {num_keyframes} | FPS: {self.fps:.1f}"
        cv2.putText(vis, stats, (260, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1)
        return vis

    def _render_trajectory(self, frame: Frame, gt_pose: Optional[np.ndarray]) -> np.ndarray:
        canvas = np.full((self.canvas_size, self.canvas_size, 3), 25, dtype=np.uint8)

        grid_step = 50
        for i in range(0, self.canvas_size, grid_step):
            cv2.line(canvas, (i, 0), (i, self.canvas_size), (40, 40, 40), 1)
            cv2.line(canvas, (0, i), (self.canvas_size, i), (40, 40, 40), 1)

        cx, cz = int(self.center_offset[0]), int(self.center_offset[1])
        cv2.drawMarker(canvas, (cx, cz), (80, 80, 80), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=1)

        if gt_pose is not None:
            px_gt = int(cx + gt_pose[0, 3] * self.scale)
            pz_gt = int(cz - gt_pose[2, 3] * self.scale)
            if 0 <= px_gt < self.canvas_size and 0 <= pz_gt < self.canvas_size:
                self.gt_traj_2d.append((px_gt, pz_gt))

        curr_est_pt = None
        if frame.has_pose:
            pos = frame.get_camera_center()
            px_est = int(cx + pos[0] * self.scale)
            pz_est = int(cz - pos[2] * self.scale)
            if 0 <= px_est < self.canvas_size and 0 <= pz_est < self.canvas_size:
                self.estimated_traj_2d.append((px_est, pz_est))
                curr_est_pt = (px_est, pz_est)

        for i in range(1, len(self.gt_traj_2d)):
            cv2.line(canvas, self.gt_traj_2d[i - 1], self.gt_traj_2d[i], (0, 180, 0), 2)

        for i in range(1, len(self.estimated_traj_2d)):
            cv2.line(canvas, self.estimated_traj_2d[i - 1], self.estimated_traj_2d[i], (255, 120, 0), 2)

        if curr_est_pt is not None and frame.has_pose:
            fwd = frame.get_rotation().T @ np.array([0.0, 0.0, 1.0])
            arrow_end = (int(curr_est_pt[0] + fwd[0] * 18.0), int(curr_est_pt[1] - fwd[2] * 18.0))
            cv2.arrowedLine(canvas, curr_est_pt, arrow_end, (0, 0, 255), 2, tipLength=0.35)
            cv2.circle(canvas, curr_est_pt, 4, (0, 255, 255), -1)

        cv2.putText(canvas, "Top-Down (X-Z)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(canvas, "- Estimated", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)
        if gt_pose is not None:
            cv2.putText(canvas, "- Ground Truth", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1)

        return canvas

    def close(self):
        cv2.destroyAllWindows()
