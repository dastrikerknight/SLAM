import queue
import threading
from typing import Optional, Tuple
import numpy as np
import cv2

from depth_module import MonocularDepthEstimator
from scale_aligner import align_depth_scale_shift
from tsdf_mapper import TSDFMapper
from slam.frame import Frame
from slam.map import Map

class KeyframeManager:
    def __init__(
        self,
        translation_threshold: float = 0.15,
        depth_estimator: Optional[MonocularDepthEstimator] = None,
        tsdf_mapper: Optional[TSDFMapper] = None,
        max_queue_size: int = 4
    ):
        self.translation_threshold = translation_threshold
        self.depth_estimator = depth_estimator or MonocularDepthEstimator()
        self.tsdf_mapper = tsdf_mapper or TSDFMapper()

        self.last_kf_pos: Optional[np.ndarray] = None
        self.num_keyframes_triggered = 0
        self.num_keyframes_integrated = 0

        # Background worker for asynchronous depth inference and TSDF fusion
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def process_frame(self, frame: Frame, slam_map: Map) -> bool:
        if not frame.has_pose:
            return False

        curr_pos = frame.get_camera_center()
        if not np.all(np.isfinite(curr_pos)):
            return False

        # Check translation displacement from last integrated keyframe
        if self.last_kf_pos is not None:
            dist = float(np.linalg.norm(curr_pos - self.last_kf_pos))
            if dist < self.translation_threshold:
                return False

        # Project active SLAM map points into the image
        pixel_coords, slam_depths = self._get_visible_depth_associations(frame, slam_map)
        if len(pixel_coords) < 3:
            return False

        self.last_kf_pos = curr_pos.copy()
        self.num_keyframes_triggered += 1

        # Package data snapshot to decouple background mapping from live tracker modifications
        img_copy = frame.image.copy() if frame.image is not None else cv2.cvtColor(frame.gray, cv2.COLOR_GRAY2BGR)
        job = {
            "img": img_copy,
            "pixel_coords": pixel_coords,
            "slam_depths": slam_depths,
            "K": frame.camera.K.copy(),
            "pose_c2w": frame.get_inverse_pose().copy(),
            "frame_id": frame.id
        }

        try:
            self.queue.put_nowait(job)
            return True
        except queue.Full:
            # Drop frame if worker is saturated to preserve main loop tracking FPS
            return False

    def _get_visible_depth_associations(self, frame: Frame, slam_map: Map) -> Tuple[np.ndarray, np.ndarray]:
        pts_3d_inliers, pts_2d_inliers, _ = frame.get_matched_3d_2d()
        pixel_coords = []
        slam_depths = []

        R_cw = frame.T_cw[:3, :3]
        t_cw = frame.T_cw[:3, 3]

        # Feature inliers actively tracked in the current frame (guaranteed visible and unoccluded)
        if len(pts_3d_inliers) > 0:
            pts_c = (R_cw @ pts_3d_inliers.T).T + t_cw
            for i in range(len(pts_c)):
                zc = float(pts_c[i, 2])
                if 0.1 < zc < 10.0 and np.isfinite(zc):
                    pixel_coords.append(pts_2d_inliers[i])
                    slam_depths.append(zc)

        if len(pixel_coords) < 3:
            return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32)

        return np.array(pixel_coords, dtype=np.float32), np.array(slam_depths, dtype=np.float32)

    def _worker_loop(self):
        while not self.stop_event.is_set():
            try:
                job = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # 1. Depth inference on current frame
                pred_depth = self.depth_estimator.predict_depth(job["img"])

                # 2 & 3. Align predicted depth with sparse SLAM metric scale and shift
                metric_depth, inlier_mask = align_depth_scale_shift(
                    job["pixel_coords"],
                    job["slam_depths"],
                    pred_depth
                )

                # 4. Integrate into volumetric TSDF grid
                self.tsdf_mapper.integrate_keyframe(
                    color_img=job["img"],
                    metric_depth_map=metric_depth,
                    intrinsic_matrix=job["K"],
                    pose_c2w=job["pose_c2w"]
                )
                self.num_keyframes_integrated += 1
            except Exception as e:
                print(f"[KeyframeManager] Integration error on frame {job.get('frame_id')}: {e}")
            finally:
                self.queue.task_done()

    def finalize(self, mesh_filename: str = "dense_tsdf_mesh.ply"):
        # Drain all remaining jobs in the queue before mesh extraction
        self.queue.join()
        mesh = self.tsdf_mapper.extract_and_save_mesh(mesh_filename)
        if len(mesh.vertices) > 0:
            print(f"[KeyframeManager] Saved TSDF mesh ({self.num_keyframes_integrated} keyframes integrated) to {mesh_filename}")
        else:
            print(f"[KeyframeManager] Completed integration ({self.num_keyframes_integrated} keyframes integrated).")
        return mesh

    def stop(self):
        self.stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)

if __name__ == "__main__":
    import os
    import time
    from slam.camera import Camera
    from slam.map_point import MapPoint

    # Mock testing KeyframeManager pipeline
    cam = Camera(fx=525.0, fy=525.0, cx=320.0, cy=240.0, width=640, height=480)
    test_map = Map()

    # Seed map with 30 mock 3D points
    for i in range(30):
        pt = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(1.2, 2.8)])
        mp = MapPoint(pt, 0)
        test_map.add_map_point(mp)

    manager = KeyframeManager(translation_threshold=0.15)

    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Frame 0: origin
    f0 = Frame(0, 0.0, img, cam)
    f0.set_pose(np.eye(4))
    t0 = manager.process_frame(f0, test_map)
    assert t0, "First frame should trigger keyframe"

    # Frame 1: moved only 0.05m -> should not trigger
    f1 = Frame(1, 0.033, img, cam)
    T1 = np.eye(4)
    T1[0, 3] = -0.05
    f1.set_pose(T1)
    t1 = manager.process_frame(f1, test_map)
    assert not t1, "Small movement should not trigger keyframe"

    # Frame 2: moved 0.20m -> should trigger keyframe
    f2 = Frame(2, 0.066, img, cam)
    T2 = np.eye(4)
    T2[0, 3] = -0.20
    f2.set_pose(T2)
    t2 = manager.process_frame(f2, test_map)
    assert t2, "Movement > 0.15m should trigger keyframe"

    test_mesh = "test_run_mesh.ply"
    try:
        mesh = manager.finalize(test_mesh)
        print(f"Self-test passed! Integrated keyframes: {manager.num_keyframes_integrated}")
    finally:
        manager.stop()
        if os.path.exists(test_mesh):
            os.remove(test_mesh)
