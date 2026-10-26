import unittest
import numpy as np
import cv2

from slam.camera import Camera
from slam.feature_tracker import FeatureTracker
from slam.visual_slam import VisualSLAM
from slam.tracking import TrackingState
from datasets.synthetic_dataset import SyntheticDataset
from evaluate import compute_ate_metrics

class TestSLAM(unittest.TestCase):
    def test_camera_projection(self):
        cam = Camera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
        pts = np.array([[0.0, 0.0, 5.0], [1.0, 0.5, 5.0]])
        uv, valid = cam.project(pts)
        self.assertTrue(valid[0])
        self.assertAlmostEqual(uv[0, 0], 320.0, delta=1e-3)
        self.assertAlmostEqual(uv[0, 1], 240.0, delta=1e-3)
        self.assertAlmostEqual(uv[1, 0], 420.0, delta=1e-3)

    def test_feature_tracker(self):
        tracker = FeatureTracker(num_features=500)
        img = np.zeros((480, 640), dtype=np.uint8)
        for x in range(100, 500, 50):
            for y in range(100, 400, 50):
                cv2.circle(img, (x, y), 8, 255, -1)
        kps, des = tracker.detect_and_compute(img)
        self.assertGreater(len(kps), 10)
        self.assertEqual(des.shape[1], 32)

    def test_umeyama_sim3(self):
        np.random.seed(0)
        true_pts = np.random.uniform(-10, 10, (50, 3))
        theta = np.radians(35)
        R_true = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        scale_true = 2.5
        t_true = np.array([5.0, -2.0, 8.0])
        transformed = (1.0 / scale_true) * (R_true.T @ (true_pts - t_true).T).T

        metrics = compute_ate_metrics(transformed, true_pts)
        self.assertAlmostEqual(metrics["sim3_scale"], scale_true, delta=1e-3)
        self.assertLess(metrics["ate_rmse"], 1e-4)

    def test_synthetic_pipeline(self):
        dataset = SyntheticDataset(num_frames=80)
        slam = VisualSLAM(camera=dataset.get_camera(), num_features=1500)
        for frame_id, ts, img, _ in dataset:
            slam.process_frame(frame_id, ts, img)
            if frame_id >= 25:
                break
        self.assertGreater(len(slam.map.map_points), 30)
        self.assertEqual(slam.state, TrackingState.OK)

if __name__ == "__main__":
    unittest.main()
