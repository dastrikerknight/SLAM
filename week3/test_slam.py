import os
import unittest
import numpy as np
import cv2

from slam.camera import Camera
from slam.feature_tracker import FeatureTracker
from slam.map import Map
from slam.map_point import MapPoint
from slam.frame import Frame
from slam.visual_slam import VisualSLAM
from slam.tracking import TrackingState
from slam.bow_vocabulary import VisualVocabulary, InvertedIndex
from slam.pose_graph import PoseGraphOptimizer
from surface_meshing import reconstruct_surface_mesh
from datasets.synthetic_dataset import SyntheticDataset
from evaluate import compute_ate_metrics

class TestSLAMComponents(unittest.TestCase):
    def test_camera_projection(self):
        cam = Camera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
        pt_cam = np.array([[0.0, 0.0, 5.0], [1.0, 0.5, 5.0]])
        uv, valid = cam.project(pt_cam)

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

    def test_umeyama_sim3_alignment(self):
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
        transformed_pts = (1.0 / scale_true) * (R_true.T @ (true_pts - t_true).T).T

        metrics = compute_ate_metrics(transformed_pts, true_pts)
        self.assertAlmostEqual(metrics["sim3_scale"], scale_true, delta=1e-3)
        self.assertLess(metrics["ate_rmse"], 1e-4)

    def test_bow_vocabulary(self):
        vocab = VisualVocabulary(num_words=16)
        dummy_des = [np.random.randint(0, 256, (30, 32), dtype=np.uint8) for _ in range(5)]
        vocab.train(dummy_des)
        self.assertTrue(vocab.is_trained())

        v0 = vocab.transform(dummy_des[0])
        self.assertEqual(len(v0), 16)
        self.assertAlmostEqual(np.sum(v0), 1.0, delta=1e-4)

        index = InvertedIndex(num_words=16)
        index.add_keyframe(101, v0)
        results = index.query(v0, min_score=0.5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 101)

    def test_pose_graph_optimization(self):
        pgo = PoseGraphOptimizer(max_iterations=15)
        # Create a circle trajectory of 5 poses with small artificial drift at the end
        poses = {}
        for i in range(5):
            angle = i * (2.0 * np.pi / 5.0)
            T = np.eye(4)
            T[0, 3] = np.cos(angle)
            T[1, 3] = np.sin(angle)
            poses[i] = T

        # Add loop edge between pose 4 and pose 0
        T_rel = poses[0] @ np.linalg.inv(poses[4])
        loop_edges = [(4, 0, T_rel, 5.0)]

        # Inject artificial drift to pose 4
        poses[4][0, 3] += 0.20
        poses[4][1, 3] -= 0.15

        opt_poses = pgo.optimize(poses, loop_edges)
        err_before = np.linalg.norm(poses[4][:3, 3] - poses[0][:3, 3])
        err_after = np.linalg.norm((opt_poses[4] @ T_rel)[:3, 3] - opt_poses[0][:3, 3])
        self.assertLess(err_after, err_before)

    def test_map_persistence(self):
        cam = Camera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
        m = Map()
        mp = MapPoint(np.array([1.0, 2.0, 3.0]), 0, np.array([255, 0, 0], dtype=np.uint8))
        m.add_map_point(mp)
        mp_id = mp.id

        img = np.zeros((480, 640), dtype=np.uint8)
        kf = Frame(0, 0.0, img, cam)
        kf.set_pose(np.eye(4))
        kf.set_features([cv2.KeyPoint(100, 100, 10)], np.ones((1, 32), dtype=np.uint8), [mp])
        m.add_keyframe(kf)

        tmp_path = "test_map_persistence.bin"
        try:
            m.save_binary(tmp_path)
            loaded = Map.load_binary(tmp_path, cam)
            self.assertEqual(len(loaded.keyframes), 1)
            self.assertEqual(len(loaded.map_points), 1)
            self.assertAlmostEqual(loaded.map_points[mp_id].point_3d[0], 1.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_surface_meshing(self):
        # 100 random points on a unit plane
        pts = np.random.uniform(-1, 1, (100, 3))
        pts[:, 2] = 0.05 * np.sin(pts[:, 0] * 3.0) # slight curve
        cols = np.ones((100, 3), dtype=np.uint8) * 180

        v, faces, c, normals = reconstruct_surface_mesh(pts, cols, max_edge_length=0.40)
        self.assertEqual(len(v), 100)
        self.assertGreater(len(faces), 10)
        self.assertEqual(faces.shape[1], 3)

    def test_synthetic_slam_pipeline(self):
        dataset = SyntheticDataset(num_frames=80)
        camera = dataset.get_camera()
        slam = VisualSLAM(camera=camera, num_features=1500)

        processed = 0
        for frame_id, ts, img, gt_pose in dataset:
            frame = slam.process_frame(frame_id, ts, img)
            processed += 1
            if processed >= 25:
                break

        self.assertGreater(len(slam.map.map_points), 30)
        self.assertGreater(len(slam.map.keyframes), 1)
        self.assertEqual(slam.state, TrackingState.OK)

if __name__ == "__main__":
    unittest.main()
