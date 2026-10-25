import os
import sys
import argparse
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slam.visual_slam import VisualSLAM
from slam.tracking import TrackingState
from datasets.synthetic_dataset import SyntheticDataset
from datasets.kitti_dataset import KITTIDataset
from datasets.tum_dataset import TUMDataset
from datasets.video_dataset import VideoDataset
from datasets.image_folder_dataset import ImageFolderDataset
from visualization.viewer_2d import Viewer2D
from visualization.viewer_3d import Viewer3D
from evaluate import compute_ate_metrics, plot_evaluation

def parse_args():
    p = argparse.ArgumentParser(description="Monocular Visual SLAM - Week 1 Baseline")
    p.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "kitti", "tum", "video", "webcam", "images"])
    p.add_argument("--path", type=str, default=None)
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--features", type=int, default=1500)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-gui", action="store_true")
    p.add_argument("--save-plot", type=str, default="slam_output_3d.png")
    p.add_argument("--save-ply", type=str, default=None)
    p.add_argument("--save-trajectory", type=str, default=None)
    p.add_argument("--delay", type=int, default=1)
    return p.parse_args()

def load_dataset(args):
    if args.dataset == "synthetic":
        return SyntheticDataset(num_frames=120)
    if args.dataset == "images":
        return ImageFolderDataset(args.path, fps=args.fps, fx=args.fx, fy=args.fy, max_frames=args.max_frames)
    if args.dataset == "kitti":
        return KITTIDataset(args.path)
    if args.dataset == "tum":
        return TUMDataset(args.path)
    if args.dataset == "video":
        return VideoDataset(args.path, max_frames=args.max_frames)
    if args.dataset == "webcam":
        return VideoDataset(0, max_frames=args.max_frames)
    raise ValueError(f"Unknown dataset: {args.dataset}")

def main():
    args = parse_args()
    dataset = load_dataset(args)
    camera = dataset.get_camera()
    total_frames = len(dataset)

    slam = VisualSLAM(camera=camera, num_features=args.features)
    viewer_2d = None if args.no_gui else Viewer2D(canvas_size=600, scale=12.0)

    print(f"\n--- Monocular Visual SLAM (Week 1 Baseline) ---")
    print(f"Frames: {total_frames} | Features: {args.features} | Camera: {camera}\n")

    t_start = time.time()
    processed = 0
    gt_hist, est_hist = [], []

    try:
        for fid, ts, img, gt_pose in dataset:
            if args.max_frames and processed >= args.max_frames:
                break

            t0 = time.time()
            frame = slam.process_frame(fid, ts, img)
            dt = time.time() - t0
            processed += 1

            if gt_pose is not None and frame.has_pose:
                gt_hist.append(gt_pose[:3, 3])
                est_hist.append(frame.get_camera_center())

            if viewer_2d:
                key = viewer_2d.draw(
                    frame=frame, state=slam.state, num_map_points=len(slam.map.map_points),
                    num_keyframes=len(slam.map.keyframes), gt_pose=gt_pose, wait_ms=args.delay
                )
                if key in (27, ord('q')):
                    break

            if processed % 10 == 0 or processed == total_frames:
                fps = 1.0 / max(dt, 1e-4)
                print(f"[{processed:04d}/{total_frames:04d}] State: {slam.state.name:<12} | Inliers: {slam.num_inliers:3d} | Points: {len(slam.map.map_points):4d} | KFs: {len(slam.map.keyframes):2d} | {fps:.1f} FPS")

    finally:
        if viewer_2d:
            viewer_2d.close()

    total_time = time.time() - t_start
    print(f"\nFinished {processed} frames in {total_time:.2f}s ({processed / max(total_time, 1e-4):.1f} FPS)")

    if len(gt_hist) > 10 and len(est_hist) > 10:
        metrics = compute_ate_metrics(np.array(est_hist), np.array(gt_hist))
        print(f"ATE RMSE: {metrics['ate_rmse']:.4f} m | Scale: {metrics['sim3_scale']:.4f}")
        plot_evaluation(np.array(est_hist), np.array(gt_hist), metrics, save_path="slam_evaluation.png")
    elif gt_hist:
        print("[Eval] Insufficient tracked poses (<10) for trajectory alignment.")

    if args.save_ply:
        slam.map.export_ply(args.save_ply)
    if args.save_trajectory:
        slam.map.export_tum_trajectory(args.save_trajectory)

    Viewer3D().plot_slam_results(slam.map, dataset.get_all_groundtruth_centers(), save_path=args.save_plot, show=not args.no_gui)

if __name__ == "__main__":
    main()
