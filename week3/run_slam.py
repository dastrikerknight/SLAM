"""CLI runner for monocular visual SLAM: TUM, KITTI, synthetic, video streams."""

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
    parser = argparse.ArgumentParser(description="Monocular Visual SLAM")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "kitti", "tum", "video", "webcam", "images"],
        help="Dataset loader (default: synthetic)"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to dataset sequence, image directory, or video file"
    )
    parser.add_argument(
        "--fx",
        type=float,
        default=None,
        help="Camera focal length fx in pixels"
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=None,
        help="Camera focal length fy in pixels"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Framerate of image sequence (default: 30.0)"
    )
    parser.add_argument(
        "--features",
        type=int,
        default=1500,
        help="ORB feature budget per frame (default: 1500)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Headless mode (no OpenCV 2D viewer)"
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="slam_output_3d.png",
        help="Output path for 3D trajectory/point plot"
    )
    parser.add_argument(
        "--save-ply",
        type=str,
        default=None,
        help="Export 3D point cloud (.ply)"
    )
    parser.add_argument(
        "--save-trajectory",
        type=str,
        default=None,
        help="Export camera poses in TUM format (.txt)"
    )
    parser.add_argument(
        "--save-map",
        type=str,
        default=None,
        help="Serialize binary map (.bin)"
    )
    parser.add_argument(
        "--load-map",
        type=str,
        default=None,
        help="Load pre-built binary map for relocalization"
    )
    parser.add_argument(
        "--meshing",
        action="store_true",
        help="Triangulate solid surface mesh from map points"
    )
    parser.add_argument(
        "--save-mesh",
        type=str,
        default=None,
        help="Output path for surface mesh (.obj)"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=1,
        help="OpenCV window delay in ms (1 = fast, 30 = ~30 fps)"
    )
    return parser.parse_args()


def load_dataset(args):
    if args.dataset == "synthetic":
        n_frames = args.max_frames if args.max_frames else 120
        print(f"[Dataset] Synthetic generator ({n_frames} frames)...")
        return SyntheticDataset(num_frames=n_frames)

    if args.dataset == "images":
        if not args.path:
            sys.exit("Error: --path required for image folder dataset")
        print(f"[Dataset] Image directory: {args.path}")
        return ImageFolderDataset(
            folder_path=args.path,
            fps=args.fps,
            fx=args.fx,
            fy=args.fy,
            max_frames=args.max_frames
        )

    if args.dataset == "kitti":
        if not args.path:
            sys.exit("Error: --path required for KITTI sequence")
        print(f"[Dataset] KITTI sequence: {args.path}")
        return KITTIDataset(args.path)

    if args.dataset == "tum":
        if not args.path:
            sys.exit("Error: --path required for TUM sequence")
        print(f"[Dataset] TUM sequence: {args.path}")
        return TUMDataset(args.path)

    if args.dataset == "video":
        if not args.path:
            sys.exit("Error: --path required for video stream")
        print(f"[Dataset] Video file: {args.path}")
        return VideoDataset(args.path, max_frames=args.max_frames)

    if args.dataset == "webcam":
        print("[Dataset] Connecting to webcam 0...")
        return VideoDataset(0, max_frames=args.max_frames)

    raise ValueError(f"Unknown dataset type: {args.dataset}")


def main():
    args = parse_args()

    dataset = load_dataset(args)
    camera = dataset.get_camera()
    total_frames = len(dataset)
    print(f"[Camera] {camera}")

    slam = VisualSLAM(
        camera=camera,
        num_features=args.features,
        min_inliers_pnp=10,
        min_parallax_deg=2.5
    )

    if args.load_map:
        from slam.map import Map
        slam.map = Map.load_binary(args.load_map, camera)
        slam.tracking.slam_map = slam.map
        slam.tracking.state = TrackingState.LOST
        print(f"[Map] Loaded map ({len(slam.map.keyframes)} KFs, {len(slam.map.map_points)} pts). Relocalizing...")

    viewer_2d = None
    if not args.no_gui:
        viewer_2d = Viewer2D(canvas_size=600, scale=12.0)

    print("\n" + "=" * 60)
    print(f" Monocular Visual SLAM (OpenCV {cv2.__version__})")
    print(f" Frames: {total_frames} | Features: {args.features}")
    if not args.no_gui:
        print(" Press 'q' or 'ESC' to stop.")
    print("=" * 60 + "\n")

    t_start = time.time()
    n_processed = 0
    gt_history = []
    est_history = []

    try:
        for item in dataset:
            if len(item) == 5:
                frame_id, timestamp, img, depth_m, gt_pose = item
            else:
                frame_id, timestamp, img, gt_pose = item
                depth_m = None

            if args.max_frames and n_processed >= args.max_frames:
                break

            t0 = time.time()
            frame = slam.process_frame(frame_id, timestamp, img, depth=depth_m)
            dt = time.time() - t0
            n_processed += 1

            if gt_pose is not None and frame.has_pose:
                gt_history.append(gt_pose[:3, 3])
                est_history.append(frame.get_camera_center())

            if viewer_2d is not None:
                key = viewer_2d.draw(
                    frame=frame,
                    state=slam.state,
                    num_map_points=len(slam.map.map_points),
                    num_keyframes=len(slam.map.keyframes),
                    gt_pose=gt_pose,
                    wait_ms=args.delay
                )
                if key in (27, ord('q')):
                    print("\n[SLAM] Interrupted by user.")
                    break

            if n_processed % 10 == 0 or n_processed == total_frames:
                fps = 1.0 / max(dt, 1e-4)
                print(f"[{n_processed:04d}/{total_frames:04d}] State: {slam.state.name:<12} | Inliers: {slam.num_inliers:3d} | Points: {len(slam.map.map_points):4d} | KFs: {len(slam.map.keyframes):2d} | {fps:5.1f} FPS")

    except KeyboardInterrupt:
        print("\n[SLAM] Interrupted by user.")

    finally:
        if viewer_2d is not None:
            viewer_2d.close()

    total_dt = time.time() - t_start
    avg_fps = n_processed / max(total_dt, 1e-4)

    print("\n" + "=" * 60)
    print(" Execution Summary")
    print(f" Frames:    {n_processed}/{total_frames}")
    print(f" Time:      {total_dt:.2f} s ({avg_fps:.1f} avg FPS)")
    print(f" KeyFrames: {len(slam.map.keyframes)}")
    print(f" Points:    {len(slam.map.map_points)}")
    print("=" * 60)

    # trajectory evaluation vs ground truth
    if len(gt_history) > 10 and len(est_history) > 10:
        gt_arr = np.array(gt_history, dtype=np.float64)
        est_arr = np.array(est_history, dtype=np.float64)

        metrics = compute_ate_metrics(est_arr, gt_arr)
        print("\n" + "-" * 60)
        print(" Benchmark Evaluation (Umeyama Sim(3))")
        print(f" ATE RMSE:     {metrics['ate_rmse']:.4f} m")
        print(f" Mean error:   {metrics['mean_error']:.4f} m")
        print(f" Median error: {metrics['median_error']:.4f} m")
        print(f" Scale s:      {metrics['sim3_scale']:.4f}")
        print("-" * 60)

        plot_evaluation(est_arr, gt_arr, metrics, save_path="slam_evaluation.png", show=False)

    if args.save_ply and slam.map.export_ply(args.save_ply):
        print(f"[Export] Saved point cloud: {args.save_ply}")

    if args.save_trajectory and slam.map.export_tum_trajectory(args.save_trajectory):
        print(f"[Export] Saved TUM trajectory: {args.save_trajectory}")

    if args.save_map and slam.map.save_binary(args.save_map):
        print(f"[Export] Saved binary map: {args.save_map}")

    if args.meshing or args.save_mesh:
        out_mesh = args.save_mesh or "scene_mesh.obj"
        pts = slam.map.get_all_points_3d()
        cols = slam.map.get_all_points_colors()
        if len(pts) >= 10:
            from surface_meshing import reconstruct_surface_mesh, export_obj
            v, faces, c, normals = reconstruct_surface_mesh(pts, cols, max_edge_length=0.15)
            export_obj(out_mesh, v, faces, c, normals)
            print(f"[Export] Saved surface mesh ({len(v)} vertices, {len(faces)} faces) -> {out_mesh}")

    viewer_3d = Viewer3D()
    viewer_3d.plot_slam_results(
        slam_map=slam.map,
        gt_centers=dataset.get_all_groundtruth_centers(),
        save_path=args.save_plot,
        show=not args.no_gui
    )


if __name__ == "__main__":
    main()
