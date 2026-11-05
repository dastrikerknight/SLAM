"""CLI entrypoint for monocular SLAM: KITTI, TUM, synthetic benchmark, video, webcam."""

import os
import sys
import argparse
import time
import numpy as np
import cv2

# Ensure local packages are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slam.visual_slam import VisualSLAM
from slam.tracking import TrackingState
from datasets.synthetic_dataset import SyntheticDataset
from datasets.kitti_dataset import KITTIDataset
from datasets.tum_dataset import TUMDataset
from datasets.video_dataset import VideoDataset
from visualization.viewer_2d import Viewer2D
from visualization.viewer_3d import Viewer3D
from evaluate import compute_ate_metrics, plot_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Monocular Visual SLAM using OpenCV")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "kitti", "tum", "video", "webcam", "images"],
        help="Dataset type to run SLAM on (default: synthetic)"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to dataset sequence, image folder, or video file"
    )
    parser.add_argument(
        "--fx",
        type=float,
        default=None,
        help="Optional camera focal length fx in pixels (if known)"
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=None,
        help="Optional camera focal length fy in pixels (if known)"
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
        help="Number of ORB features to detect per frame (default: 1500)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run in headless mode without 2D OpenCV windows"
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="slam_output_3d.png",
        help="Path to save final 3D reconstruction plot"
    )
    parser.add_argument(
        "--save-ply",
        type=str,
        default=None,
        help="Optional path to export 3D point cloud as a .ply file (e.g. map.ply)"
    )
    parser.add_argument(
        "--save-trajectory",
        type=str,
        default=None,
        help="Optional path to export estimated trajectory in TUM format (e.g. trajectory.txt)"
    )
    parser.add_argument(
        "--save-map",
        type=str,
        default=None,
        help="Optional path to serialize and save binary SLAM map (e.g. map.bin)"
    )
    parser.add_argument(
        "--load-map",
        type=str,
        default=None,
        help="Optional path to load pre-built binary SLAM map for relocalization"
    )
    parser.add_argument(
        "--meshing",
        action="store_true",
        help="Reconstruct solid 3D triangle surface mesh from map points"
    )
    parser.add_argument(
        "--save-mesh",
        type=str,
        default=None,
        help="Path to export 3D triangle mesh (e.g. scene_mesh.obj)"
    )
    parser.add_argument(
        "--dense-tsdf",
        action="store_true",
        help="Run real-time neural depth estimation & volumetric TSDF mapping"
    )
    parser.add_argument(
        "--tsdf-mesh-out",
        type=str,
        default="dense_tsdf_mesh.ply",
        help="Output path for watertight TSDF mesh (.ply)"
    )
    parser.add_argument(
        "--kf-trans",
        type=float,
        default=0.15,
        help="Translation displacement (m) to trigger a TSDF keyframe (default: 0.15)"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=1,
        help="Wait delay in ms for OpenCV viewer (1 for max speed, 30 for ~30 FPS)"
    )
    return parser.parse_args()


from datasets.image_folder_dataset import ImageFolderDataset


def load_dataset(args):
    """Instantiates the appropriate dataset loader."""
    if args.dataset == "synthetic":
        num_frames = args.max_frames if args.max_frames else 120
        print(f"[Dataset] Initializing Synthetic 3D Benchmark ({num_frames} frames)...")
        return SyntheticDataset(num_frames=num_frames)

    elif args.dataset == "images":
        if not args.path:
            print("Error: --path is required for images dataset (e.g. --path /path/to/my_frames/)")
            sys.exit(1)
        print(f"[Dataset] Loading image sequence from folder: {args.path}")
        return ImageFolderDataset(
            folder_path=args.path,
            fps=args.fps,
            fx=args.fx,
            fy=args.fy,
            max_frames=args.max_frames
        )

    elif args.dataset == "kitti":
        if not args.path:
            print("Error: --path is required for KITTI dataset (e.g. --path /path/to/sequences/00)")
            sys.exit(1)
        print(f"[Dataset] Loading KITTI Odometry dataset from: {args.path}")
        return KITTIDataset(args.path)

    elif args.dataset == "tum":
        if not args.path:
            print("Error: --path is required for TUM dataset (e.g. --path /path/to/rgbd_dataset_freiburg1_xyz)")
            sys.exit(1)
        print(f"[Dataset] Loading TUM RGB-D dataset from: {args.path}")
        return TUMDataset(args.path)

    elif args.dataset == "video":
        if not args.path:
            print("Error: --path is required for video file (e.g. --path input.mp4)")
            sys.exit(1)
        print(f"[Dataset] Loading video stream from: {args.path}")
        return VideoDataset(args.path, max_frames=args.max_frames)

    elif args.dataset == "webcam":
        print("[Dataset] Connecting to live webcam (Device 0)...")
        return VideoDataset(0, max_frames=args.max_frames)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def main():
    args = parse_args()

    # dataset setup
    dataset = load_dataset(args)
    camera = dataset.get_camera()
    total_frames = len(dataset)
    print(f"[Camera] Model: {camera}")

    # SLAM tracker
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
        print(f"[Map] Loaded pre-built map ({len(slam.map.keyframes)} keyframes, {len(slam.map.map_points)} points). Operating in relocalization mode.")

    if args.dense_tsdf:
        print("[Notice] For full volumetric TSDF reconstruction, neural depth, and semantic meshing, use: python reconstruct_scene.py")

    # 2D preview
    viewer_2d = None
    if not args.no_gui:
        viewer_2d = Viewer2D(canvas_size=600, scale=12.0)

    print("\n" + "=" * 60)
    print(f" Starting Monocular Visual SLAM (OpenCV {cv2.__version__})")
    print(f" Total Frames: {total_frames} | Features per frame: {args.features}")
    if not args.no_gui:
        print(" Press 'q' or 'ESC' in any OpenCV window to stop early.")
    print("=" * 60 + "\n")

    start_time = time.time()
    processed_count = 0
    gt_history = []
    est_history = []

    try:
        for frame_id, timestamp, img, gt_pose in dataset:
            if args.max_frames and processed_count >= args.max_frames:
                break

            t0 = time.time()
            frame = slam.process_frame(frame_id, timestamp, img)
            process_duration = time.time() - t0
            processed_count += 1

            # Collect poses for evaluation if GT is available
            if gt_pose is not None and frame.has_pose:
                gt_pos = gt_pose[:3, 3]
                est_pos = frame.get_camera_center()
                gt_history.append(gt_pos)
                est_history.append(est_pos)

            # Update 2D GUI
            if viewer_2d is not None:
                key = viewer_2d.draw(
                    frame=frame,
                    state=slam.state,
                    num_map_points=len(slam.map.map_points),
                    num_keyframes=len(slam.map.keyframes),
                    gt_pose=gt_pose,
                    wait_ms=args.delay
                )
                if key == 27 or key == ord('q'):
                    print("\n[SLAM] User interrupted execution.")
                    break

            # Console logging every 10 frames
            if processed_count % 10 == 0 or processed_count == total_frames:
                state_str = slam.state.name
                inliers = slam.num_inliers
                map_pts = len(slam.map.map_points)
                kfs = len(slam.map.keyframes)
                fps = 1.0 / max(process_duration, 1e-4)
                print(f"[Frame {processed_count:04d}/{total_frames:04d}] State: {state_str:<12} | Inliers: {inliers:3d} | 3D Points: {map_pts:4d} | KeyFrames: {kfs:2d} | Speed: {fps:.1f} FPS")

    except KeyboardInterrupt:
        print("\n[SLAM] Interrupted by user.")

    finally:
        if viewer_2d is not None:
            viewer_2d.close()

    total_time = time.time() - start_time
    avg_fps = processed_count / max(total_time, 1e-4)

    print("\n" + "=" * 60)
    print(" SLAM Execution Summary")
    print(f" Processed Frames: {processed_count}/{total_frames}")
    print(f" Total Time:       {total_time:.2f} s")
    print(f" Average Speed:    {avg_fps:.1f} FPS")
    print(f" KeyFrames:        {len(slam.map.keyframes)}")
    print(f" 3D MapPoints:     {len(slam.map.map_points)}")
    print("=" * 60)

    # trajectory evaluation (Sim(3) Umeyama alignment & ATE RMSE)
    if len(gt_history) > 10 and len(est_history) > 10:
        gt_arr = np.array(gt_history, dtype=np.float64)
        est_arr = np.array(est_history, dtype=np.float64)

        metrics = compute_ate_metrics(est_arr, gt_arr)
        print("\n" + "-" * 60)
        print(" Benchmark Evaluation (Umeyama Sim(3) Scale-Aligned)")
        print(f" Absolute Trajectory Error (ATE RMSE): {metrics['ate_rmse']:.4f} m")
        print(f" Mean Error:                           {metrics['mean_error']:.4f} m")
        print(f" Median Error:                         {metrics['median_error']:.4f} m")
        print(f" Max Error:                            {metrics['max_error']:.4f} m")
        print(f" Estimated Monocular Scale Factor:     {metrics['sim3_scale']:.4f}")
        print("-" * 60)

        eval_plot_path = "slam_evaluation.png"
        plot_evaluation(est_arr, gt_arr, metrics, save_path=eval_plot_path, show=False)

    # 5. Export 3D Point Cloud (.ply) and Trajectory (.txt) if requested
    if args.save_ply:
        ok = slam.map.export_ply(args.save_ply)
        if ok:
            print(f"[Export] Saved 3D point cloud to: {args.save_ply}")

    if args.save_trajectory:
        ok = slam.map.export_tum_trajectory(args.save_trajectory)
        if ok:
            print(f"[Export] Saved TUM-format trajectory to: {args.save_trajectory}")

    if args.save_map:
        ok = slam.map.save_binary(args.save_map)
        if ok:
            print(f"[Export] Saved binary SLAM map to: {args.save_map}")

    if args.meshing or args.save_mesh:
        out_mesh = args.save_mesh or "scene_mesh.obj"
        pts = slam.map.get_all_points_3d()
        cols = slam.map.get_all_points_colors()
        if len(pts) >= 10:
            from surface_meshing import reconstruct_surface_mesh, export_obj
            v, faces, c, normals = reconstruct_surface_mesh(pts, cols, max_edge_length=0.15)
            export_obj(out_mesh, v, faces, c, normals)
            print(f"[Export] Reconstructed solid 3D surface mesh ({len(v)} vertices, {len(faces)} faces) -> {out_mesh}")

    # 6. 3D Map & Trajectory Visualization
    gt_centers = dataset.get_all_groundtruth_centers()
    viewer_3d = Viewer3D()
    viewer_3d.plot_slam_results(
        slam_map=slam.map,
        gt_centers=gt_centers,
        save_path=args.save_plot,
        show=not args.no_gui
    )


if __name__ == "__main__":
    main()
