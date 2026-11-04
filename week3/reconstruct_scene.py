"""
Full 3D Scene Reconstruction from RGB-D SLAM & Vision Foundation Models.
Combines real-time Visual SLAM, Depth Anything V2 metric depth estimation,
lightweight semantic segmentation, volumetric TSDF fusion, and 360° cinematic rendering.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import cv2
import open3d as o3d

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure local imports are available
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slam.visual_slam import VisualSLAM
from datasets.tum_dataset import TUMDataset
from depth_module import MonocularDepthEstimator
from scale_aligner import align_depth_scale_shift
from semantic_module import SemanticSegmenter
from dense_reconstruction.scene_tsdf_mapper import SceneTSDFMapper
from dense_reconstruction.texture_projector import TextureProjector
from dense_reconstruction.orbit_renderer import OrbitRenderer
from evaluate import compute_ate_metrics, plot_evaluation
from visualization.viewer_3d import Viewer3D

def parse_args():
    parser = argparse.ArgumentParser(description="Full 3D Scene Reconstruction using SLAM and Neural Foundation Models")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=r"C:\Users\hp\Desktop\rgbd_dataset_freiburg1_desk",
        help="Path to TUM RGB-D sequence folder"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reconstructed_desk_scene",
        help="Directory to save final 3D scene models, textures, and video"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process (None for full dataset)"
    )
    parser.add_argument(
        "--features",
        type=int,
        default=1500,
        help="Number of ORB features for SLAM tracking (default: 1500)"
    )
    parser.add_argument(
        "--room-length",
        type=float,
        default=4.0,
        help="Bounding cube size in meters for TSDF volume (default: 4.0)"
    )
    parser.add_argument(
        "--tsdf-res",
        type=int,
        default=256,
        help="TSDF grid resolution (default: 256 for ~1.5cm voxels)"
    )
    parser.add_argument(
        "--kf-trans",
        type=float,
        default=0.12,
        help="Keyframe translation threshold in meters (default: 0.12)"
    )
    parser.add_argument(
        "--kf-rot",
        type=float,
        default=8.0,
        help="Keyframe rotation threshold in degrees (default: 8.0)"
    )
    parser.add_argument(
        "--no-semantics",
        action="store_true",
        help="Disable semantic segmentation"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip rendering 360 cinematic orbit video"
    )
    return parser.parse_args()

def should_trigger_dense_keyframe(curr_frame, last_kf_pos, last_kf_rot, trans_thresh=0.12, rot_thresh=8.0):
    if last_kf_pos is None:
        return True
    curr_pos = curr_frame.get_camera_center()
    if not np.all(np.isfinite(curr_pos)):
        return False
    dist = float(np.linalg.norm(curr_pos - last_kf_pos))
    if dist >= trans_thresh:
        return True
    R_curr = curr_frame.get_rotation()
    R_rel = R_curr @ last_kf_rot.T
    trace_val = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.degrees(np.arccos(trace_val))
    if angle >= rot_thresh:
        return True
    return False

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print(" [Spatial AI] Full 3D Scene Reconstruction from SLAM")
    print("=" * 65)
    print(f"[Dataset] Loading from: {args.dataset_path}")
    has_depth = os.path.exists(os.path.join(args.dataset_path, "depth.txt"))
    dataset = TUMDataset(args.dataset_path, is_rgbd=has_depth)
    camera = dataset.get_camera()
    total_frames = len(dataset) if args.max_frames is None else min(args.max_frames, len(dataset))
    print(f"[Dataset] Total sequence frames to process: {total_frames} (RGB-D Mode: {has_depth})")

    # 1. Initialize SLAM Tracker
    print("[SLAM] Initializing Visual SLAM Tracking Engine...")
    slam = VisualSLAM(camera=camera, num_features=args.features)
    if has_depth:
        slam.local_mapping.optimizer.max_iterations = 0  # True sensor depth handles scale; avoid unconstrained monocular drift

    # 2. Initialize Foundation Neural Models
    print("[AI Models] Loading Depth Anything V2 Small Foundation Model...")
    depth_estimator = MonocularDepthEstimator()

    semantic_segmenter = None
    if not args.no_semantics:
        print("[AI Models] Loading Lightweight LRASPP MobileNet Semantic Segmenter...")
        semantic_segmenter = SemanticSegmenter()

    # 3. Initialize Volumetric TSDF Mapper
    voxel_size = args.room_length / float(args.tsdf_res)
    print(f"[TSDF] Initializing Volumetric TSDF Grid ({args.room_length:.1f}m room, resolution {args.tsdf_res}^3 -> {voxel_size*100:.1f} cm voxels)...")
    tsdf_mapper = SceneTSDFMapper(
        volume_type="uniform",
        room_length=args.room_length,
        resolution=args.tsdf_res,
        sdf_trunc=max(voxel_size * 3.5, 0.05)
    )

    start_time = time.time()
    processed_count = 0
    keyframe_snapshots = []
    last_kf_pos = None
    last_kf_rot = None

    gt_history = []
    est_history = []

    print("\n" + "-" * 65)
    print(" [SLAM] Executing Tracking & Neural Volumetric Integration")
    print("-" * 65)

    try:
        for item in dataset:
            if args.max_frames and processed_count >= args.max_frames:
                break

            if len(item) == 5:
                frame_id, timestamp, img_bgr, depth_m, gt_pose = item
            else:
                frame_id, timestamp, img_bgr, gt_pose = item
                depth_m = None

            t0 = time.time()
            frame = slam.process_frame(frame_id, timestamp, img_bgr, depth=depth_m)
            duration = time.time() - t0
            processed_count += 1

            if gt_pose is not None and frame.has_pose:
                gt_history.append(gt_pose[:3, 3])
                est_history.append(frame.get_camera_center())

            # Check Keyframe trigger for neural depth & volumetric integration
            if frame.has_pose:
                if should_trigger_dense_keyframe(frame, last_kf_pos, last_kf_rot, trans_thresh=args.kf_trans, rot_thresh=args.kf_rot):
                    last_kf_pos = frame.get_camera_center().copy()
                    last_kf_rot = frame.get_rotation().copy()

                    # Neural depth completion or direct metric depth
                    if depth_m is not None:
                        metric_depth = depth_m.copy()
                        invalid_mask = (metric_depth < 0.2) | (metric_depth > 4.5) | (~np.isfinite(metric_depth))
                        if np.any(invalid_mask):
                            raw_pred_disp = depth_estimator.predict_depth(img_bgr)
                            pts_3d_inliers, pts_2d_inliers, _ = frame.get_matched_3d_2d()
                            if len(pts_3d_inliers) >= 4:
                                R_cw = frame.T_cw[:3, :3]
                                t_cw = frame.T_cw[:3, 3]
                                pts_c = (R_cw @ pts_3d_inliers.T).T + t_cw
                                valid_z = (pts_c[:, 2] > 0.1) & np.isfinite(pts_c[:, 2])
                                if np.sum(valid_z) >= 4:
                                    neural_metric, _ = align_depth_scale_shift(
                                        pts_2d_inliers[valid_z],
                                        pts_c[valid_z, 2],
                                        raw_pred_disp
                                    )
                                    metric_depth[invalid_mask] = neural_metric[invalid_mask]
                    else:
                        # 1. Neural Depth Inference via Depth Anything V2
                        raw_pred_disp = depth_estimator.predict_depth(img_bgr)

                        # 2. Extract tracked inlier landmarks (guaranteed visible and unoccluded)
                        pts_3d_inliers, pts_2d_inliers, _ = frame.get_matched_3d_2d()
                        if len(pts_3d_inliers) >= 4:
                            R_cw = frame.T_cw[:3, :3]
                            t_cw = frame.T_cw[:3, 3]
                            pts_c = (R_cw @ pts_3d_inliers.T).T + t_cw
                            valid_z = (pts_c[:, 2] > 0.1) & np.isfinite(pts_c[:, 2])

                            pixel_coords = pts_2d_inliers[valid_z]
                            slam_depths = pts_c[valid_z, 2]

                            # 3. Inverse-disparity metric alignment
                            metric_depth, inlier_mask = align_depth_scale_shift(
                                pixel_coords,
                                slam_depths,
                                raw_pred_disp
                            )
                        else:
                            metric_depth = np.zeros(img_bgr.shape[:2], dtype=np.float32)

                    # 4. Integrate into volumetric TSDF
                    pose_c2w = frame.get_inverse_pose()
                    tsdf_mapper.integrate_keyframe(
                        color_bgr=img_bgr,
                        metric_depth=metric_depth,
                        intrinsic_matrix=camera.K,
                        pose_c2w=pose_c2w
                    )

                    # 5. Semantic Segmentation on Keyframe
                    sem_mask = None
                    if semantic_segmenter is not None:
                        sem_mask, _ = semantic_segmenter.predict_mask(img_bgr)

                    keyframe_snapshots.append({
                        "img": img_bgr.copy(),
                        "T_cw": frame.T_cw.copy(),
                        "K": camera.K.copy(),
                        "semantic_mask": sem_mask
                    })

            if processed_count % 20 == 0 or processed_count == total_frames:
                fps = 1.0 / max(duration, 1e-4)
                pts_count = len(slam.map.map_points)
                kfs_count = len(keyframe_snapshots)
                print(f"[Frame {processed_count:04d}/{total_frames:04d}] Inliers: {slam.num_inliers:3d} | Points: {pts_count:4d} | Dense KFs: {kfs_count:2d} | Tracking: {fps:5.1f} FPS")

    except KeyboardInterrupt:
        print("\n[SLAM] Interrupted by user.")

    total_duration = time.time() - start_time
    print("-" * 65)
    print(f"[OK] Tracking complete! Processed {processed_count} frames in {total_duration:.2f} s ({processed_count / max(total_duration, 1e-3):.1f} avg FPS).")
    print(f"[TSDF] Total Dense Keyframes Integrated: {len(keyframe_snapshots)}")

    # 4. Extract Watertight Solid Mesh
    print("\n" + "=" * 65)
    print(" [Mesh] Polygonizing Continuous Solid 3D Scene Mesh")
    print("=" * 65)
    raw_mesh = tsdf_mapper.extract_clean_mesh(min_cluster_faces=120, smooth_iterations=4)
    print(f"[Mesh] Extracted watertight solid surface: {len(raw_mesh.vertices):,} vertices, {len(raw_mesh.triangles):,} triangles.")

    # 5. Texture Projection & Semantic Scene Graph
    print("\n[Texture & Semantics] Projecting photographic keyframes & semantic labels onto 3D surfaces...")
    projector = TextureProjector()
    photo_mesh, semantic_mesh, semantics_meta = projector.colorize_mesh_from_keyframes(
        mesh=raw_mesh,
        keyframes=keyframe_snapshots,
        semantic_segmenter=semantic_segmenter
    )

    export_paths = projector.export_textured_model(
        output_dir=args.output_dir,
        photo_mesh=photo_mesh,
        semantic_mesh=semantic_mesh,
        semantics_meta=semantics_meta
    )
    print(f"[Export] Saved 3D Scene Mesh:       {export_paths.get('mesh_obj')}")
    print(f"[Export] Saved Semantic Color Mesh:  {export_paths.get('semantic_ply')}")
    print(f"[Export] Saved Semantic Scene Graph: {export_paths.get('semantics_json')}")

    if semantics_meta:
        print("\n[Spatial AI] Detected Physical 3D Objects:")
        for obj_name, data in semantics_meta.items():
            dims = data["dimensions_xyz_meters"]
            print(f"   * {obj_name:<18} -> Width: {dims[0]:.2f}m, Height: {dims[1]:.2f}m, Depth: {dims[2]:.2f}m ({data['point_count']} vertices)")

    # 6. Render 360-degree Cinematic Orbit Video
    orbit_video_path = os.path.join(args.output_dir, "cinematic_orbit.mp4")
    if not args.no_video and len(photo_mesh.vertices) > 0:
        print("\n" + "=" * 65)
        print(" [Orbit] Rendering Hardware-Accelerated 360-Degree Cinematic Fly-Through Video")
        print("=" * 65)
        renderer = OrbitRenderer(width=640, height=480, fps=30)
        renderer.render_orbit_video(photo_mesh, orbit_video_path, num_frames=90)

    # 7. Trajectory Export & Sim(3) Benchmark Evaluation
    traj_path = os.path.join(args.output_dir, "trajectory_tum.txt")
    slam.map.export_tum_trajectory(traj_path)
    print(f"\n[Export] Saved TUM-format trajectory to: {traj_path}")

    if len(gt_history) > 10 and len(est_history) > 10:
        gt_arr = np.array(gt_history, dtype=np.float64)
        est_arr = np.array(est_history, dtype=np.float64)
        metrics = compute_ate_metrics(est_arr, gt_arr)

        print("\n" + "-" * 65)
        print(" [Benchmark] Evaluation (Umeyama Sim(3) Scale-Aligned)")
        print(f" Absolute Trajectory Error (ATE RMSE): {metrics['ate_rmse']:.4f} m")
        print(f" Mean Error:                           {metrics['mean_error']:.4f} m")
        print(f" Estimated Monocular Scale Factor:     {metrics['sim3_scale']:.4f}")
        print("-" * 65)

        eval_plot_path = os.path.join(args.output_dir, "slam_evaluation.png")
        plot_evaluation(est_arr, gt_arr, metrics, save_path=eval_plot_path, show=False)

    # 8. Render 3D Map Visualization
    plot_3d_path = os.path.join(args.output_dir, "slam_output_3d.png")
    viewer_3d = Viewer3D()
    viewer_3d.plot_slam_results(
        slam_map=slam.map,
        gt_centers=dataset.get_all_groundtruth_centers(),
        save_path=plot_3d_path,
        show=False
    )

    print("\n" + "=" * 65)
    print(" [SUCCESS] 3D SCENE RECONSTRUCTION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    print(f"All deliverables saved in: {os.path.abspath(args.output_dir)}")
    for f in os.listdir(args.output_dir):
        fp = os.path.join(args.output_dir, f)
        sz = os.path.getsize(fp) / 1024.0
        print(f"   * {f:<26} ({sz:6.1f} KB)")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
