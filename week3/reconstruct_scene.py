"""
3D scene reconstruction: SLAM tracking + TSDF volumetric fusion + textured mesh export.
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
    parser = argparse.ArgumentParser(description="3D Scene Reconstruction from SLAM + Monocular Depth")
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
        help="Target folder for output models, textures, and video"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process (None = all)"
    )
    parser.add_argument(
        "--features",
        type=int,
        default=1500,
        help="ORB feature budget per frame (default: 1500)"
    )
    parser.add_argument(
        "--room-length",
        type=float,
        default=4.0,
        help="TSDF volume box side length in meters (default: 4.0)"
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
        help="Disable semantic segmentation pass"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip orbit MP4 video rendering"
    )
    return parser.parse_args()


def should_trigger_dense_keyframe(curr_frame, last_kf_pos, last_kf_rot, trans_thresh=0.12, rot_thresh=8.0):
    if last_kf_pos is None:
        return True
    curr_pos = curr_frame.get_camera_center()
    if not np.all(np.isfinite(curr_pos)):
        return False
    if float(np.linalg.norm(curr_pos - last_kf_pos)) >= trans_thresh:
        return True
    R_curr = curr_frame.get_rotation()
    R_rel = R_curr @ last_kf_rot.T
    trace_val = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.degrees(np.arccos(trace_val))
    return angle >= rot_thresh


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print(" [Reconstruction] 3D Scene Reconstruction from SLAM")
    print("=" * 65)
    print(f"[Dataset] Loading from: {args.dataset_path}")
    has_depth = os.path.exists(os.path.join(args.dataset_path, "depth.txt"))
    dataset = TUMDataset(args.dataset_path, is_rgbd=has_depth)
    camera = dataset.get_camera()
    total_frames = len(dataset) if args.max_frames is None else min(args.max_frames, len(dataset))
    print(f"[Dataset] Frames: {total_frames} (RGB-D: {has_depth})")

    # init SLAM
    print("[SLAM] Initializing tracker...")
    slam = VisualSLAM(camera=camera, num_features=args.features)
    if has_depth:
        # lock scale to true sensor depth to avoid gauge drift
        slam.local_mapping.optimizer.max_iterations = 0

    print("[Depth] Loading Depth Anything V2...")
    depth_estimator = MonocularDepthEstimator()

    semantic_segmenter = None
    if not args.no_semantics:
        print("[Semantics] Loading LRASPP segmenter...")
        semantic_segmenter = SemanticSegmenter()

    voxel_size = args.room_length / float(args.tsdf_res)
    print(f"[TSDF] Initializing volume ({args.room_length:.1f}m box, {args.tsdf_res}^3 -> {voxel_size*100:.1f}cm voxels)...")
    tsdf_mapper = SceneTSDFMapper(
        volume_type="uniform",
        room_length=args.room_length,
        resolution=args.tsdf_res,
        sdf_trunc=max(voxel_size * 3.5, 0.05)
    )

    t_start = time.time()
    n_processed = 0
    keyframes = []
    last_kf_pos = None
    last_kf_rot = None

    gt_history = []
    est_history = []

    print("\n" + "-" * 65)
    print(" [Tracking] Processing sequence & integrating keyframes")
    print("-" * 65)

    try:
        for item in dataset:
            if args.max_frames and n_processed >= args.max_frames:
                break

            if len(item) == 5:
                frame_id, timestamp, img_bgr, depth_m, gt_pose = item
            else:
                frame_id, timestamp, img_bgr, gt_pose = item
                depth_m = None

            t0 = time.time()
            frame = slam.process_frame(frame_id, timestamp, img_bgr, depth=depth_m)
            dt = time.time() - t0
            n_processed += 1

            if gt_pose is not None and frame.has_pose:
                gt_history.append(gt_pose[:3, 3])
                est_history.append(frame.get_camera_center())

            # integrate into TSDF whenever camera moved enough
            if frame.has_pose and should_trigger_dense_keyframe(
                frame, last_kf_pos, last_kf_rot, trans_thresh=args.kf_trans, rot_thresh=args.kf_rot
            ):
                last_kf_pos = frame.get_camera_center().copy()
                last_kf_rot = frame.get_rotation().copy()

                if depth_m is not None:
                    metric_depth = depth_m.copy()
                    # fill sensor dropouts with monocular depth
                    bad_mask = (metric_depth < 0.2) | (metric_depth > 4.5) | (~np.isfinite(metric_depth))
                    if np.any(bad_mask):
                        disp = depth_estimator.predict_depth(img_bgr)
                        pts3d, pts2d, _ = frame.get_matched_3d_2d()
                        if len(pts3d) >= 4:
                            pts_c = (frame.T_cw[:3, :3] @ pts3d.T).T + frame.T_cw[:3, 3]
                            valid = (pts_c[:, 2] > 0.1) & np.isfinite(pts_c[:, 2])
                            if np.sum(valid) >= 4:
                                aligned_d, _ = align_depth_scale_shift(pts2d[valid], pts_c[valid, 2], disp)
                                metric_depth[bad_mask] = aligned_d[bad_mask]
                else:
                    disp = depth_estimator.predict_depth(img_bgr)
                    pts3d, pts2d, _ = frame.get_matched_3d_2d()
                    if len(pts3d) >= 4:
                        pts_c = (frame.T_cw[:3, :3] @ pts3d.T).T + frame.T_cw[:3, 3]
                        valid = (pts_c[:, 2] > 0.1) & np.isfinite(pts_c[:, 2])
                        metric_depth, _ = align_depth_scale_shift(pts2d[valid], pts_c[valid, 2], disp)
                    else:
                        metric_depth = np.zeros(img_bgr.shape[:2], dtype=np.float32)

                tsdf_mapper.integrate_keyframe(
                    color_bgr=img_bgr,
                    metric_depth=metric_depth,
                    intrinsic_matrix=camera.K,
                    pose_c2w=frame.get_inverse_pose()
                )

                sem_mask = None
                if semantic_segmenter is not None:
                    sem_mask, _ = semantic_segmenter.predict_mask(img_bgr)

                keyframes.append({
                    "img": img_bgr.copy(),
                    "T_cw": frame.T_cw.copy(),
                    "K": camera.K.copy(),
                    "semantic_mask": sem_mask
                })

            if n_processed % 20 == 0 or n_processed == total_frames:
                fps = 1.0 / max(dt, 1e-4)
                print(f"[{n_processed:04d}/{total_frames:04d}] Inliers: {slam.num_inliers:3d} | Points: {len(slam.map.map_points):4d} | KFs: {len(keyframes):2d} | {fps:5.1f} FPS")

    except KeyboardInterrupt:
        print("\n[SLAM] Interrupted by user.")

    total_dt = time.time() - t_start
    print("-" * 65)
    print(f"[Done] {n_processed} frames in {total_dt:.2f}s ({n_processed / max(total_dt, 1e-3):.1f} avg FPS). Keyframes integrated: {len(keyframes)}")

    # extract mesh via marching cubes
    print("\n" + "=" * 65)
    print(" [Mesh] Extracting TSDF surface mesh")
    print("=" * 65)
    raw_mesh = tsdf_mapper.extract_clean_mesh(min_cluster_faces=120, smooth_iterations=4)
    print(f"[Mesh] Surface: {len(raw_mesh.vertices):,} vertices, {len(raw_mesh.triangles):,} triangles.")

    # texture projection & semantics
    print("\n[Texture] Projecting keyframe textures & semantic labels onto mesh...")
    projector = TextureProjector()
    photo_mesh, semantic_mesh, semantics_meta = projector.colorize_mesh_from_keyframes(
        mesh=raw_mesh,
        keyframes=keyframes,
        semantic_segmenter=semantic_segmenter
    )

    paths = projector.export_textured_model(
        output_dir=args.output_dir,
        photo_mesh=photo_mesh,
        semantic_mesh=semantic_mesh,
        semantics_meta=semantics_meta
    )
    print(f"[Export] Mesh OBJ:     {paths.get('mesh_obj')}")
    print(f"[Export] Semantic PLY: {paths.get('semantic_ply')}")
    print(f"[Export] Semantics:    {paths.get('semantics_json')}")

    if semantics_meta:
        print("\n[Semantics] Detected 3D bounding boxes:")
        for name, data in semantics_meta.items():
            d = data["dimensions_xyz_meters"]
            print(f"   * {name:<18} -> {d[0]:.2f}m x {d[1]:.2f}m x {d[2]:.2f}m ({data['point_count']} vertices)")

    # 360 orbit render
    orbit_video_path = os.path.join(args.output_dir, "cinematic_orbit.mp4")
    if not args.no_video and len(photo_mesh.vertices) > 0:
        print("\n" + "=" * 65)
        print(" [Orbit] Rendering 360 fly-through video")
        print("=" * 65)
        renderer = OrbitRenderer(width=640, height=480, fps=30)
        renderer.render_orbit_video(photo_mesh, orbit_video_path, num_frames=90)

    # trajectory export & benchmark evaluation
    traj_path = os.path.join(args.output_dir, "trajectory_tum.txt")
    slam.map.export_tum_trajectory(traj_path)
    print(f"\n[Export] Trajectory saved: {traj_path}")

    if len(gt_history) > 10 and len(est_history) > 10:
        gt_arr = np.array(gt_history, dtype=np.float64)
        est_arr = np.array(est_history, dtype=np.float64)
        metrics = compute_ate_metrics(est_arr, gt_arr)

        print("\n" + "-" * 65)
        print(f" ATE RMSE:   {metrics['ate_rmse']:.4f} m")
        print(f" Mean error: {metrics['mean_error']:.4f} m")
        print(f" Scale s:    {metrics['sim3_scale']:.4f}")
        print("-" * 65)

        eval_plot_path = os.path.join(args.output_dir, "slam_evaluation.png")
        plot_evaluation(est_arr, gt_arr, metrics, save_path=eval_plot_path, show=False)

    # 3D scatter plot
    plot_3d_path = os.path.join(args.output_dir, "slam_output_3d.png")
    viewer_3d = Viewer3D()
    viewer_3d.plot_slam_results(
        slam_map=slam.map,
        gt_centers=dataset.get_all_groundtruth_centers(),
        save_path=plot_3d_path,
        show=False
    )

    print("\n" + "=" * 65)
    print(" [Done] Pipeline finished.")
    print("=" * 65)
    for f in os.listdir(args.output_dir):
        fp = os.path.join(args.output_dir, f)
        sz = os.path.getsize(fp) / 1024.0
        print(f"   * {f:<26} ({sz:6.1f} KB)")


if __name__ == "__main__":
    main()
