"""
Dense 3D Scene Reconstruction for Monocular Visual SLAM and RGB-D Sequences.
Constructs a photorealistic, dense 3D point cloud and textured scene model
from video footage and estimated camera trajectories.
"""

import os
import sys
import argparse
import time
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2

from slam.camera import Camera
from datasets.tum_dataset import quaternion_to_rotation_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dense 3D Scene Reconstruction from Video / RGB-D Sequences"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to TUM RGB-D dataset folder (e.g. C:\\Users\\hp\\Desktop\\rgbd_dataset_freiburg1_desk) or video directory"
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default=None,
        help="Path to estimated trajectory file in TUM format (e.g. traj_desk.txt). If omitted, uses groundtruth.txt"
    )
    parser.add_argument(
        "--output-ply",
        type=str,
        default="dense_scene_reconstruction.ply",
        help="Output filename for dense 3D colored point cloud (e.g. dense_desk.ply)"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.008,
        help="Voxel downsampling resolution in meters (default: 0.008 = 8mm for crisp detail)"
    )
    parser.add_argument(
        "--keyframe-step",
        type=int,
        default=5,
        help="Sample every N-th frame for reconstruction (default: 5 for balanced speed & density)"
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=3.5,
        help="Maximum depth in meters to include in 3D scene (default: 3.5m)"
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.25,
        help="Minimum depth in meters to include in 3D scene (default: 0.25m)"
    )
    return parser.parse_args()


def load_tum_associations(dataset_path: str) -> List[Tuple[float, str, str]]:
    """
    Synchronizes RGB and Depth image pairs from TUM dataset by timestamp.
    Returns list of (timestamp, rgb_path, depth_path).
    """
    rgb_txt = os.path.join(dataset_path, "rgb.txt")
    depth_txt = os.path.join(dataset_path, "depth.txt")

    if not os.path.exists(rgb_txt) or not os.path.exists(depth_txt):
        raise FileNotFoundError(f"Missing rgb.txt or depth.txt in {dataset_path}")

    rgb_list = []
    with open(rgb_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    rgb_list.append((float(parts[0]), parts[1]))

    depth_list = []
    with open(depth_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    depth_list.append((float(parts[0]), parts[1]))

    depth_timestamps = np.array([d[0] for d in depth_list])

    matched_pairs = []
    for rgb_ts, rgb_rel in rgb_list:
        idx = np.searchsorted(depth_timestamps, rgb_ts)
        candidates = []
        if idx < len(depth_timestamps):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)

        best_idx = min(candidates, key=lambda i: abs(depth_timestamps[i] - rgb_ts))
        if abs(depth_timestamps[best_idx] - rgb_ts) < 0.05:  # within 50ms
            matched_pairs.append((
                rgb_ts,
                os.path.join(dataset_path, rgb_rel),
                os.path.join(dataset_path, depth_list[best_idx][1])
            ))

    return matched_pairs


def load_trajectory(traj_path: str) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """
    Parses TUM trajectory file.
    Returns list of (timestamp, R_wc, t_wc).
    """
    entries = []
    with open(traj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [float(x) for x in line.split()]
            if len(parts) >= 8:
                ts = parts[0]
                tx, ty, tz = parts[1:4]
                qx, qy, qz, qw = parts[4:8]
                R = quaternion_to_rotation_matrix(np.array([qx, qy, qz, qw]))
                t = np.array([tx, ty, tz], dtype=np.float64)
                entries.append((ts, R, t))
    return entries


def export_binary_ply(
    filename: str,
    points: np.ndarray,
    colors: np.ndarray
) -> None:
    """
    Fast binary little-endian PLY exporter.
    Saves hundreds of thousands of colored points in milliseconds.
    """
    num_points = len(points)
    header = (
        f"ply\n"
        f"format binary_little_endian 1.0\n"
        f"element vertex {num_points}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"property uchar red\n"
        f"property uchar green\n"
        f"property uchar blue\n"
        f"end_header\n"
    )

    vertex_dtype = np.dtype([
        ('x', '<f4'),
        ('y', '<f4'),
        ('z', '<f4'),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1')
    ])

    vertex_data = np.empty(num_points, dtype=vertex_dtype)
    vertex_data['x'] = points[:, 0]
    vertex_data['y'] = points[:, 1]
    vertex_data['z'] = points[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]

    with open(filename, 'wb') as f:
        f.write(header.encode('ascii'))
        vertex_data.tofile(f)


def reconstruct_dense_scene(
    dataset_path: str,
    trajectory_path: Optional[str] = None,
    output_ply: str = "dense_scene.ply",
    voxel_size: float = 0.008,
    keyframe_step: int = 5,
    min_depth: float = 0.25,
    max_depth: float = 3.5
) -> None:
    print("\n" + "=" * 65)
    print(" Dense 3D Scene Reconstruction (RGB-D Fusion)")
    print(f" Dataset:    {dataset_path}")
    print(f" Output:     {output_ply}")
    print(f" Voxel Size: {voxel_size * 1000:.1f} mm")
    print(f" Step:       Every {keyframe_step} frames")
    print("=" * 65)

    # 1. Determine Trajectory file
    if trajectory_path is None:
        gt_path = os.path.join(dataset_path, "groundtruth.txt")
        if os.path.exists(gt_path):
            trajectory_path = gt_path
            print(f"[Trajectory] Using synchronized dataset trajectory: {gt_path}")
        else:
            raise FileNotFoundError(f"No trajectory file provided and groundtruth.txt not found in {dataset_path}")
    else:
        print(f"[Trajectory] Using trajectory file: {trajectory_path}")

    traj_entries = load_trajectory(trajectory_path)
    if not traj_entries:
        raise ValueError(f"No valid camera poses in {trajectory_path}")
    print(f"[Trajectory] Loaded {len(traj_entries)} camera poses.")

    # 2. Synchronize RGB + Depth
    pairs = load_tum_associations(dataset_path)
    print(f"[Dataset] Synchronized {len(pairs)} RGB-D frame pairs.")

    traj_timestamps = np.array([t[0] for t in traj_entries])

    # 3. TUM Freiburg 1 Camera Intrinsics
    fx, fy = 517.3, 516.5
    cx, cy = 318.6, 255.3

    all_points_list = []
    all_colors_list = []

    sampled_count = 0
    start_time = time.time()

    # Pre-generate pixel grid
    sample_img = cv2.imread(pairs[0][1])
    h, w = sample_img.shape[:2]
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))

    for idx in range(0, len(pairs), keyframe_step):
        rgb_ts, rgb_file, depth_file = pairs[idx]

        # Find matching trajectory pose
        c_idx = np.searchsorted(traj_timestamps, rgb_ts)
        cands = []
        if c_idx < len(traj_timestamps):
            cands.append(c_idx)
        if c_idx > 0:
            cands.append(c_idx - 1)

        best_c = min(cands, key=lambda i: abs(traj_timestamps[i] - rgb_ts))
        if abs(traj_timestamps[best_c] - rgb_ts) > 0.06:
            continue

        _, R_wc, t_wc = traj_entries[best_c]

        # Load RGB and Depth
        rgb = cv2.imread(rgb_file)
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue

        # TUM depth scaling: 16-bit PNG, 5000 units = 1 meter
        depth_m = depth.astype(np.float32) / 5000.0

        valid_mask = (depth_m > min_depth) & (depth_m < max_depth)
        if np.sum(valid_mask) < 100:
            continue

        z = depth_m[valid_mask]
        x = (u_grid[valid_mask] - cx) * z / fx
        y = (v_grid[valid_mask] - cy) * z / fy

        pts_c = np.column_stack((x, y, z))
        # Transform camera to world coordinates: P_w = R_wc * P_c + t_wc
        pts_w = (R_wc @ pts_c.T + t_wc.reshape(3, 1)).T

        colors = rgb[valid_mask]

        all_points_list.append(pts_w)
        all_colors_list.append(colors)
        sampled_count += 1

        if sampled_count % 15 == 0:
            total_raw = sum(len(p) for p in all_points_list)
            print(f"  Processed {sampled_count} keyframes... Cumulative raw points: {total_raw:,}")

    if not all_points_list:
        print("Error: No 3D points could be reconstructed. Check dataset path and timestamps.")
        return

    raw_points = np.vstack(all_points_list)
    raw_colors = np.vstack(all_colors_list)
    print(f"\n[Fusion] Total raw points unprojected: {len(raw_points):,}")

    # 4. Fast Voxel Grid Downsampling & Outlier Pruning
    print(f"[Filtering] Voxel grid downsampling at {voxel_size * 1000:.1f} mm resolution...")
    voxel_coords = np.floor(raw_points / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(voxel_coords, axis=0, return_index=True)

    fused_points = raw_points[unique_idx]
    fused_colors = raw_colors[unique_idx]
    print(f"[Filtering] Cleaned and fused into {len(fused_points):,} distinct surface points.")

    # 5. Export to Binary PLY
    export_binary_ply(output_ply, fused_points, fused_colors)
    elapsed = time.time() - start_time

    file_size_mb = os.path.getsize(output_ply) / (1024 * 1024)
    print("\n" + "=" * 65)
    print(f" SUCCESS: 3D Scene Reconstruction Complete in {elapsed:.2f} s")
    print(f" Output File:  {output_ply}")
    print(f" File Size:    {file_size_mb:.2f} MB")
    print(f" Total Points: {len(fused_points):,} (fully colored RGB)")
    print("=" * 65)
    print("\nTo view your reconstructed 3D scene:")
    print(f"  - Open Blender > File > Import > Stanford (.ply) > '{output_ply}'")
    print(f"  - Or open directly in MeshLab, CloudCompare, or Windows 3D Viewer.")


def main():
    args = parse_args()
    reconstruct_dense_scene(
        dataset_path=args.dataset_path,
        trajectory_path=args.trajectory,
        output_ply=args.output_ply,
        voxel_size=args.voxel_size,
        keyframe_step=args.keyframe_step,
        min_depth=args.min_depth,
        max_depth=args.max_depth
    )


if __name__ == "__main__":
    main()
