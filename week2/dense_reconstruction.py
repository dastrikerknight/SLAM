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
    p = argparse.ArgumentParser(description="Dense 3D Scene Reconstruction from Video / RGB-D Sequences")
    p.add_argument("--dataset-path", type=str, required=True, help="Path to TUM RGB-D sequence folder")
    p.add_argument("--trajectory", type=str, default=None, help="Estimated trajectory file in TUM format (uses groundtruth.txt if omitted)")
    p.add_argument("--output-ply", type=str, default="dense_scene_reconstruction.ply", help="Output .ply file path")
    p.add_argument("--voxel-size", type=float, default=0.008, help="Voxel size in meters for downsampling (default 8mm)")
    p.add_argument("--keyframe-step", type=int, default=5, help="Subsample step across frames")
    p.add_argument("--max-depth", type=float, default=3.5, help="Max depth in meters")
    p.add_argument("--min-depth", type=float, default=0.25, help="Min depth in meters")
    return p.parse_args()

def load_tum_associations(dataset_path: str) -> List[Tuple[float, str, str]]:
    rgb_txt = os.path.join(dataset_path, "rgb.txt")
    depth_txt = os.path.join(dataset_path, "depth.txt")
    if not os.path.exists(rgb_txt) or not os.path.exists(depth_txt):
        raise FileNotFoundError(f"Missing rgb.txt or depth.txt in {dataset_path}")

    rgb_list, depth_list = [], []
    with open(rgb_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts and not parts[0].startswith("#") and len(parts) >= 2:
                rgb_list.append((float(parts[0]), parts[1]))
    with open(depth_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts and not parts[0].startswith("#") and len(parts) >= 2:
                depth_list.append((float(parts[0]), parts[1]))

    depth_ts = np.array([d[0] for d in depth_list])
    matched = []
    for r_ts, r_path in rgb_list:
        idx = int(np.argmin(np.abs(depth_ts - r_ts)))
        if abs(depth_ts[idx] - r_ts) < 0.04:
            matched.append((r_ts, r_path, depth_list[idx][1]))
    return matched

def load_tum_poses(pose_file: str) -> Dict[float, np.ndarray]:
    poses = {}
    with open(pose_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts and not parts[0].startswith("#") and len(parts) >= 8:
                ts = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = [tx, ty, tz]
                poses[ts] = T
    return poses

def interpolate_pose(ts: float, poses: Dict[float, np.ndarray]) -> Optional[np.ndarray]:
    if not poses:
        return None
    timestamps = np.array(sorted(poses.keys()))
    idx = int(np.searchsorted(timestamps, ts))
    if idx == 0:
        return poses[timestamps[0]]
    if idx >= len(timestamps):
        return poses[timestamps[-1]]
    t0, t1 = timestamps[idx - 1], timestamps[idx]
    return poses[t0] if abs(ts - t0) <= abs(ts - t1) else poses[t1]

def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, colors
    coords = np.floor(points / voxel_size).astype(np.int64)
    # Pack coords into 64-bit integer keys for ultra fast hash grouping
    keys = coords[:, 0] * 73856093 ^ coords[:, 1] * 19349663 ^ coords[:, 2] * 83492791
    _, unique_indices = np.unique(keys, return_index=True)
    return points[unique_indices], colors[unique_indices]

def save_ply_binary(filename: str, points: np.ndarray, colors: np.ndarray) -> bool:
    n = len(points)
    pts = points.astype(np.float32)
    cols = colors.astype(np.uint8)
    vertex_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")
    ])
    arr = np.empty(n, dtype=vertex_dtype)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr["red"], arr["green"], arr["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())
    return True

def main():
    args = parse_args()
    pairs = load_tum_associations(args.dataset_path)
    pose_file = args.trajectory or os.path.join(args.dataset_path, "groundtruth.txt")
    poses = load_tum_poses(pose_file)

    # TUM Freiburg1 intrinsics
    fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
    sampled = pairs[::args.keyframe_step]

    all_pts, all_cols = [], []
    for ts, rgb_rel, depth_rel in sampled:
        pose = interpolate_pose(ts, poses)
        if pose is None:
            continue

        rgb_path = os.path.join(args.dataset_path, rgb_rel)
        depth_path = os.path.join(args.dataset_path, depth_rel)
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            continue

        z = depth.astype(np.float32) / 5000.0
        mask = (z > args.min_depth) & (z < args.max_depth)
        v, u = np.where(mask)
        z_v = z[v, u]

        x_c = (u - cx) * z_v / fx
        y_c = (v - cy) * z_v / fy
        pts_c = np.column_stack((x_c, y_c, z_v))
        pts_w = (pose[:3, :3] @ pts_c.T + pose[:3, 3:4]).T

        bgr = rgb[v, u]
        rgb_cols = np.column_stack((bgr[:, 2], bgr[:, 1], bgr[:, 0]))

        all_pts.append(pts_w)
        all_cols.append(rgb_cols)

    pts = np.vstack(all_pts)
    cols = np.vstack(all_cols)
    print(f"Raw points: {len(pts):,}. Downsampling at {args.voxel_size*1000:.1f}mm resolution...")
    pts_sub, cols_sub = voxel_downsample(pts, cols, args.voxel_size)
    print(f"Final clean points: {len(pts_sub):,}. Exporting to {args.output_ply}...")
    save_ply_binary(args.output_ply, pts_sub, cols_sub)
    print("Done!")

if __name__ == "__main__":
    main()
