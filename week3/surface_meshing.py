import os
import argparse
from typing import Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree, Delaunay

def estimate_normals(points: np.ndarray, k: int = 15, view_dir: Optional[np.ndarray] = None) -> np.ndarray:
    tree = cKDTree(points)
    _, idxs = tree.query(points, k=k)

    normals = np.zeros_like(points)
    for i, nbr_idx in enumerate(idxs):
        nbrs = points[nbr_idx]
        cov = np.cov(nbrs.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]

        # Orient normals towards camera view direction if given, else towards origin
        v = view_dir[i] if view_dir is not None else -points[i]
        if np.dot(normal, v) < 0:
            normal = -normal
        normals[i] = normal

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return normals / norms

def reconstruct_surface_mesh(
    points: np.ndarray,
    colors: np.ndarray,
    max_edge_length: float = 0.12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(points) < 4:
        raise ValueError("Insufficient points for meshing")

    # Project to 2D using PCA dominant plane for Delaunay meshing
    pts_mean = points.mean(axis=0)
    cov = np.cov((points - pts_mean).T)
    _, eigvecs = np.linalg.eigh(cov)
    proj_axes = eigvecs[:, 1:3]
    pts_2d = (points - pts_mean) @ proj_axes

    tri = Delaunay(pts_2d)
    simplices = tri.simplices

    # Filter out long skinny triangles that jump across empty space
    v0, v1, v2 = points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]]
    d01 = np.linalg.norm(v0 - v1, axis=1)
    d12 = np.linalg.norm(v1 - v2, axis=1)
    d20 = np.linalg.norm(v2 - v0, axis=1)

    valid_mask = (d01 < max_edge_length) & (d12 < max_edge_length) & (d20 < max_edge_length)
    filtered_faces = simplices[valid_mask]

    normals = estimate_normals(points, k=15)
    return points, filtered_faces, colors, normals

def export_obj(path: str, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, normals: np.ndarray):
    with open(path, "w") as f:
        f.write("# Monocular SLAM Week 3 Surface Reconstruction\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")

        # Vertex positions with RGB colors [0..1]
        for i, (v, c) in enumerate(zip(vertices, colors)):
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]/255.0:.3f} {c[1]/255.0:.3f} {c[2]/255.0:.3f}\n")

        for vn in normals:
            f.write(f"vn {vn[0]:.4f} {vn[1]:.4f} {vn[2]:.4f}\n")

        # Faces (1-indexed in OBJ format)
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

def export_ply_mesh(path: str, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")

        for v, c in zip(vertices, colors):
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def main():
    parser = argparse.ArgumentParser(description="3D Watertight Surface Mesh Reconstruction")
    parser.add_argument("--input-ply", type=str, required=True, help="Input point cloud (.ply)")
    parser.add_argument("--output-obj", type=str, default="reconstructed_mesh.obj", help="Output .obj mesh")
    parser.add_argument("--max-edge", type=float, default=0.10, help="Max triangle edge length in meters")
    args = parser.parse_args()

    # Load input point cloud
    pts, cols = [], []
    with open(args.input_ply, "r", encoding="ascii", errors="ignore") as f:
        in_header = True
        for line in f:
            if in_header:
                if line.strip() == "end_header":
                    in_header = False
                continue
            p = line.strip().split()
            if len(p) >= 6:
                pts.append([float(p[0]), float(p[1]), float(p[2])])
                cols.append([int(p[3]), int(p[4]), int(p[5])])

    pts_arr = np.array(pts, dtype=np.float64)
    cols_arr = np.array(cols, dtype=np.uint8)

    print(f"Loaded {len(pts_arr):,} points. Reconstructing solid 3D triangle mesh...")
    v, faces, c, normals = reconstruct_surface_mesh(pts_arr, cols_arr, max_edge_length=args.max_edge)
    print(f"Mesh generated: {len(v):,} vertices, {len(faces):,} triangle faces.")

    export_obj(args.output_obj, v, faces, c, normals)
    print(f"Saved OBJ mesh to: {args.output_obj}")

if __name__ == "__main__":
    main()
