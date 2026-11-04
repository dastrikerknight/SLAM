from typing import List, Dict, Tuple, Optional
import os
import sys
import json
import numpy as np
import cv2
import open3d as o3d

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantic_module import SemanticSegmenter

class TextureProjector:
    """Projects photographic keyframes and semantic masks onto 3D mesh surfaces."""

    def __init__(self, texture_size: int = 2048):
        self.texture_size = texture_size

    def colorize_mesh_from_keyframes(
        self,
        mesh: o3d.geometry.TriangleMesh,
        keyframes: List[Dict],
        semantic_segmenter: Optional[SemanticSegmenter] = None
    ) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh, Dict]:
        """
        Projects photographic color and semantic classes onto mesh vertices.
        Returns:
            (textured_mesh, semantic_colored_mesh, semantics_metadata)
        """
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else np.zeros_like(vertices)
        n_verts = len(vertices)

        if n_verts == 0 or not keyframes:
            return mesh, mesh, {}

        accum_colors = np.zeros((n_verts, 3), dtype=np.float64)
        accum_weights = np.zeros(n_verts, dtype=np.float64)
        semantic_votes: Dict[int, Dict[int, int]] = {i: {} for i in range(n_verts)}

        # Process each keyframe projection
        for kf in keyframes:
            img_bgr = kf["img"]
            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            T_cw = kf["T_cw"]
            K = kf["K"]
            sem_mask = kf.get("semantic_mask", None)

            # Camera center in world coordinates
            cam_center = -T_cw[:3, :3].T @ T_cw[:3, 3]

            # Transform vertices to camera frame: p_c = R * p_w + t
            pts_c = (T_cw[:3, :3] @ vertices.T).T + T_cw[:3, 3]
            valid_z = pts_c[:, 2] > 0.25

            if not np.any(valid_z):
                continue

            # Project only forward-facing points to prevent divide-by-zero
            z_safe = np.where(valid_z, pts_c[:, 2], 1.0)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            u = np.where(valid_z, np.round(fx * (pts_c[:, 0] / z_safe) + cx), -1).astype(np.int64)
            v = np.where(valid_z, np.round(fy * (pts_c[:, 1] / z_safe) + cy), -1).astype(np.int64)

            # Check image boundaries
            in_view = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)

            # Viewing angle cosine: n_w . view_dir
            view_dirs = cam_center - vertices
            view_norms = np.linalg.norm(view_dirs, axis=1, keepdims=True)
            view_dirs = view_dirs / np.maximum(view_norms, 1e-6)

            cos_angles = np.sum(normals * view_dirs, axis=1)
            valid_angles = cos_angles > 0.15
            active_mask = in_view & valid_angles

            active_indices = np.where(active_mask)[0]
            if len(active_indices) == 0:
                continue

            sampled_u = u[active_indices]
            sampled_v = v[active_indices]
            sampled_colors = img_rgb[sampled_v, sampled_u].astype(np.float64) / 255.0
            weights = cos_angles[active_indices]

            for i_local, v_idx in enumerate(active_indices):
                w_val = weights[i_local]
                accum_colors[v_idx] += sampled_colors[i_local] * w_val
                accum_weights[v_idx] += w_val

                if sem_mask is not None:
                    cid = int(sem_mask[sampled_v[i_local], sampled_u[i_local]])
                    semantic_votes[v_idx][cid] = semantic_votes[v_idx].get(cid, 0) + 1

        # Normalize accumulated colors
        has_color = accum_weights > 0
        final_colors = np.full((n_verts, 3), 0.7, dtype=np.float64)
        final_colors[has_color] = accum_colors[has_color] / accum_weights[has_color, None]

        photo_mesh = o3d.geometry.TriangleMesh(mesh)
        photo_mesh.vertex_colors = o3d.utility.Vector3dVector(final_colors)

        # Generate semantic mesh & 3D bounding boxes
        palette = SemanticSegmenter.PALETTE / 255.0
        categories = semantic_segmenter.categories if semantic_segmenter else {}
        sem_colors = np.full((n_verts, 3), 0.25, dtype=np.float64)
        sem_labels_per_vert = np.zeros(n_verts, dtype=np.int32)
        object_points: Dict[int, List[np.ndarray]] = {}

        for v_idx in range(n_verts):
            votes = semantic_votes[v_idx]
            if votes:
                top_cid = max(votes.keys(), key=lambda c: votes[c])
                sem_labels_per_vert[v_idx] = top_cid
                sem_colors[v_idx] = palette[min(top_cid, len(palette) - 1)]
                if top_cid > 0: # not background
                    if top_cid not in object_points:
                        object_points[top_cid] = []
                    object_points[top_cid].append(vertices[v_idx])

        semantic_mesh = o3d.geometry.TriangleMesh(mesh)
        semantic_mesh.vertex_colors = o3d.utility.Vector3dVector(sem_colors)

        # Compute 3D bounding boxes and physical dimensions for detected objects
        semantics_meta = {}
        for cid, pts in object_points.items():
            if len(pts) >= 50:
                pts_arr = np.array(pts)
                mins = pts_arr.min(axis=0)
                maxs = pts_arr.max(axis=0)
                center = (mins + maxs) / 2.0
                dims = maxs - mins
                name = categories[cid] if cid < len(categories) else f"object_{cid}"
                semantics_meta[name] = {
                    "class_id": int(cid),
                    "point_count": len(pts),
                    "center": [float(c) for c in center],
                    "dimensions_xyz_meters": [float(d) for d in dims],
                    "bbox_min": [float(m) for m in mins],
                    "bbox_max": [float(m) for m in maxs]
                }

        return photo_mesh, semantic_mesh, semantics_meta

    def export_textured_model(
        self,
        output_dir: str,
        photo_mesh: o3d.geometry.TriangleMesh,
        semantic_mesh: o3d.geometry.TriangleMesh,
        semantics_meta: Dict
    ) -> Dict[str, str]:
        """Exports textured OBJ/MTL model, semantic PLY mesh, and semantic metadata."""
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        # 1. Export vertex-colored photographic OBJ + PLY
        obj_path = os.path.join(output_dir, "scene_mesh.obj")
        ply_path = os.path.join(output_dir, "scene_mesh.ply")
        o3d.io.write_triangle_mesh(obj_path, photo_mesh, write_vertex_colors=True)
        o3d.io.write_triangle_mesh(ply_path, photo_mesh)
        paths["mesh_obj"] = obj_path
        paths["mesh_ply"] = ply_path

        # 2. Export semantic mesh
        sem_ply_path = os.path.join(output_dir, "semantic_mesh.ply")
        o3d.io.write_triangle_mesh(sem_ply_path, semantic_mesh)
        paths["semantic_ply"] = sem_ply_path

        # 3. Export semantic scene graph JSON
        meta_path = os.path.join(output_dir, "scene_semantics.json")
        with open(meta_path, "w") as f:
            json.dump(semantics_meta, f, indent=2)
        paths["semantics_json"] = meta_path

        return paths

if __name__ == "__main__":
    projector = TextureProjector()
    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=0.8, depth=0.5)
    mesh.compute_vertex_normals()

    dummy_kf = {
        "img": np.full((480, 640, 3), 200, dtype=np.uint8),
        "T_cw": np.eye(4),
        "K": np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
        "semantic_mask": np.full((480, 640), 11, dtype=np.uint8) # desk
    }
    photo, sem, meta = projector.colorize_mesh_from_keyframes(mesh, [dummy_kf])
    assert photo.has_vertex_colors()
    assert sem.has_vertex_colors()
    print(f"TextureProjector verified: colorized {len(photo.vertices)} vertices, detected {list(meta.keys())}")
