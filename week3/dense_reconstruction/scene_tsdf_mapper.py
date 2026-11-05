from typing import Union, Optional
import numpy as np
import cv2
import open3d as o3d


class SceneTSDFMapper:
    def __init__(
        self,
        volume_type: str = "uniform",
        voxel_length: float = 0.015,
        sdf_trunc: float = 0.06,
        depth_trunc: float = 3.5,
        room_length: float = 4.0,
        resolution: int = 256
    ):
        self.volume_type = volume_type
        self.depth_trunc = depth_trunc
        self.sdf_trunc = sdf_trunc

        if volume_type == "uniform":
            # start 0.6m behind cam to catch geometry right at the starting line
            origin = np.array([-room_length / 2.0, -room_length / 2.0, -0.6], dtype=np.float64)
            self.volume = o3d.pipelines.integration.UniformTSDFVolume(
                length=room_length,
                resolution=resolution,
                sdf_trunc=sdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                origin=origin
            )
        else:
            self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=voxel_length,
                sdf_trunc=sdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                volume_unit_resolution=64
            )
        self.integrated_frames = 0

    def clean_depth_edges(self, depth_map: np.ndarray, max_grad: float = 0.15) -> np.ndarray:
        # kill flying pixels at depth edges so we don't get rubber-sheet bridges across occlusions
        cleaned = depth_map.copy()
        valid = cleaned > 0.1
        if not np.any(valid):
            return cleaned

        gx = cv2.Sobel(cleaned, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(cleaned, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)

        cleaned[(grad > max_grad) & valid] = 0.0
        return cleaned

    def integrate_keyframe(
        self,
        color_bgr: np.ndarray,
        metric_depth: np.ndarray,
        intrinsic_matrix: Union[np.ndarray, o3d.camera.PinholeCameraIntrinsic],
        pose_c2w: np.ndarray
    ) -> None:
        h, w = metric_depth.shape[:2]
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB) if color_bgr.ndim == 3 and color_bgr.shape[2] == 3 else color_bgr

        filtered_depth = self.clean_depth_edges(metric_depth)

        color_o3d = o3d.geometry.Image(np.ascontiguousarray(color_rgb, dtype=np.uint8))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(filtered_depth, dtype=np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False
        )

        if isinstance(intrinsic_matrix, np.ndarray):
            intrinsics = o3d.camera.PinholeCameraIntrinsic(
                w, h,
                float(intrinsic_matrix[0, 0]),
                float(intrinsic_matrix[1, 1]),
                float(intrinsic_matrix[0, 2]),
                float(intrinsic_matrix[1, 2])
            )
        else:
            intrinsics = intrinsic_matrix

        extrinsic_w2c = np.linalg.inv(pose_c2w)
        self.volume.integrate(rgbd, intrinsics, extrinsic_w2c)
        self.integrated_frames += 1

    def extract_clean_mesh(
        self,
        min_cluster_faces: int = 150,
        smooth_iterations: int = 5
    ) -> o3d.geometry.TriangleMesh:
        mesh = self.volume.extract_triangle_mesh()
        if len(mesh.vertices) == 0:
            return mesh

        mesh.compute_vertex_normals()

        # toss small disconnected floating specks
        if min_cluster_faces > 0 and len(mesh.triangles) > min_cluster_faces * 2:
            try:
                clusters, n_tris, _ = mesh.cluster_connected_triangles()
                clusters = np.asarray(clusters)
                n_tris = np.asarray(n_tris)

                keep_mask = n_tris >= min_cluster_faces
                tris_to_keep = keep_mask[clusters]
                mesh.remove_triangles_by_mask(~tris_to_keep)
                mesh.remove_unreferenced_vertices()
            except Exception:
                pass

        # taubin smoothing keeps volume while rounding off voxel staircasing
        if smooth_iterations > 0 and len(mesh.vertices) > 0:
            mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
            mesh.compute_vertex_normals()

        return mesh

    def export_mesh(self, filename: str, mesh: Optional[o3d.geometry.TriangleMesh] = None) -> bool:
        if mesh is None:
            mesh = self.extract_clean_mesh()
        if len(mesh.vertices) > 0:
            o3d.io.write_triangle_mesh(filename, mesh)
            return True
        return False


if __name__ == "__main__":
    mapper = SceneTSDFMapper(volume_type="uniform", room_length=4.0, resolution=128)
    dummy_bgr = np.full((240, 320, 3), 180, dtype=np.uint8)
    dummy_depth = np.full((240, 320), 1.5, dtype=np.float32)
    K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])

    mapper.integrate_keyframe(dummy_bgr, dummy_depth, K, np.eye(4))
    m = mapper.extract_clean_mesh(min_cluster_faces=10)
    assert len(m.vertices) > 0
    print(f"TSDF ok: {len(m.vertices)} verts, {len(m.triangles)} tris")
