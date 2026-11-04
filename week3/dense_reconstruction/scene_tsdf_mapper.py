from typing import Union, Optional
import numpy as np
import cv2
import open3d as o3d

class SceneTSDFMapper:
    """Volumetric TSDF fusion engine for dense, watertight 3D scene reconstruction."""

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
            # Origin centered horizontally (-2m to +2m), starting slightly behind camera (-0.5m)
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
        """Filters out flying edge pixels at depth silhouettes where gradient spikes."""
        cleaned = depth_map.copy()
        valid = cleaned > 0.1
        if not np.any(valid):
            return cleaned

        gx = cv2.Sobel(cleaned, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(cleaned, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)

        # Invalidate pixels with extreme depth discontinuity (flying pixels)
        cleaned[(grad_mag > max_grad) & valid] = 0.0
        return cleaned

    def integrate_keyframe(
        self,
        color_bgr: np.ndarray,
        metric_depth: np.ndarray,
        intrinsic_matrix: Union[np.ndarray, o3d.camera.PinholeCameraIntrinsic],
        pose_c2w: np.ndarray
    ) -> None:
        """Fuses a calibrated keyframe RGB-D pair into the TSDF volume."""
        h, w = metric_depth.shape[:2]
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB) if color_bgr.ndim == 3 and color_bgr.shape[2] == 3 else color_bgr

        # Edge discontinuity filtering to prevent surface stretching
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
        """Extracts zero-crossing Marching Cubes mesh, strips floating islands, and smooths normals."""
        mesh = self.volume.extract_triangle_mesh()
        if len(mesh.vertices) == 0:
            return mesh

        mesh.compute_vertex_normals()

        # Prune small disconnected floating fragments if any
        if min_cluster_faces > 0 and len(mesh.triangles) > min_cluster_faces * 2:
            try:
                triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
                triangle_clusters = np.asarray(triangle_clusters)
                cluster_n_triangles = np.asarray(cluster_n_triangles)

                valid_cluster_mask = cluster_n_triangles >= min_cluster_faces
                triangles_to_keep = valid_cluster_mask[triangle_clusters]
                mesh.remove_triangles_by_mask(~triangles_to_keep)
                mesh.remove_unreferenced_vertices()
            except Exception:
                pass

        # Taubin smoothing: eliminates voxel discretization staircases without shrinking geometry
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
    assert len(m.vertices) > 0, "Marching cubes should extract mesh vertices"
    print(f"SceneTSDFMapper verified: extracted {len(m.vertices)} vertices, {len(m.triangles)} triangles")
