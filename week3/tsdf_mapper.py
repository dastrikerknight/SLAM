from typing import Union
import numpy as np
import cv2
import open3d as o3d

class TSDFMapper:
    def __init__(
        self,
        voxel_length: float = 0.04,
        sdf_trunc: float = 0.12,
        color_type: o3d.pipelines.integration.TSDFVolumeColorType = o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    ):
        self.voxel_length = voxel_length
        self.sdf_trunc = sdf_trunc
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=color_type,
            volume_unit_resolution=32
        )

    def integrate_keyframe(
        self,
        color_img: np.ndarray,
        metric_depth_map: np.ndarray,
        intrinsic_matrix: Union[np.ndarray, o3d.camera.PinholeCameraIntrinsic],
        pose_c2w: np.ndarray
    ) -> None:
        h, w = metric_depth_map.shape[:2]

        # Ensure color image is uint8 RGB
        if color_img.ndim == 2:
            color_rgb = cv2.cvtColor(color_img, cv2.COLOR_GRAY2RGB)
        elif color_img.shape[2] == 3:
            color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        elif color_img.shape[2] == 4:
            color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGRA2RGB)
        else:
            color_rgb = color_img

        color_o3d = o3d.geometry.Image(np.ascontiguousarray(color_rgb, dtype=np.uint8))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(metric_depth_map, dtype=np.float32))

        # 1. Convert to Open3D RGBDImage with metric scale (1.0) and 4m truncation
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=4.0,
            convert_rgb_to_intensity=False
        )

        # Build Open3D intrinsic object if a 3x3 matrix was supplied
        if isinstance(intrinsic_matrix, np.ndarray):
            fx = float(intrinsic_matrix[0, 0])
            fy = float(intrinsic_matrix[1, 1])
            cx = float(intrinsic_matrix[0, 2])
            cy = float(intrinsic_matrix[1, 2])
            intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        else:
            intrinsics = intrinsic_matrix

        # 2. Invert Camera-to-World pose (c2w) to obtain World-to-Camera (w2c) extrinsics
        extrinsic_w2c = np.linalg.inv(pose_c2w)

        # 3. Fuse RGB-D frame into the scalable voxel grid
        self.volume.integrate(rgbd, intrinsics, extrinsic_w2c)

    def extract_and_save_mesh(self, filename: str) -> o3d.geometry.TriangleMesh:
        # Marching cubes extraction from zero-crossings of TSDF volume
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()

        if len(mesh.vertices) > 0:
            o3d.io.write_triangle_mesh(filename, mesh)
            print(f"[TSDFMapper] Exported mesh ({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles) -> {filename}")
        else:
            print(f"[TSDFMapper] Notice: Extracted TSDF volume returned 0 surface vertices.")
        return mesh

if __name__ == "__main__":
    import os

    # Self-test integration with synthetic keyframe
    mapper = TSDFMapper(voxel_length=0.04, sdf_trunc=0.12)

    h, w = 480, 640
    dummy_bgr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    dummy_depth = np.random.uniform(0.5, 3.0, (h, w)).astype(np.float32)

    K = np.array([
        [525.0, 0.0, 319.5],
        [0.0, 525.0, 239.5],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Identity camera-to-world pose
    pose_c2w = np.eye(4, dtype=np.float64)

    mapper.integrate_keyframe(dummy_bgr, dummy_depth, K, pose_c2w)

    test_out = "test_tsdf_mesh.ply"
    try:
        mesh = mapper.extract_and_save_mesh(test_out)
        assert os.path.exists(test_out), "Mesh file was not created"
        assert len(mesh.vertices) > 0, "No vertices in extracted mesh"
        assert len(mesh.triangles) > 0, "No triangles in extracted mesh"
        print(f"TSDFMapper verified: extracted mesh with {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles -> {test_out}")
    finally:
        if os.path.exists(test_out):
            os.remove(test_out)
