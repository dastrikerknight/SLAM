from typing import Optional
import os
import numpy as np
import cv2
import open3d as o3d

class OrbitRenderer:
    """Renders a smooth 360-degree cinematic fly-through video of a 3D reconstructed scene."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

    def render_orbit_video(
        self,
        mesh: o3d.geometry.TriangleMesh,
        output_mp4: str,
        num_frames: int = 90,
        elevation_angle: float = 25.0
    ) -> bool:
        """
        Renders a 360-degree circular camera orbit around the mesh bounding center.
        Exports a 30 FPS MP4 video and a showcase still image.
        """
        if len(mesh.vertices) == 0:
            print("[OrbitRenderer] Warning: mesh has 0 vertices, skipping video render.")
            return False

        os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)

        center = mesh.get_center()
        min_bound = mesh.get_min_bound()
        max_bound = mesh.get_max_bound()
        extent = max_bound - min_bound
        radius = float(max(np.linalg.norm(extent) * 0.65, 1.2))

        # Offscreen renderer using Vulkan hardware acceleration
        try:
            renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
            renderer.scene.set_background([0.08, 0.08, 0.10, 1.0]) # dark studio background
            
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultLit"
            renderer.scene.add_geometry("scene_mesh", mesh, mat)
        except Exception as e:
            print(f"[OrbitRenderer] Hardware offscreen renderer init failed: {e}")
            return False

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_mp4, fourcc, self.fps, (self.width, self.height))

        # Up vector in SLAM / TUM coordinates (Y is down in OpenCV, so up is negative Y or positive Z)
        up_vec = [0.0, -1.0, 0.0]
        elev_rad = np.radians(elevation_angle)

        first_frame = None

        for i in range(num_frames):
            theta = 2.0 * np.pi * (i / float(num_frames))
            # Circular orbit in X-Z plane with elevation offset in Y
            eye_x = center[0] + radius * np.cos(theta) * np.cos(elev_rad)
            eye_z = center[2] + radius * np.sin(theta) * np.cos(elev_rad)
            eye_y = center[1] - radius * np.sin(elev_rad)
            eye = [eye_x, eye_y, eye_z]

            renderer.setup_camera(55.0, center, eye, up_vec)
            img_o3d = renderer.render_to_image()
            img_np = np.asarray(img_o3d) # RGB uint8

            # Convert RGB to BGR for OpenCV VideoWriter
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            writer.write(img_bgr)

            if i == 0:
                first_frame = img_bgr.copy()

        writer.release()

        # Save showcase still image alongside video
        snapshot_path = os.path.splitext(output_mp4)[0] + "_showcase.png"
        if first_frame is not None:
            cv2.imwrite(snapshot_path, first_frame)

        print(f"[OrbitRenderer] Successfully rendered 360-degree video ({num_frames} frames) -> {output_mp4}")
        return True

if __name__ == "__main__":
    renderer = OrbitRenderer(width=320, height=240, fps=15)
    test_mesh = o3d.geometry.TriangleMesh.create_box()
    test_mesh.compute_vertex_normals()
    test_mesh.paint_uniform_color([0.2, 0.7, 0.3])
    
    test_out = "test_orbit.mp4"
    ok = renderer.render_orbit_video(test_mesh, test_out, num_frames=10)
    assert ok, "Orbit render should succeed"
    assert os.path.exists(test_out), "MP4 file must exist"
    if os.path.exists(test_out):
        os.remove(test_out)
    snap = "test_orbit_showcase.png"
    if os.path.exists(snap):
        os.remove(snap)
    print("OrbitRenderer self-test passed!")
