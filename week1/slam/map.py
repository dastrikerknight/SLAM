from typing import Dict, List
import numpy as np
from slam.map_point import MapPoint
from slam.frame import Frame

class Map:
    def __init__(self):
        self.map_points: Dict[int, MapPoint] = {}
        self.keyframes: Dict[int, Frame] = {}
        self.all_frames: List[Frame] = []

    def add_keyframe(self, kf: Frame):
        kf.is_keyframe = True
        self.keyframes[kf.id] = kf

    def add_frame(self, f: Frame):
        self.all_frames.append(f)

    def add_map_point(self, mp: MapPoint):
        self.map_points[mp.id] = mp

    def remove_map_point(self, pid: int):
        mp = self.map_points.pop(pid, None)
        if mp:
            mp.mark_bad()

    def get_all_points_3d(self) -> np.ndarray:
        pts = [mp.point_3d for mp in self.map_points.values() if not mp.is_bad]
        return np.array(pts, dtype=np.float64) if pts else np.empty((0, 3), dtype=np.float64)

    def get_all_points_colors(self) -> np.ndarray:
        cols = [mp.color for mp in self.map_points.values() if not mp.is_bad]
        return np.array(cols, dtype=np.uint8) if cols else np.empty((0, 3), dtype=np.uint8)

    def get_keyframe_poses(self) -> List[np.ndarray]:
        return [kf.get_inverse_pose() for kf in self.keyframes.values()]

    def get_trajectory_poses(self) -> List[np.ndarray]:
        return [f.get_inverse_pose() for f in self.all_frames if f.has_pose]

    def get_trajectory_centers(self) -> np.ndarray:
        centers = [f.get_camera_center() for f in self.all_frames if f.has_pose]
        return np.array(centers, dtype=np.float64) if centers else np.empty((0, 3), dtype=np.float64)

    def cull_map_points(self, min_observations: int = 2) -> int:
        prune = [pid for pid, mp in self.map_points.items() if mp.is_bad or len(mp.observations) < min_observations]
        for pid in prune:
            self.remove_map_point(pid)
        return len(prune)

    def export_ply(self, path: str) -> bool:
        pts = self.get_all_points_3d()
        cols = self.get_all_points_colors()
        if len(pts) == 0:
            return False

        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for i, p in enumerate(pts):
                c = cols[i] if len(cols) > i else [200, 200, 200]
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        return True

    def export_tum_trajectory(self, path: str) -> bool:
        poses = [(f.timestamp, f.get_inverse_pose()) for f in self.all_frames if f.has_pose]
        if not poses:
            return False

        with open(path, "w") as f:
            for ts, T in poses:
                t = T[:3, 3]
                R = T[:3, :3]
                tr = np.trace(R)
                if tr > 0:
                    s = np.sqrt(tr + 1.0) * 2.0
                    qw = 0.25 * s
                    qx = (R[2, 1] - R[1, 2]) / s
                    qy = (R[0, 2] - R[2, 0]) / s
                    qz = (R[1, 0] - R[0, 1]) / s
                elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                    s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                    qw = (R[2, 1] - R[1, 2]) / s
                    qx = 0.25 * s
                    qy = (R[0, 1] + R[1, 0]) / s
                    qz = (R[0, 2] + R[2, 0]) / s
                elif R[1, 1] > R[2, 2]:
                    s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                    qw = (R[0, 2] - R[2, 0]) / s
                    qx = (R[0, 1] + R[1, 0]) / s
                    qy = 0.25 * s
                    qz = (R[1, 2] + R[2, 1]) / s
                else:
                    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                    qw = (R[1, 0] - R[0, 1]) / s
                    qx = (R[0, 2] + R[2, 0]) / s
                    qy = (R[1, 2] + R[2, 1]) / s
                    qz = 0.25 * s
                f.write(f"{ts:.6f} {t[0]:.4f} {t[1]:.4f} {t[2]:.4f} {qx:.5f} {qy:.5f} {qz:.5f} {qw:.5f}\n")
        return True
