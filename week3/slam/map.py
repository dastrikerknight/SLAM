"""Global Map container: keyframes, map points, culling, binary serialization, and PLY export."""

from typing import Dict, List, Optional, Tuple
import pickle
import numpy as np
import cv2

from slam.map_point import MapPoint
from slam.frame import Frame
from slam.camera import Camera

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
        pts = [mp.point_3d for mp in self.map_points.values() if not mp.is_bad and np.all(np.isfinite(mp.point_3d))]
        return np.array(pts, dtype=np.float64) if pts else np.empty((0, 3), dtype=np.float64)

    def get_all_points_colors(self) -> np.ndarray:
        cols = [mp.color for mp in self.map_points.values() if not mp.is_bad and np.all(np.isfinite(mp.point_3d))]
        return np.array(cols, dtype=np.uint8) if cols else np.empty((0, 3), dtype=np.uint8)

    def get_keyframe_poses(self) -> List[np.ndarray]:
        return [kf.get_inverse_pose() for kf in self.keyframes.values() if np.all(np.isfinite(kf.get_inverse_pose()))]

    def get_trajectory_poses(self) -> List[np.ndarray]:
        return [f.get_inverse_pose() for f in self.all_frames if f.has_pose and np.all(np.isfinite(f.get_inverse_pose()))]

    def get_trajectory_centers(self) -> np.ndarray:
        centers = [f.get_camera_center() for f in self.all_frames if f.has_pose and np.all(np.isfinite(f.get_camera_center()))]
        return np.array(centers, dtype=np.float64) if centers else np.empty((0, 3), dtype=np.float64)

    def cull_map_points(self, min_observations: int = 2, current_kf_id: Optional[int] = None) -> int:
        prune = []
        for pid, mp in self.map_points.items():
            if mp.is_bad:
                prune.append(pid)
                continue
            # 3-frame grace period so freshly triangulated points survive to be tracked
            if current_kf_id is not None and isinstance(mp.first_frame_id, (int, np.integer)):
                if (current_kf_id - mp.first_frame_id) < 3:
                    continue
            if len(mp.observations) < min_observations:
                prune.append(pid)

        for pid in prune:
            self.remove_map_point(pid)
        return len(prune)

    def save_binary(self, path: str) -> bool:
        data = {
            "keyframes": [
                {
                    "id": kf.id,
                    "ts": kf.timestamp,
                    "pose": kf.pose,
                    "pts": np.array([kp.pt for kp in kf.kps], dtype=np.float32) if kf.kps else np.empty((0, 2)),
                    "des": kf.des,
                    "mp_ids": [mp.id if mp else None for mp in kf.map_points]
                }
                for kf in self.keyframes.values()
            ],
            "map_points": [
                {
                    "id": mp.id,
                    "pt": mp.point_3d,
                    "col": mp.color,
                    "obs": list(mp.observations.items())
                }
                for mp in self.map_points.values() if not mp.is_bad
            ],
            "trajectory": [(f.timestamp, f.get_inverse_pose()) for f in self.all_frames if f.has_pose]
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    @classmethod
    def load_binary(cls, path: str, camera: Camera) -> "Map":
        with open(path, "rb") as f:
            data = pickle.load(f)

        slam_map = cls()
        for m in data["map_points"]:
            mp = MapPoint(m["pt"], 0, m["col"])
            mp.id = m["id"]
            for kid, feat_idx in m["obs"]:
                mp.add_observation(kid, feat_idx)
            slam_map.add_map_point(mp)

        for k in data["keyframes"]:
            kps = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=20) for p in k["pts"]]
            kf = Frame(k["id"], k["ts"], np.zeros((camera.height, camera.width), dtype=np.uint8), camera)
            kf.set_features(kps, k["des"])
            kf.set_pose(k["pose"])
            kf.is_keyframe = True
            for feat_idx, mp_id in enumerate(k["mp_ids"]):
                if mp_id is not None and mp_id in slam_map.map_points:
                    kf.map_points[feat_idx] = slam_map.map_points[mp_id]
            slam_map.add_keyframe(kf)

        return slam_map

    def export_ply(self, path: str) -> bool:
        pts = self.get_all_points_3d()
        cols = self.get_all_points_colors()
        if len(pts) == 0:
            return False

        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(pts, cols):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        return True

    def export_tum_trajectory(self, path: str) -> bool:
        valid_frames = [f for f in self.all_frames if f.has_pose]
        if not valid_frames:
            return False

        with open(path, "w") as f:
            for f_obj in valid_frames:
                T_wc = f_obj.get_inverse_pose()
                t = T_wc[:3, 3]
                R = T_wc[:3, :3]
                qw = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
                if qw > 1e-4:
                    qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
                    qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
                    qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)
                else:
                    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

                f.write(f"{f_obj.timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
        return True
