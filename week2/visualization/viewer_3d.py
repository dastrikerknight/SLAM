from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from slam.map import Map

def draw_camera_frustum(ax, T_wc: np.ndarray, scale: float = 0.5, color: str = "red"):
    R = T_wc[:3, :3]
    t = T_wc[:3, 3]

    w, h, d = scale * 0.6, scale * 0.45, scale
    corners_c = np.array([
        [0, 0, 0],
        [-w, -h, d],
        [w, -h, d],
        [w, h, d],
        [-w, h, d]
    ], dtype=np.float64)
    corners_w = (R @ corners_c.T).T + t

    for i in range(1, 5):
        ax.plot([corners_w[0, 0], corners_w[i, 0]],
                [corners_w[0, 1], corners_w[i, 1]],
                [corners_w[0, 2], corners_w[i, 2]], color=color, linewidth=1.2)

    base = [1, 2, 3, 4, 1]
    ax.plot(corners_w[base, 0], corners_w[base, 1], corners_w[base, 2], color=color, linewidth=1.2)

class Viewer3D:
    def __init__(self, title: str = "Monocular Visual SLAM - 3D Map"):
        self.title = title

    def plot_slam_results(
        self,
        slam_map: Map,
        gt_centers: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(self.title, fontsize=13, pad=15)

        points_3d = slam_map.get_all_points_3d()
        if len(points_3d) > 0:
            scatter = ax.scatter(
                points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                c=points_3d[:, 2], cmap="viridis", s=4, alpha=0.6,
                label=f"Points ({len(points_3d)})"
            )
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
            cbar.set_label("Z Depth (m)", rotation=270, labelpad=15)

        est_centers = slam_map.get_trajectory_centers()
        if len(est_centers) > 1:
            ax.plot(
                est_centers[:, 0], est_centers[:, 1], est_centers[:, 2],
                color="blue", linewidth=2.2, label=f"Estimated ({len(est_centers)})"
            )

        kf_poses = slam_map.get_keyframe_poses()
        step = max(1, len(kf_poses) // 15)
        for idx, kf_pose in enumerate(kf_poses):
            if idx % step == 0 or idx == len(kf_poses) - 1:
                draw_camera_frustum(ax, kf_pose, scale=0.4, color="orange")

        if gt_centers is not None and len(gt_centers) > 1:
            ax.plot(
                gt_centers[:, 0], gt_centers[:, 1], gt_centers[:, 2],
                color="green", linestyle="--", linewidth=1.8, label="Ground Truth"
            )

        ax.set_xlabel("X (m)", labelpad=10)
        ax.set_ylabel("Y (m)", labelpad=10)
        ax.set_zlabel("Z (m)", labelpad=10)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.5)

        if len(points_3d) > 0 or len(est_centers) > 0:
            all_pts = [p for p in (points_3d, est_centers, gt_centers) if p is not None and len(p) > 0]
            comb = np.vstack(all_pts)
            max_range = (comb.max(axis=0) - comb.min(axis=0)).max() / 2.0
            mid = (comb.max(axis=0) + comb.min(axis=0)) * 0.5

            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[Viewer3D] Saved 3D plot to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)
