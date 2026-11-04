"""
Interactive 3D Visualizer for Monocular Visual SLAM using Matplotlib.
Plots 3D MapPoint point clouds, camera trajectories, keyframe frustums,
and ground-truth trajectory alignment.
"""

from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from slam.map import Map


def draw_camera_frustum(ax, T_wc: np.ndarray, scale: float = 0.5, color: str = 'red') -> None:
    """Draws a 3D pyramid representing a camera frustum at pose T_wc."""
    if not np.all(np.isfinite(T_wc)):
        return

    R = T_wc[:3, :3]
    t = T_wc[:3, 3]

    # Frustum corners in camera frame
    w = scale * 0.6
    h = scale * 0.45
    d = scale

    corners_c = np.array([
        [0, 0, 0],         # apex
        [-w, -h, d],       # top-left
        [w, -h, d],        # top-right
        [w, h, d],         # bottom-right
        [-w, h, d]         # bottom-left
    ], dtype=np.float64)

    # Transform to world
    corners_w = (R @ corners_c.T).T + t

    # Lines from apex to 4 corners
    for i in range(1, 5):
        ax.plot([corners_w[0, 0], corners_w[i, 0]],
                [corners_w[0, 1], corners_w[i, 1]],
                [corners_w[0, 2], corners_w[i, 2]], color=color, linewidth=1.2)

    # Lines connecting the 4 base corners
    base_indices = [1, 2, 3, 4, 1]
    ax.plot(corners_w[base_indices, 0],
            corners_w[base_indices, 1],
            corners_w[base_indices, 2], color=color, linewidth=1.2)


class Viewer3D:
    def __init__(self, title: str = "Monocular Visual SLAM - 3D Map & Trajectory"):
        self.title = title

    def plot_slam_results(
        self,
        slam_map: Map,
        gt_centers: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Generates 3D visualization of the SLAM map, estimated trajectory, and ground truth.
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(self.title, fontsize=14, fontweight='bold', pad=20)

        # 1. Plot 3D MapPoints
        points_3d = slam_map.get_all_points_3d()
        if len(points_3d) > 0:
            points_3d = points_3d[np.isfinite(points_3d).all(axis=1)]

        if len(points_3d) > 0:
            # Color by depth (Z)
            z_vals = points_3d[:, 2]
            scatter = ax.scatter(
                points_3d[:, 0],
                points_3d[:, 1],
                points_3d[:, 2],
                c=z_vals,
                cmap='viridis',
                s=4,
                alpha=0.6,
                label=f"Map Points ({len(points_3d)})"
            )
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
            cbar.set_label("Z Depth (m)", rotation=270, labelpad=15)

        # 2. Plot Estimated Trajectory
        est_centers = slam_map.get_trajectory_centers()
        if len(est_centers) > 0:
            est_centers = est_centers[np.isfinite(est_centers).all(axis=1)]

        if len(est_centers) > 1:
            ax.plot(
                est_centers[:, 0],
                est_centers[:, 1],
                est_centers[:, 2],
                color='blue',
                linewidth=2.5,
                label=f"Estimated Path ({len(est_centers)} frames)"
            )

        # 3. Plot Keyframe Frustums
        kf_poses = slam_map.get_keyframe_poses()
        valid_kf_poses = [p for p in kf_poses if np.all(np.isfinite(p))]
        for idx, kf_pose in enumerate(valid_kf_poses):
            if idx % max(1, len(valid_kf_poses) // 15) == 0 or idx == len(valid_kf_poses) - 1:
                draw_camera_frustum(ax, kf_pose, scale=0.4, color='orange')

        # 4. Plot Ground Truth (aligned to SLAM map frame for direct visual comparison)
        gt_in_slam = None
        if gt_centers is not None and len(gt_centers) > 1 and len(est_centers) > 1:
            valid_gt = gt_centers[np.isfinite(gt_centers).all(axis=1)]
            min_len = min(len(est_centers), len(valid_gt))
            if min_len > 10:
                try:
                    from evaluate import align_umeyama_sim3
                    s, R, t, _ = align_umeyama_sim3(est_centers[:min_len], valid_gt[:min_len])
                    # Transform ground truth into SLAM frame: gt_in_slam = inv(R) @ (gt - t) / s
                    gt_in_slam = (R.T @ ((valid_gt[:min_len] - t) / max(s, 1e-4)).T).T
                    ax.plot(
                        gt_in_slam[:, 0],
                        gt_in_slam[:, 1],
                        gt_in_slam[:, 2],
                        color='green',
                        linestyle='--',
                        linewidth=2.0,
                        label=f"Ground Truth (Sim3 Aligned, {min_len} frames)"
                    )
                except Exception:
                    gt_in_slam = None

        # Set axes labels and styling
        ax.set_xlabel("X (m)", fontsize=11, labelpad=10)
        ax.set_ylabel("Y (m)", fontsize=11, labelpad=10)
        ax.set_zlabel("Z (m)", fontsize=11, labelpad=10)

        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle=':', alpha=0.6)

        # Equalize aspect ratio
        all_pts = []
        if len(points_3d) > 0:
            all_pts.append(points_3d)
        if len(est_centers) > 0:
            all_pts.append(est_centers)
        if gt_in_slam is not None and len(gt_in_slam) > 0:
            all_pts.append(gt_in_slam)

        if all_pts:
            combined = np.vstack(all_pts)
            combined = combined[np.isfinite(combined).all(axis=1)]

            if len(combined) > 0:
                mins = combined.min(axis=0)
                maxs = combined.max(axis=0)
                ranges = maxs - mins
                max_range = float(ranges.max()) / 2.0
                if not np.isfinite(max_range) or max_range < 0.1:
                    max_range = 1.0

                mids = (maxs + mins) * 0.5
                if np.all(np.isfinite(mids)) and np.isfinite(max_range):
                    ax.set_xlim(mids[0] - max_range, mids[0] + max_range)
                    ax.set_ylim(mids[1] - max_range, mids[1] + max_range)
                    ax.set_zlim(mids[2] - max_range, mids[2] + max_range)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"[Viewer3D] Saved 3D reconstruction plot to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)
