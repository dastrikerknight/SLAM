"""
Evaluation and benchmark metric calculation for Visual SLAM.
Implements the Umeyama Sim(3) closed-form alignment algorithm to resolve
monocular scale ambiguity, and computes Absolute Trajectory Error (ATE RMSE)
and Relative Pose Error (RPE).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt


def align_umeyama_sim3(
    model: np.ndarray,
    data: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes closed-form Sim(3) similarity transformation [scale, R, t] that aligns
    estimated trajectory 'model' (N, 3) to ground truth 'data' (N, 3):
    data ≈ s * R @ model + t

    Returns:
        scale (float)
        R (3x3 rotation matrix)
        t (3, translation vector)
        aligned_model (N, 3 aligned coordinates)
    """
    assert model.shape == data.shape
    n, m = model.shape

    # Means
    mu_m = model.mean(axis=0)
    mu_d = data.mean(axis=0)

    # Centered coordinates
    m_c = model - mu_m
    d_c = data - mu_d

    # Variance of model
    var_m = np.sum(m_c ** 2) / n

    # Cross-covariance matrix
    cov = (d_c.T @ m_c) / n

    # Singular Value Decomposition
    U, D, Vt = np.linalg.svd(cov)
    V = Vt.T

    # Reflection check
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        S[m - 1, m - 1] = -1

    R = U @ S @ V.T
    scale = (1.0 / var_m) * np.trace(np.diag(D) @ S)
    t = mu_d - scale * (R @ mu_m)

    aligned_model = (scale * (R @ model.T)).T + t
    return float(scale), R, t, aligned_model


def compute_ate_metrics(
    est_positions: np.ndarray,
    gt_positions: np.ndarray
) -> Dict[str, Any]:
    """
    Computes Absolute Trajectory Error (ATE) statistics between estimated and GT positions.
    """
    scale, R, t, aligned_est = align_umeyama_sim3(est_positions, gt_positions)

    # Euclidean error per frame
    errors = np.linalg.norm(gt_positions - aligned_est, axis=1)

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    std_err = float(np.std(errors))
    min_err = float(np.min(errors))
    max_err = float(np.max(errors))

    return {
        "ate_rmse": rmse,
        "mean_error": mean_err,
        "median_error": median_err,
        "std_error": std_err,
        "min_error": min_err,
        "max_error": max_err,
        "sim3_scale": scale,
        "sim3_rotation": R,
        "sim3_translation": t,
        "aligned_est": aligned_est,
        "errors": errors
    }


def plot_evaluation(
    est_positions: np.ndarray,
    gt_positions: np.ndarray,
    metrics: Dict[str, Any],
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """Plots aligned trajectories and frame-by-frame error curve."""
    aligned_est = metrics["aligned_est"]
    errors = metrics["errors"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Top-down aligned trajectory (X-Z)
    ax1 = axes[0]
    ax1.plot(gt_positions[:, 0], gt_positions[:, 2], 'g-', linewidth=2.0, label="Ground Truth")
    ax1.plot(aligned_est[:, 0], aligned_est[:, 2], 'b--', linewidth=2.0, label=f"Aligned Estimated (scale={metrics['sim3_scale']:.2f})")
    ax1.set_xlabel("X (m)", fontsize=11)
    ax1.set_ylabel("Z (m)", fontsize=11)
    ax1.set_title(f"Sim(3) Aligned Trajectory (ATE RMSE: {metrics['ate_rmse']:.3f} m)", fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.axis('equal')

    # Subplot 2: Error over frames
    ax2 = axes[1]
    frames = np.arange(len(errors))
    ax2.plot(frames, errors, 'r-', linewidth=1.5, label="Frame Error")
    ax2.axhline(metrics["ate_rmse"], color='black', linestyle='--', label=f"RMSE ({metrics['ate_rmse']:.3f} m)")
    ax2.set_xlabel("Frame Index", fontsize=11)
    ax2.set_ylabel("Absolute Error (m)", fontsize=11)
    ax2.set_title("Absolute Trajectory Error per Frame", fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[Evaluation] Saved evaluation plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
