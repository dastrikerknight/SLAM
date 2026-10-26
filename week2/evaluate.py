from typing import Tuple, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

def align_umeyama_sim3(model: np.ndarray, data: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Closed-form Sim(3) alignment: data ~= s * R @ model + t.
    Solves monocular scale ambiguity against ground truth.
    """
    assert model.shape == data.shape
    n, m = model.shape

    mu_m = model.mean(axis=0)
    mu_d = data.mean(axis=0)

    m_c = model - mu_m
    d_c = data - mu_d

    var_m = np.sum(m_c ** 2) / n
    cov = (d_c.T @ m_c) / n

    U, D, Vt = np.linalg.svd(cov)
    V = Vt.T

    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        S[m - 1, m - 1] = -1

    R = U @ S @ V.T
    scale = float((1.0 / var_m) * np.trace(np.diag(D) @ S))
    t = mu_d - scale * (R @ mu_m)

    aligned = (scale * (R @ model.T)).T + t
    return scale, R, t, aligned

def compute_ate_metrics(est_positions: np.ndarray, gt_positions: np.ndarray) -> Dict[str, Any]:
    scale, R, t, aligned = align_umeyama_sim3(est_positions, gt_positions)
    errors = np.linalg.norm(gt_positions - aligned, axis=1)

    return {
        "ate_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "std_error": float(np.std(errors)),
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
        "sim3_scale": scale,
        "sim3_rotation": R,
        "sim3_translation": t,
        "aligned_est": aligned,
        "errors": errors
    }

def plot_evaluation(est_positions: np.ndarray, gt_positions: np.ndarray, metrics: Dict[str, Any], save_path: Optional[str] = None, show: bool = False):
    aligned = metrics["aligned_est"]
    errors = metrics["errors"]

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], "k--", label="Ground Truth", linewidth=2)
    ax1.plot(aligned[:, 0], aligned[:, 1], aligned[:, 2], "b-", label=f"Aligned Est (RMSE={metrics['ate_rmse']:.3f}m)", linewidth=1.5)
    ax1.set_title("Sim(3) Aligned Trajectory")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(errors, "r-", label="Per-frame ATE")
    ax2.axhline(metrics["ate_rmse"], color="blue", linestyle="--", label=f"RMSE: {metrics['ate_rmse']:.3f}m")
    ax2.axhline(metrics["mean_error"], color="green", linestyle=":", label=f"Mean: {metrics['mean_error']:.3f}m")
    ax2.set_title("Absolute Trajectory Error over Time")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Error (m)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()
