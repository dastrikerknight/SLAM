"""Inverse-disparity metric scale and shift alignment for neural depth."""

from typing import Tuple, Union, Sequence
import numpy as np


def _ransac_line_1d(
    x: np.ndarray,
    y: np.ndarray,
    max_iters: int = 50,
    thresh_rel: float = 0.20
) -> Tuple[float, float, np.ndarray]:
    # 1D line fit with RANSAC to avoid pulling in heavy scikit-learn
    n = len(x)
    if n < 2:
        return 1.0, 0.0, np.ones(n, dtype=bool)

    rng = np.random.default_rng(42)
    best_inliers = np.zeros(n, dtype=bool)
    best_a, best_b = 1.0, 0.0
    thresh = max(float(np.std(y) * thresh_rel), 0.05)

    for _ in range(max_iters):
        idx = rng.choice(n, size=2, replace=False)
        x0, x1 = x[idx[0]], x[idx[1]]
        y0, y1 = y[idx[0]], y[idx[1]]
        if abs(x1 - x0) < 1e-6:
            continue
        a = (y1 - y0) / (x1 - x0)
        b = y0 - a * x0
        inliers = np.abs(y - (a * x + b)) < thresh
        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_a, best_b = a, b

    # re-estimate with least squares on inliers
    if np.sum(best_inliers) >= 2:
        xi = x[best_inliers]
        yi = y[best_inliers]
        denom = np.var(xi)
        if denom > 1e-8:
            best_a = float(np.cov(xi, yi)[0, 1] / denom)
            best_b = float(np.mean(yi) - best_a * np.mean(xi))

    return float(best_a), float(best_b), best_inliers


def align_depth_scale_shift(
    pixel_coords: Union[np.ndarray, Sequence[Sequence[float]]],
    slam_depths: Union[np.ndarray, Sequence[float]],
    predicted_depth_map: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(pixel_coords, dtype=np.float32)
    depths = np.asarray(slam_depths, dtype=np.float32).ravel()
    pred_map = np.asarray(predicted_depth_map, dtype=np.float32)

    n_pts = len(coords)
    inlier_mask = np.zeros(n_pts, dtype=bool)
    if n_pts < 2 or len(depths) != n_pts:
        return pred_map.copy(), inlier_mask

    h, w = pred_map.shape[:2]
    u = np.round(coords[:, 0]).astype(np.int64)
    v = np.round(coords[:, 1]).astype(np.int64)

    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (depths > 0) & np.isfinite(depths)
    val_idx = np.where(valid)[0]
    if len(val_idx) < 2:
        return pred_map.copy(), inlier_mask

    pred_samples = pred_map[v[val_idx], u[val_idx]]
    good_samples = np.isfinite(pred_samples) & (pred_samples > 0)
    val_idx = val_idx[good_samples]
    pred_samples = pred_samples[good_samples]
    if len(val_idx) < 2:
        return pred_map.copy(), inlier_mask

    y_metric = depths[val_idx]
    x_pred = pred_samples

    # Depth Anything V2 outputs relative disparity (higher = closer, inversely proportional to depth)
    is_disparity = False
    if len(y_metric) >= 3:
        corr = np.corrcoef(x_pred, y_metric)[0, 1]
        if np.isfinite(corr) and corr < -0.1:
            is_disparity = True

    if is_disparity:
        # fit: 1 / z_slam = a * disp + b
        inv_y = 1.0 / np.clip(y_metric, 0.1, 20.0)
        a, b, sub_inliers = _ransac_line_1d(x_pred, inv_y)
        inlier_mask[val_idx[sub_inliers]] = True

        pred_disp = np.maximum(pred_map, 1e-4)
        aligned_inv_z = a * pred_disp + b
        aligned_inv_z = np.maximum(aligned_inv_z, 1.0 / 20.0)
        metric_depth = 1.0 / aligned_inv_z
    else:
        # direct linear fit: z_slam = a * pred + b
        a, b, sub_inliers = _ransac_line_1d(x_pred, y_metric)
        inlier_mask[val_idx[sub_inliers]] = True
        metric_depth = a * pred_map + b

    # sensible room depth bounds
    metric_depth = np.clip(metric_depth, 0.2, 8.0)
    return metric_depth.astype(np.float32), inlier_mask
