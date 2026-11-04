from typing import Tuple, Union, Sequence
import numpy as np
from sklearn.linear_model import RANSACRegressor

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

    # Extract pixel coordinates (u: col, v: row)
    u = np.round(coords[:, 0]).astype(np.int64)
    v = np.round(coords[:, 1]).astype(np.int64)

    # Filter points within image boundaries and with positive SLAM depth
    valid_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (depths > 0) & np.isfinite(depths)
    valid_indices = np.where(valid_bounds)[0]

    if len(valid_indices) < 2:
        return pred_map.copy(), inlier_mask

    # Sample predicted depth at valid pixel locations
    pred_samples = pred_map[v[valid_indices], u[valid_indices]]
    valid_samples = np.isfinite(pred_samples) & (pred_samples > 0)
    valid_indices = valid_indices[valid_samples]
    pred_samples = pred_samples[valid_samples]

    if len(valid_indices) < 2:
        return pred_map.copy(), inlier_mask

    x_pred = pred_samples.reshape(-1, 1)
    y_metric = depths[valid_indices]

    # Check if model outputs relative disparity (larger = closer, negative correlation with depth)
    is_disparity = False
    if len(y_metric) >= 3:
        corr_mat = np.corrcoef(x_pred.ravel(), y_metric.ravel())
        if corr_mat.ndim == 2 and corr_mat.shape[0] >= 2 and corr_mat.shape[1] >= 2:
            corr = corr_mat[0, 1]
            if np.isfinite(corr) and corr < -0.1:
                is_disparity = True

    ransac = RANSACRegressor(
        min_samples=2,
        residual_threshold=None,
        random_state=42
    )

    try:
        if is_disparity:
            # Fit inverse depth (disparity space): 1/z = scale * pred_disp + shift
            y_target = (1.0 / np.maximum(y_metric, 1e-3)).reshape(-1, 1)
            ransac.fit(x_pred, y_target.ravel())
            scale = float(ransac.estimator_.coef_[0])
            shift = float(ransac.estimator_.intercept_)
            if hasattr(ransac, "inlier_mask_") and ransac.inlier_mask_ is not None:
                inlier_mask[valid_indices[ransac.inlier_mask_]] = True
            denom = np.maximum(scale * pred_map + shift, 1e-3)
            metric_depth_map = (1.0 / denom).astype(np.float32)
        else:
            # Direct metric linear fit: metric_z = scale * pred_z + shift
            ransac.fit(x_pred, y_metric)
            scale = float(ransac.estimator_.coef_[0])
            shift = float(ransac.estimator_.intercept_)
            if hasattr(ransac, "inlier_mask_") and ransac.inlier_mask_ is not None:
                inlier_mask[valid_indices[ransac.inlier_mask_]] = True
            metric_depth_map = np.maximum(scale * pred_map + shift, 1e-3).astype(np.float32)
    except Exception:
        # Robust fallback: median scaling
        scale = float(np.median(y_metric) / np.maximum(np.median(pred_samples), 1e-3))
        metric_depth_map = np.maximum(scale * pred_map, 1e-3).astype(np.float32)

    return metric_depth_map, inlier_mask

if __name__ == "__main__":
    # Self-test verification
    np.random.seed(42)
    h, w = 480, 640
    # Create synthetic smooth depth map
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    synth_pred = (2.0 + 1.5 * np.sin(x_grid / 100.0) + 0.5 * (y_grid / h)).astype(np.float32)

    # Ground truth affine relation
    true_scale = 1.75
    true_shift = 0.50

    # Sample 40 sparse landmark points (u, v)
    u_pts = np.random.randint(50, w - 50, size=40)
    v_pts = np.random.randint(50, h - 50, size=40)
    coords = np.column_stack([u_pts, v_pts])

    pred_z = synth_pred[v_pts, u_pts]
    metric_z = true_scale * pred_z + true_shift

    # Add 10% outliers
    metric_z[:4] += 10.0

    aligned_map, mask = align_depth_scale_shift(coords, metric_z, synth_pred)

    assert aligned_map.shape == (h, w)
    assert aligned_map.dtype == np.float32
    assert mask.shape == (40,)
    assert mask.sum() >= 30, f"Expected >= 30 inliers, got {mask.sum()}"
    print(f"Self-test passed: {mask.sum()}/40 inliers identified, aligned map shape {aligned_map.shape} ({aligned_map.dtype})")
