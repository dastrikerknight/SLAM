"""
Feature detection, extraction, and tracking using Pyramidal KLT Optical Flow and ORB.
Provides bidirectional optical flow for high-speed continuous tracking
and uniform ORB extraction for KeyFrames and relocalization.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2


class FeatureTracker:
    def __init__(
        self,
        num_features: int = 1500,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        fast_threshold: int = 20,
        grid_rows: int = 4,
        grid_cols: int = 6
    ):
        self.num_features = num_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        self.fast_threshold = fast_threshold
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        # Primary ORB detector & descriptor extractor
        self.orb = cv2.ORB_create(
            nfeatures=self.num_features,
            scaleFactor=self.scale_factor,
            nlevels=self.n_levels,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=self.fast_threshold
        )

        # Fast feature detector for replenishment
        self.fast = cv2.FastFeatureDetector_create(threshold=self.fast_threshold, nonmaxSuppression=True)

        # Hamming distance brute force matcher for binary ORB descriptors
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detects features combining Shi-Tomasi corners (goodFeaturesToTrack with subpixel refinement)
        and multi-scale ORB features, followed by uniform spatial distribution.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 1. Extract Shi-Tomasi high-gradient corners with subpixel refinement
        gftt_pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.num_features // 2, qualityLevel=0.01, minDistance=10)
        gftt_kps = []
        if gftt_pts is not None and len(gftt_pts) > 0:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.02)
            try:
                gftt_subpix = cv2.cornerSubPix(gray, gftt_pts, (3, 3), (-1, -1), criteria)
                gftt_kps = [cv2.KeyPoint(x=float(p[0, 0]), y=float(p[0, 1]), size=31.0) for p in gftt_subpix]
            except Exception:
                gftt_kps = [cv2.KeyPoint(x=float(p[0, 0]), y=float(p[0, 1]), size=31.0) for p in gftt_pts]

        # 2. Extract multi-scale ORB features
        orb_kps = self.orb.detect(gray, None)
        orb_kps = list(orb_kps) if orb_kps is not None else []

        combined_kps = gftt_kps + orb_kps
        if not combined_kps:
            return [], np.empty((0, 32), dtype=np.uint8)

        # Compute descriptors
        kps, des = self.orb.compute(gray, combined_kps)
        if des is None or len(kps) == 0:
            return [], np.empty((0, 32), dtype=np.uint8)

        kps, des = self._distribute_features_grid(gray.shape[:2], list(kps), des)
        return kps, des

    def detect_new_features(
        self,
        gray: np.ndarray,
        existing_pts: np.ndarray,
        min_distance: int = 12,
        max_new: int = 1000
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detects fresh features in areas of the image not already covered by existing points.
        Combines Shi-Tomasi corners and ORB features.
        """
        h, w = gray.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)

        # Mask out radius around existing points
        if len(existing_pts) > 0:
            for pt in existing_pts:
                x, y = int(round(pt[0])), int(round(pt[1]))
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(mask, (x, y), min_distance, 0, -1)

        # Detect Shi-Tomasi corners in unmasked region
        gftt_pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_new // 2, qualityLevel=0.01, minDistance=min_distance, mask=mask)
        gftt_kps = []
        if gftt_pts is not None and len(gftt_pts) > 0:
            gftt_kps = [cv2.KeyPoint(x=float(p[0, 0]), y=float(p[0, 1]), size=31.0) for p in gftt_pts]

        orb_kps = list(self.orb.detect(gray, mask=mask))
        combined_kps = gftt_kps + orb_kps

        if not combined_kps:
            return [], np.empty((0, 32), dtype=np.uint8)

        combined_kps.sort(key=lambda k: k.response, reverse=True)
        combined_kps = combined_kps[:max_new]

        new_kps, new_des = self.orb.compute(gray, combined_kps)
        if new_des is None or len(new_kps) == 0:
            return [], np.empty((0, 32), dtype=np.uint8)

        return list(new_kps), new_des

    def track_klt(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        pts_prev: np.ndarray,
        max_fb_err: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tracks 2D points from prev_gray to curr_gray using Pyramidal Lucas-Kanade Optical Flow.
        Applies forward-backward consistency check to reject outliers and motion blur.
        Returns:
            pts_curr (N, 2): tracked coordinates
            valid_mask (N,): boolean mask of successfully tracked points
        """
        if len(pts_prev) == 0:
            return np.empty((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)

        pts_p = np.asarray(pts_prev, dtype=np.float32)

        # Forward flow
        pts_c, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts_p, None,
            winSize=(25, 25), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Backward flow for forward-backward consistency check
        pts_p_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, pts_c, None,
            winSize=(25, 25), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        h, w = curr_gray.shape[:2]
        fb_dist = np.linalg.norm(pts_p - pts_p_back, axis=1)

        # In-bounds check
        in_bounds = (
            (pts_c[:, 0] >= 5) & (pts_c[:, 0] < w - 5) &
            (pts_c[:, 1] >= 5) & (pts_c[:, 1] < h - 5)
        )

        valid = (status_fwd.flatten() == 1) & (status_bwd.flatten() == 1) & (fb_dist <= max_fb_err) & in_bounds
        return pts_c, valid

    def _distribute_features_grid(
        self,
        image_shape: Tuple[int, int],
        kps: List[cv2.KeyPoint],
        des: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Distributes keypoints uniformly across image grid cells."""
        h, w = image_shape
        cell_h = h / self.grid_rows
        cell_w = w / self.grid_cols
        max_per_cell = int(np.ceil(self.num_features / (self.grid_rows * self.grid_cols)) * 1.5)

        cells = [[[] for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]

        for idx, kp in enumerate(kps):
            r = min(int(kp.pt[1] / cell_h), self.grid_rows - 1)
            c = min(int(kp.pt[0] / cell_w), self.grid_cols - 1)
            cells[r][c].append(idx)

        selected_indices = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                indices = cells[r][c]
                if len(indices) <= max_per_cell:
                    selected_indices.extend(indices)
                else:
                    sorted_indices = sorted(indices, key=lambda i: kps[i].response, reverse=True)
                    selected_indices.extend(sorted_indices[:max_per_cell])

        if len(selected_indices) == len(kps):
            return kps, des

        selected_kps = [kps[i] for i in selected_indices]
        selected_des = des[selected_indices]
        return selected_kps, selected_des

    def match_features(
        self,
        des1: np.ndarray,
        des2: np.ndarray,
        ratio_thresh: float = 0.80,
        max_hamming_dist: int = 65
    ) -> List[Tuple[int, int, float]]:
        """Matches descriptors using KNN (k=2) and Lowe's ratio test."""
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []

        knn_matches = self.matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance and m.distance <= max_hamming_dist:
                    good_matches.append((m.queryIdx, m.trainIdx, float(m.distance)))

        return good_matches
