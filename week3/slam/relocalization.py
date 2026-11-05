"""Kidnapped robot relocalization via visual vocabulary and PnP-RANSAC."""

from typing import Optional, Tuple
import numpy as np
import cv2

from slam.frame import Frame
from slam.map import Map
from slam.camera import Camera
from slam.bow_vocabulary import VisualVocabulary, InvertedIndex
from slam.feature_tracker import FeatureTracker


class Relocalizer:
    def __init__(
        self,
        camera: Camera,
        feature_tracker: FeatureTracker,
        min_inliers_pnp: int = 12
    ):
        self.camera = camera
        self.tracker = feature_tracker
        self.min_inliers_pnp = min_inliers_pnp

    def relocalize(
        self,
        curr_frame: Frame,
        slam_map: Map,
        vocab: VisualVocabulary,
        index: InvertedIndex
    ) -> bool:
        if not slam_map.keyframes:
            return False

        if curr_frame.num_features < 50:
            kps, des = self.tracker.detect_and_compute(curr_frame.image)
            curr_frame.set_features(kps, des)

        if curr_frame.des is None or len(curr_frame.des) < 20:
            return False

        # try fast BoW query first if vocabulary is ready
        candidates = []
        if vocab.is_trained():
            bow_vec = vocab.transform(curr_frame.des)
            candidates = [cid for cid, _ in index.query(bow_vec, min_score=0.15, top_k=4)]

        # fallback: try keyframes holding the most 3D points if BoW has low coverage
        if len(candidates) < 2:
            sorted_kfs = sorted(slam_map.keyframes.values(), key=lambda k: k.count_tracked_points(), reverse=True)
            for kf in sorted_kfs[:4]:
                if kf.id not in candidates:
                    candidates.append(kf.id)

        for kid in candidates:
            cand_kf = slam_map.keyframes.get(kid)
            if cand_kf is None or cand_kf.des is None:
                continue

            matches = self.tracker.match_features(cand_kf.des, curr_frame.des, ratio_thresh=0.80)
            if len(matches) < self.min_inliers_pnp:
                continue

            pts_3d, pts_2d, matched_mps = [], [], []
            for q_idx, t_idx, _ in matches:
                mp = cand_kf.map_points[q_idx]
                if mp is not None and not mp.is_bad:
                    pts_3d.append(mp.point_3d)
                    pts_2d.append(curr_frame.kps_pts[t_idx])
                    matched_mps.append((t_idx, mp))

            if len(pts_3d) < self.min_inliers_pnp:
                continue

            pts_3d_arr = np.array(pts_3d, dtype=np.float64)
            pts_2d_arr = np.array(pts_2d, dtype=np.float64)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d_arr, pts_2d_arr,
                self.camera.K, self.camera.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=3.5,
                confidence=0.99,
                iterationsCount=150
            )

            if success and inliers is not None and len(inliers) >= self.min_inliers_pnp:
                rvec, tvec = cv2.solvePnPRefineLM(
                    pts_3d_arr[inliers.ravel()],
                    pts_2d_arr[inliers.ravel()],
                    self.camera.K, self.camera.dist_coeffs,
                    rvec, tvec
                )

                R, _ = cv2.Rodrigues(rvec)
                curr_frame.set_R_t(R, tvec.ravel())

                for idx in inliers.ravel():
                    feat_idx, mp = matched_mps[idx]
                    curr_frame.map_points[feat_idx] = mp

                return True

        return False
