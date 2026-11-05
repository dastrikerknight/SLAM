"""Loop closing: BoW candidate retrieval, PnP verification, and Pose Graph Optimization."""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import cv2

from slam.frame import Frame
from slam.map import Map
from slam.map_point import MapPoint
from slam.camera import Camera
from slam.bow_vocabulary import VisualVocabulary, InvertedIndex
from slam.pose_graph import PoseGraphOptimizer


class LoopClosing:
    def __init__(
        self,
        camera: Camera,
        min_inliers: int = 35,
        min_time_diff: int = 80,
        min_bow_score: float = 0.40
    ):
        self.camera = camera
        self.min_inliers = min_inliers
        self.min_time_diff = min_time_diff
        self.min_bow_score = min_bow_score

        self.vocab = VisualVocabulary(num_words=128)
        self.index = InvertedIndex(num_words=128)
        self.pgo = PoseGraphOptimizer(max_iterations=20)
        self.loop_edges: List[Tuple[int, int, np.ndarray, float]] = []
        self.closed_pairs: Set[Tuple[int, int]] = set()
        self.last_loop_kf_id = -100

    def process_keyframe(self, curr_kf: Frame, slam_map: Map) -> Optional[Tuple[int, int]]:
        if curr_kf.des is None or len(curr_kf.des) < 20:
            return None

        # cooldown: don't thrash PGO on the same turn repeatedly
        if (curr_kf.id - self.last_loop_kf_id) < 40:
            return None

        # train vocabulary on the fly from initial keyframe descriptors
        if not self.vocab.is_trained():
            des_pool = [kf.des for kf in slam_map.keyframes.values() if kf.des is not None]
            if len(des_pool) >= 4:
                self.vocab.train(des_pool)
            else:
                return None

        bow_vec = self.vocab.transform(curr_kf.des)
        self.index.add_keyframe(curr_kf.id, bow_vec)

        # ignore recent keyframes so we only trigger on actual revisited areas
        exclude_ids = {kf.id for kf in slam_map.keyframes.values() if abs(curr_kf.id - kf.id) < self.min_time_diff}
        candidates = self.index.query(bow_vec, min_score=self.min_bow_score, exclude_ids=exclude_ids, top_k=3)
        if not candidates:
            return None

        for cand_id, score in candidates:
            if (curr_kf.id, cand_id) in self.closed_pairs or (cand_id, curr_kf.id) in self.closed_pairs:
                continue

            cand_kf = slam_map.keyframes.get(cand_id)
            if cand_kf is None or cand_kf.des is None:
                continue

            success, T_rel, inliers = self._verify_loop(curr_kf, cand_kf)
            if success:
                self.closed_pairs.add((curr_kf.id, cand_id))
                self.loop_edges.append((curr_kf.id, cand_id, T_rel, 3.0))
                self.last_loop_kf_id = curr_kf.id

                # distribute accumulated drift across all keyframes and map points
                self._apply_loop_closure(curr_kf, cand_kf, slam_map, inliers)
                return (curr_kf.id, cand_id)

        return None

    def _verify_loop(self, kf1: Frame, kf2: Frame) -> Tuple[bool, Optional[np.ndarray], List[Tuple[int, int]]]:
        # Lowe's ratio test: 0.7 filters out repetitive clutter
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(kf1.des, kf2.des, k=2)
        matches = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.70 * m[1].distance]

        if len(matches) < self.min_inliers:
            return False, None, []

        pts2_3d, pts1_2d, matched_pairs = [], [], []
        for m in matches:
            mp2 = kf2.map_points[m.trainIdx]
            if mp2 is not None and not mp2.is_bad:
                pts2_3d.append(mp2.point_3d)
                pts1_2d.append(kf1.kps_pts[m.queryIdx])
                mp1 = kf1.map_points[m.queryIdx]
                if mp1 is not None and not mp1.is_bad:
                    matched_pairs.append((mp1.id, mp2.id))

        if len(pts2_3d) < self.min_inliers:
            return False, None, []

        pts2_3d_arr = np.array(pts2_3d, dtype=np.float64)
        pts1_2d_arr = np.array(pts1_2d, dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts2_3d_arr,
            pts1_2d_arr,
            self.camera.K,
            None,
            iterationsCount=250,
            reprojectionError=2.5,
            confidence=0.99
        )
        if not success or inliers is None or len(inliers) < self.min_inliers:
            return False, None, []

        R_c1_w, _ = cv2.Rodrigues(rvec)
        T_c1_w = np.eye(4, dtype=np.float64)
        T_c1_w[:3, :3] = R_c1_w
        T_c1_w[:3, 3] = tvec.ravel()

        T_rel = kf2.pose @ np.linalg.inv(T_c1_w)
        inlier_indices = set(inliers.flatten())
        verified_pairs = [matched_pairs[i] for i in range(len(matched_pairs)) if i in inlier_indices]

        return True, T_rel, verified_pairs

    def _apply_loop_closure(self, curr_kf: Frame, cand_kf: Frame, slam_map: Map, point_matches: List[Tuple[int, int]]):
        old_poses = {kid: kf.pose.copy() for kid, kf in slam_map.keyframes.items()}
        new_poses = self.pgo.optimize(old_poses, self.loop_edges)

        for kid, T_new in new_poses.items():
            kf = slam_map.keyframes.get(kid)
            if kf is not None:
                kf.set_pose(T_new)

        # warp points into updated coordinates using ref keyframe delta
        for pid, mp in slam_map.map_points.items():
            if mp.is_bad or not mp.observations:
                continue
            ref_kid = next(iter(mp.observations.keys()))
            if ref_kid in old_poses and ref_kid in new_poses:
                T_old = old_poses[ref_kid]
                T_new = new_poses[ref_kid]
                p_c = T_old[:3, :3] @ mp.point_3d + T_old[:3, 3]
                p_w_new = np.linalg.inv(T_new[:3, :3]) @ (p_c - T_new[:3, 3])
                mp.set_position(p_w_new)

        # merge duplicate landmarks from the two viewpoints
        for id1, id2 in point_matches:
            mp1 = slam_map.map_points.get(id1)
            mp2 = slam_map.map_points.get(id2)
            if mp1 is not None and mp2 is not None and mp1 is not mp2:
                if np.linalg.norm(mp1.point_3d - mp2.point_3d) < 0.15:
                    for kid, feat_idx in list(mp1.observations.items()):
                        mp2.add_observation(kid, feat_idx)
                        kf = slam_map.keyframes.get(kid)
                        if kf and feat_idx < len(kf.map_points):
                            kf.map_points[feat_idx] = mp2
                    slam_map.remove_map_point(mp1.id)
