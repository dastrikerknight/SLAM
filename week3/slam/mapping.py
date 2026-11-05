"""Local mapping: triangulate points between keyframes, fuse covisible points, run local BA."""

from typing import List, Tuple
import numpy as np
import cv2
from slam.camera import Camera
from slam.frame import Frame
from slam.map import Map
from slam.map_point import MapPoint
from slam.feature_tracker import FeatureTracker
from slam.optimizer import LocalBundleAdjustment
from slam.loop_closing import LoopClosing


class LocalMapping:
    def __init__(
        self,
        camera: Camera,
        feature_tracker: FeatureTracker,
        min_parallax_deg: float = 1.2,
        max_reprojection_err: float = 2.5
    ):
        self.camera = camera
        self.feature_tracker = feature_tracker
        self.min_parallax_deg = min_parallax_deg
        self.max_reprojection_err = max_reprojection_err
        self.optimizer = LocalBundleAdjustment(window_size=5, max_iterations=10, max_points=120)
        self.loop_closing = LoopClosing(camera=self.camera)
        self.kf_counter = 0

    def process_new_keyframe(self, current_kf: Frame, slam_map: Map) -> int:
        self.kf_counter += 1
        all_kfs = list(slam_map.keyframes.values())
        if len(all_kfs) < 2:
            return 0

        # in monocular mode, triangulate against last 1-2 keyframes
        new_points_count = 0
        if current_kf.depth is None:
            new_points_count += self.triangulate_with_keyframe(all_kfs[-2], current_kf, slam_map)
            if len(all_kfs) >= 3:
                new_points_count += self.triangulate_with_keyframe(all_kfs[-3], current_kf, slam_map)

        # fuse covisible points if trajectory loops back near an old keyframe
        if len(all_kfs) > 35:
            curr_pos = current_kf.get_camera_center()
            curr_rot = current_kf.get_rotation()
            loop_kfs = [kf for kf in all_kfs if (current_kf.id - kf.id) >= 30]
            if loop_kfs:
                centers = np.array([kf.get_camera_center() for kf in loop_kfs])
                dists = np.linalg.norm(centers - curr_pos, axis=1)
                close_indices = np.where(dists < 0.50)[0]
                for idx in close_indices:
                    old_kf = loop_kfs[idx]
                    rot_rel = curr_rot @ old_kf.get_rotation().T
                    angle = np.degrees(np.arccos(np.clip((np.trace(rot_rel) - 1.0) / 2.0, -1.0, 1.0)))
                    if angle < 40.0:
                        self.fuse_covisible_keyframes(old_kf, current_kf, slam_map)
                        break

        # drop points that failed tracking support (young points get a 3-frame grace period)
        slam_map.cull_map_points(min_observations=2, current_kf_id=current_kf.id)

        # BoW loop closing + pose graph optimization
        loop_res = self.loop_closing.process_keyframe(current_kf, slam_map)
        if loop_res is not None:
            kid1, kid2 = loop_res
            print(f"[Loop] Closed loop between KeyFrame {kid1} and {kid2}! Pose graph optimized.")

        # run local BA every 3rd keyframe to avoid stalling tracking
        if current_kf.depth is None and len(slam_map.keyframes) >= 3 and (self.kf_counter % 3 == 0):
            self.optimizer.optimize(slam_map, self.camera)

        return new_points_count

    def fuse_covisible_keyframes(
        self,
        old_kf: Frame,
        curr_kf: Frame,
        slam_map: Map
    ) -> int:
        if old_kf.des is None or curr_kf.des is None or len(old_kf.des) < 20 or len(curr_kf.des) < 20:
            return 0

        matches = self.feature_tracker.match_features(old_kf.des, curr_kf.des, ratio_thresh=0.80)
        if len(matches) < 15:
            return 0

        R1, t1 = old_kf.get_rotation(), old_kf.get_translation()
        R2, t2 = curr_kf.get_rotation(), curr_kf.get_translation()
        R12 = R2 @ R1.T
        t12 = t2.flatten() - R12 @ t1.flatten()
        tx = np.array([
            [0.0, -t12[2], t12[1]],
            [t12[2], 0.0, -t12[0]],
            [-t12[1], t12[0], 0.0]
        ], dtype=np.float64)
        K_inv = np.linalg.inv(self.camera.K)
        F = K_inv.T @ tx @ R12 @ K_inv

        fused = 0
        for m_old, m_curr, _ in matches:
            p1 = old_kf.kps_pts[m_old]
            p2 = curr_kf.kps_pts[m_curr]

            p1_h = np.array([p1[0], p1[1], 1.0], dtype=np.float64)
            line2 = F @ p1_h
            denom = np.hypot(line2[0], line2[1])
            if denom < 1e-6:
                continue
            dist_epipolar = abs(line2[0] * p2[0] + line2[1] * p2[1] + line2[2]) / denom
            if dist_epipolar > 2.5:
                continue

            mp_old = old_kf.map_points[m_old]
            mp_curr = curr_kf.map_points[m_curr]

            if mp_old is not None and not mp_old.is_bad:
                if mp_curr is None:
                    curr_kf.associate_map_point(m_curr, mp_old)
                    mp_old.add_observation(curr_kf.id, m_curr)
                    fused += 1
                elif mp_curr.id != mp_old.id and not mp_curr.is_bad:
                    for obs_kid, obs_idx in list(mp_curr.observations.items()):
                        if obs_kid in slam_map.keyframes:
                            slam_map.keyframes[obs_kid].associate_map_point(obs_idx, mp_old)
                            mp_old.add_observation(obs_kid, obs_idx)
                    mp_curr.is_bad = True
                    slam_map.remove_map_point(mp_curr.id)
                    fused += 1

        return fused

    def triangulate_with_keyframe(
        self,
        kf1: Frame,
        kf2: Frame,
        slam_map: Map
    ) -> int:
        unmatched_idx1 = [i for i, mp in enumerate(kf1.map_points) if mp is None or mp.is_bad]
        unmatched_idx2 = [i for i, mp in enumerate(kf2.map_points) if mp is None or mp.is_bad]

        if len(unmatched_idx1) < 10 or len(unmatched_idx2) < 10:
            return 0

        R1, t1 = kf1.get_rotation(), kf1.get_translation()
        R2, t2 = kf2.get_rotation(), kf2.get_translation()

        R12 = R2 @ R1.T
        t12 = t2.flatten() - R12 @ t1.flatten()
        baseline = float(np.linalg.norm(t12))

        matched_pairs = []

        # search along epipolar lines if baseline is big enough (>1cm) to constrain search
        if baseline >= 0.01:
            tx = np.array([
                [0.0, -t12[2], t12[1]],
                [t12[2], 0.0, -t12[0]],
                [-t12[1], t12[0], 0.0]
            ], dtype=np.float64)
            E = tx @ R12
            K_inv = np.linalg.inv(self.camera.K)
            F = K_inv.T @ E @ K_inv

            pts1_unm = kf1.kps_pts[unmatched_idx1]
            pts2_unm = kf2.kps_pts[unmatched_idx2]
            des1_unm = kf1.des[unmatched_idx1]
            des2_unm = kf2.des[unmatched_idx2]

            P1_h = np.hstack((pts1_unm, np.ones((len(pts1_unm), 1), dtype=np.float64)))
            P2_h = np.hstack((pts2_unm, np.ones((len(pts2_unm), 1), dtype=np.float64)))

            L2 = (F @ P1_h.T).T
            denom = np.sqrt(L2[:, 0]**2 + L2[:, 1]**2 + 1e-12)[:, None]
            L2_norm = L2 / denom

            dist_mat = np.abs(L2_norm @ P2_h.T)

            matched_i2 = set()
            max_epipolar_dist = 2.5
            max_hamming = 50

            for i1 in range(len(unmatched_idx1)):
                cand_indices = np.where(dist_mat[i1] <= max_epipolar_dist)[0]
                if len(cand_indices) == 0:
                    continue

                d1 = des1_unm[i1]
                best_cand = None
                best_dist = max_hamming + 1
                second_best_dist = 256

                for c in cand_indices:
                    if c in matched_i2:
                        continue
                    d2 = des2_unm[c]
                    h_dist = cv2.norm(d1, d2, cv2.NORM_HAMMING)
                    if h_dist < best_dist:
                        second_best_dist = best_dist
                        best_dist = h_dist
                        best_cand = c
                    elif h_dist < second_best_dist:
                        second_best_dist = h_dist

                if best_cand is not None and best_dist <= max_hamming:
                    if best_dist < 0.88 * second_best_dist or best_dist <= 30:
                        matched_pairs.append((unmatched_idx1[i1], unmatched_idx2[best_cand]))
                        matched_i2.add(best_cand)

        # fallback to standard ratio test if epipolar geometry gave too few matches
        if len(matched_pairs) < 10:
            des1 = kf1.des[unmatched_idx1]
            des2 = kf2.des[unmatched_idx2]
            raw_matches = self.feature_tracker.match_features(des1, des2, ratio_thresh=0.80)
            if raw_matches:
                existing_m1 = {p[0] for p in matched_pairs}
                existing_m2 = {p[1] for p in matched_pairs}
                for m0, m1, _ in raw_matches:
                    u1 = unmatched_idx1[m0]
                    u2 = unmatched_idx2[m1]
                    if u1 not in existing_m1 and u2 not in existing_m2:
                        matched_pairs.append((u1, u2))
                        existing_m1.add(u1)
                        existing_m2.add(u2)

        if not matched_pairs:
            return 0

        m_idx1 = [p[0] for p in matched_pairs]
        m_idx2 = [p[1] for p in matched_pairs]

        pts1 = kf1.kps_pts[m_idx1]
        pts2 = kf2.kps_pts[m_idx2]

        P1 = self.camera.K @ np.hstack((R1, t1))
        P2 = self.camera.K @ np.hstack((R2, t2))

        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = (pts4d[:3] / (pts4d[3] + 1e-12)).T

        c1 = kf1.get_camera_center()
        c2 = kf2.get_camera_center()

        new_count = 0
        for i, pt_w in enumerate(pts3d):
            pt_c1 = R1 @ pt_w + t1.flatten()
            pt_c2 = R2 @ pt_w + t2.flatten()

            if pt_c1[2] < 0.2 or pt_c1[2] > 5.0 or pt_c2[2] < 0.2 or pt_c2[2] > 5.0:
                continue
            if not np.all(np.isfinite(pt_w)):
                continue

            ray1 = pt_w - c1
            ray2 = pt_w - c2
            norm1 = np.linalg.norm(ray1)
            norm2 = np.linalg.norm(ray2)
            if norm1 < 1e-4 or norm2 < 1e-4:
                continue

            cos_parallax = np.dot(ray1, ray2) / (norm1 * norm2)
            cos_parallax = np.clip(cos_parallax, -1.0, 1.0)
            parallax_deg = np.degrees(np.arccos(cos_parallax))

            if parallax_deg < self.min_parallax_deg:
                continue

            uv1, ok1 = self.camera.project(pt_c1.reshape(1, 3))
            uv2, ok2 = self.camera.project(pt_c2.reshape(1, 3))

            if not ok1 or not ok2:
                continue

            err1 = np.linalg.norm(uv1[0] - pts1[i])
            err2 = np.linalg.norm(uv2[0] - pts2[i])

            if err1 > self.max_reprojection_err or err2 > self.max_reprojection_err:
                continue

            orig_idx1 = m_idx1[i]
            orig_idx2 = m_idx2[i]

            desc1 = kf1.des[orig_idx1]
            desc2 = kf2.des[orig_idx2]

            u, v = int(round(pts2[i][0])), int(round(pts2[i][1]))
            if 0 <= v < kf2.image.shape[0] and 0 <= u < kf2.image.shape[1]:
                pixel_val = kf2.image[v, u]
                if len(kf2.image.shape) == 3:
                    color = np.array(pixel_val[::-1], dtype=np.uint8)
                else:
                    color = np.array([pixel_val, pixel_val, pixel_val], dtype=np.uint8)
            else:
                color = np.array([255, 255, 0], dtype=np.uint8)

            map_point = MapPoint(pt_w, first_frame_id=kf1.id, color=color, descriptor=desc2)
            map_point.update_descriptor([desc1, desc2])

            slam_map.add_map_point(map_point)
            kf1.associate_map_point(orig_idx1, map_point)
            kf2.associate_map_point(orig_idx2, map_point)

            new_count += 1

        return new_count
