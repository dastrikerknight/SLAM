"""Visual vocabulary (k-means) and inverted index for BoW loop detection."""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import cv2


class VisualVocabulary:
    def __init__(self, num_words: int = 128):
        self.num_words = num_words
        self.centers: Optional[np.ndarray] = None
        self.idf: np.ndarray = np.ones(num_words, dtype=np.float32)

    def is_trained(self) -> bool:
        return self.centers is not None

    def train(self, descriptors_list: List[np.ndarray]):
        all_des = [d for d in descriptors_list if d is not None and len(d) > 0]
        if not all_des:
            return

        stacked = np.vstack(all_des).astype(np.float32)
        if len(stacked) > 4000:
            idx = np.random.choice(len(stacked), size=4000, replace=False)
            stacked = stacked[idx]

        k = min(self.num_words, len(stacked))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
        _, _, centers = cv2.kmeans(stacked, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        self.centers = centers
        self.num_words = k
        self.idf = np.ones(self.num_words, dtype=np.float32)

    def transform(self, des: np.ndarray) -> np.ndarray:
        if not self.is_trained() or des is None or len(des) == 0:
            return np.zeros(self.num_words, dtype=np.float32)

        des_f = des.astype(np.float32)
        # vectorized ||x - c||^2 without slow loops: x^2 + c^2 - 2xc
        x2 = np.sum(des_f ** 2, axis=1, keepdims=True)
        c2 = np.sum(self.centers ** 2, axis=1, keepdims=True).T
        dist = x2 + c2 - 2.0 * (des_f @ self.centers.T)
        words = np.argmin(dist, axis=1)

        counts = np.bincount(words, minlength=self.num_words).astype(np.float32)
        tfidf = counts * self.idf
        norm = np.sum(np.abs(tfidf))
        if norm > 1e-6:
            tfidf /= norm
        return tfidf


class InvertedIndex:
    def __init__(self, num_words: int = 128):
        self.num_words = num_words
        self.word_to_kfs: Dict[int, Set[int]] = {i: set() for i in range(num_words)}
        self.kf_vectors: Dict[int, np.ndarray] = {}

    def add_keyframe(self, kf_id: int, bow_vec: np.ndarray):
        self.kf_vectors[kf_id] = bow_vec
        nz = np.where(bow_vec > 1e-4)[0]
        for w in nz:
            self.word_to_kfs[w].add(kf_id)

    def remove_keyframe(self, kf_id: int):
        self.kf_vectors.pop(kf_id, None)
        for s in self.word_to_kfs.values():
            s.discard(kf_id)

    def query(
        self,
        query_vec: np.ndarray,
        min_score: float = 0.20,
        exclude_ids: Optional[Set[int]] = None,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        if len(self.kf_vectors) == 0 or np.sum(np.abs(query_vec)) < 1e-6:
            return []

        exclude = exclude_ids or set()
        nz_words = np.where(query_vec > 1e-4)[0]

        # inverted index lookup: only score keyframes that actually share visual words
        candidates: Set[int] = set()
        for w in nz_words:
            candidates.update(self.word_to_kfs[w])
        candidates.difference_update(exclude)

        if not candidates:
            return []

        # standard DBoW2 L1 similarity: 1 - 0.5 * sum(|a - b|)
        scores = []
        for cid in candidates:
            cand_vec = self.kf_vectors[cid]
            l1_diff = float(np.sum(np.abs(query_vec - cand_vec)))
            score = max(0.0, 1.0 - 0.5 * l1_diff)
            if score >= min_score:
                scores.append((cid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
