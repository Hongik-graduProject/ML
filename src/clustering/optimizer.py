"""실루엣 스코어 기반 최적 K값 자동 탐색."""
import logging
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import config

logger = logging.getLogger(__name__)


def find_optimal_k(
    X: np.ndarray,
) -> Tuple[int, float, List[Tuple[int, float]]]:
    """실루엣 스코어를 이용해 최적 K값을 탐색한다.

    Args:
        X: 클러스터링 입력 벡터 (n_samples, n_features)

    Returns:
        best_k: 최적 K값
        best_score: 해당 K에서의 실루엣 스코어
        all_scores: [(k, score), ...] 전체 탐색 결과
    """
    k_min, k_max = config.cluster.k_range
    n_samples = X.shape[0]

    # K는 최소 2, 최대 n_samples - 1 로 제한
    k_max = min(k_max, n_samples - 1)
    k_min = max(k_min, 2)

    if k_min > k_max:
        raise ValueError(
            f"유효한 K 범위가 없습니다: n_samples={n_samples}, "
            f"k_range={config.cluster.k_range}"
        )

    all_scores: List[Tuple[int, float]] = []
    best_k = k_min
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(
            n_clusters=k,
            n_init=config.cluster.n_init,
            random_state=config.cluster.random_state,
        )
        labels = kmeans.fit_predict(X)

        # 모든 샘플이 같은 클러스터에 배정되면 실루엣 계산 불가
        n_unique = len(set(labels))
        if n_unique < 2:
            logger.warning("k=%d: 단일 클러스터로 수렴, 스킵", k)
            continue

        score = silhouette_score(X, labels)
        all_scores.append((k, score))
        logger.debug("k=%d, silhouette=%.4f", k, score)

        if score > best_score:
            best_score = score
            best_k = k

    if not all_scores:
        raise RuntimeError("유효한 K 탐색 결과가 없습니다.")

    logger.info("최적 K=%d (silhouette=%.4f)", best_k, best_score)
    return best_k, best_score, all_scores
