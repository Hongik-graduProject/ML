"""가중 하이브리드 벡터에 K-Means 클러스터링."""
import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans

from src.config import config

logger = logging.getLogger(__name__)


def cluster_themes(
    X: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, KMeans]:
    """K-Means 클러스터링을 수행한다.

    Args:
        X: 입력 벡터 (n_samples, n_features)
        k: 클러스터 개수

    Returns:
        labels: 각 샘플의 클러스터 레이블 (n_samples,)
        model: 학습된 KMeans 모델
    """
    model = KMeans(
        n_clusters=k,
        n_init=config.cluster.n_init,
        random_state=config.cluster.random_state,
    )
    labels: np.ndarray = model.fit_predict(X)
    logger.info("K-Means 클러스터링 완료: k=%d, 샘플수=%d", k, len(labels))
    return labels, model


def get_theme_summary(
    labels: np.ndarray,
    corp_names: List[str],
    years: List[int],
) -> Dict[int, List[str]]:
    """클러스터 레이블을 기반으로 테마별 기업-연도 목록을 반환한다.

    Args:
        labels: 클러스터 레이블 배열 (n_samples,)
        corp_names: 기업명 리스트 (n_samples,)  — labels와 동일 순서
        years: 연도 리스트 (n_samples,)          — labels와 동일 순서

    Returns:
        theme_map: {cluster_id: ["기업명(연도)", ...]} 딕셔너리
    """
    if len(labels) != len(corp_names) or len(labels) != len(years):
        raise ValueError(
            "labels, corp_names, years의 길이가 일치해야 합니다. "
            f"lengths: {len(labels)}, {len(corp_names)}, {len(years)}"
        )

    theme_map: Dict[int, List[str]] = {}
    for label, name, year in zip(labels, corp_names, years):
        cluster_id = int(label)
        theme_map.setdefault(cluster_id, [])
        theme_map[cluster_id].append(f"{name}({year})")

    for cluster_id in sorted(theme_map):
        logger.debug(
            "테마 %d: %d개 항목", cluster_id, len(theme_map[cluster_id])
        )

    return theme_map
