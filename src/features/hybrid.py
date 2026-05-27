"""하이브리드 벡터 결합 모듈: 텍스트(768D) + 재무(10D) + 매크로(3D) → 781D."""

import logging
from typing import Optional, Tuple, Dict

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import config

logger = logging.getLogger(__name__)


def build_hybrid_vector(
    text_vec: np.ndarray,
    financial_vec: np.ndarray,
    macro_vec: np.ndarray,
    scalers: Optional[Dict[str, StandardScaler]] = None,
) -> np.ndarray:
    """단일 샘플 하이브리드 벡터 생성.

    각 모달리티 벡터를 선택적으로 정규화한 뒤 가중치를 곱해 이어붙인다.

    Args:
        text_vec: 텍스트 벡터, shape (768,).
        financial_vec: 재무 벡터, shape (10,).
        macro_vec: 매크로 Beta 벡터, shape (3,).
        scalers: 각 모달리티의 StandardScaler dict.
            키: "text", "financial", "macro".
            None이면 정규화 없이 가중치만 적용.

    Returns:
        np.ndarray: shape (781,) float32 가중 결합 벡터.
    """
    w_text = config.cluster.weight_text
    w_fin = config.cluster.weight_financial
    w_macro = config.cluster.weight_macro

    t = text_vec.astype(np.float32).reshape(-1)
    f = financial_vec.astype(np.float32).reshape(-1)
    m = macro_vec.astype(np.float32).reshape(-1)

    if scalers is not None:
        if "text" in scalers:
            t = scalers["text"].transform(t.reshape(1, -1)).reshape(-1).astype(np.float32)
        if "financial" in scalers:
            f = scalers["financial"].transform(f.reshape(1, -1)).reshape(-1).astype(np.float32)
        if "macro" in scalers:
            m = scalers["macro"].transform(m.reshape(1, -1)).reshape(-1).astype(np.float32)

    hybrid = np.concatenate([w_text * t, w_fin * f, w_macro * m])
    return hybrid.astype(np.float32)


def build_hybrid_matrix(
    text_vecs: np.ndarray,
    financial_vecs: np.ndarray,
    macro_vecs: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, StandardScaler]]:
    """다수 샘플의 하이브리드 벡터 행렬 생성.

    각 모달리티를 독립적으로 StandardScaler로 정규화한 뒤 가중치를 곱해 이어붙인다.

    Args:
        text_vecs: shape (N, 768).
        financial_vecs: shape (N, 10).
        macro_vecs: shape (N, 3).

    Returns:
        Tuple:
            - np.ndarray: shape (N, 781) float32 하이브리드 행렬.
            - Dict[str, StandardScaler]: 각 모달리티의 fitted scaler dict.
                키: "text", "financial", "macro".
    """
    w_text = config.cluster.weight_text
    w_fin = config.cluster.weight_financial
    w_macro = config.cluster.weight_macro

    n = text_vecs.shape[0]
    logger.info("Building hybrid matrix for %d samples.", n)

    t = text_vecs.astype(np.float32)
    f = financial_vecs.astype(np.float32)
    m = macro_vecs.astype(np.float32)

    scaler_text = StandardScaler()
    scaler_fin = StandardScaler()
    scaler_macro = StandardScaler()

    t_scaled = scaler_text.fit_transform(t).astype(np.float32)
    f_scaled = scaler_fin.fit_transform(f).astype(np.float32)
    m_scaled = scaler_macro.fit_transform(m).astype(np.float32)

    hybrid_matrix = np.concatenate(
        [w_text * t_scaled, w_fin * f_scaled, w_macro * m_scaled],
        axis=1,
    ).astype(np.float32)

    scalers: Dict[str, StandardScaler] = {
        "text": scaler_text,
        "financial": scaler_fin,
        "macro": scaler_macro,
    }

    logger.info("Hybrid matrix shape: %s", hybrid_matrix.shape)
    return hybrid_matrix, scalers
