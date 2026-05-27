"""텍스트 인코더: ko-sbert-sts 모델로 한국어 텍스트를 768D 벡터로 인코딩."""

import logging
from typing import List, Optional

import numpy as np

from src.config import config

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """싱글턴 패턴으로 SentenceTransformer 모델을 한 번만 로드."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer model: %s", config.feature.text_model_name)
            _model = SentenceTransformer(config.feature.text_model_name)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load SentenceTransformer model: %s", e)
            raise
    return _model


def encode_text(text: Optional[str]) -> np.ndarray:
    """단일 텍스트를 768D 벡터로 인코딩.

    Args:
        text: 인코딩할 텍스트 문자열. None 또는 빈 문자열이면 제로 벡터 반환.

    Returns:
        np.ndarray: shape (768,) float32 벡터.
    """
    dim = config.feature.text_dim
    if not text or not text.strip():
        logger.debug("Empty or None text input; returning zero vector.")
        return np.zeros(dim, dtype=np.float32)

    try:
        model = _get_model()
        vector = model.encode(text, convert_to_numpy=True)
        return vector.astype(np.float32)
    except Exception as e:
        logger.error("Error encoding text: %s", e)
        return np.zeros(dim, dtype=np.float32)


def encode_texts_batch(texts: List[Optional[str]]) -> np.ndarray:
    """텍스트 리스트를 배치로 인코딩.

    Args:
        texts: 텍스트 문자열 리스트. None 또는 빈 문자열 항목은 제로 벡터로 처리.

    Returns:
        np.ndarray: shape (N, 768) float32 행렬.
    """
    dim = config.feature.text_dim
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)

    # 유효/무효 인덱스 분리
    valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
    valid_texts = [texts[i] for i in valid_indices]

    result = np.zeros((len(texts), dim), dtype=np.float32)

    if not valid_texts:
        logger.debug("All texts are empty or None; returning zero matrix.")
        return result

    try:
        model = _get_model()
        encoded = model.encode(valid_texts, convert_to_numpy=True, batch_size=32)
        for idx, arr in zip(valid_indices, encoded):
            result[idx] = arr.astype(np.float32)
    except Exception as e:
        logger.error("Error in batch encoding: %s", e)

    return result
