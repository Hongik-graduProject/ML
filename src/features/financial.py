"""재무 피처 모듈: 재무제표 raw 데이터에서 10D 피처 벡터 생성."""

import logging
from typing import Optional

import numpy as np

from src.config import config

logger = logging.getLogger(__name__)


def safe_div(num: float, den: float) -> float:
    """0 나누기 방지 안전 나눗셈.

    Args:
        num: 분자.
        den: 분모.

    Returns:
        num / den, 분모가 0이면 0.0 반환.
    """
    if den == 0 or den is None:
        return 0.0
    return float(num) / float(den)


def compute_financial_vector(
    current: dict,
    previous: Optional[dict] = None,
) -> np.ndarray:
    """재무제표 딕셔너리에서 10D 피처 벡터 생성.

    Args:
        current: 현재 연도 재무 데이터. 기대 키:
            - revenue (매출액)
            - op_income (영업이익)
            - net_income (당기순이익)
            - assets (자산총계)
            - liabilities (부채총계)
            - equity (자본총계)
        previous: 전년도 재무 데이터 (동일 구조). None이면 YoY 성장률은 0.0.

    Returns:
        np.ndarray: shape (10,) float32 벡터.

    Index mapping:
        0: 영업이익률   = op_income / revenue
        1: 순이익률     = net_income / revenue
        2: 부채비율     = liabilities / equity
        3: 매출 규모    = log1p(abs(revenue))
        4: ROA          = net_income / assets
        5: ROE          = net_income / equity
        6: 자산회전율   = revenue / assets
        7: 유동비율proxy = equity / liabilities
        8: 매출 성장률  = (current_revenue - prev_revenue) / prev_revenue
        9: 영업이익 성장률 = (current_op_income - prev_op_income) / prev_op_income
    """
    dim = config.feature.financial_dim  # 10
    vec = np.zeros(dim, dtype=np.float32)

    revenue = float(current.get("revenue", 0) or 0)
    op_income = float(current.get("op_income", 0) or 0)
    net_income = float(current.get("net_income", 0) or 0)
    assets = float(current.get("assets", 0) or 0)
    liabilities = float(current.get("liabilities", 0) or 0)
    equity = float(current.get("equity", 0) or 0)

    vec[0] = safe_div(op_income, revenue)       # 영업이익률
    vec[1] = safe_div(net_income, revenue)       # 순이익률
    vec[2] = safe_div(liabilities, equity)       # 부채비율
    vec[3] = float(np.log1p(abs(revenue)))       # 매출 규모 (log)
    vec[4] = safe_div(net_income, assets)        # ROA
    vec[5] = safe_div(net_income, equity)        # ROE
    vec[6] = safe_div(revenue, assets)           # 자산회전율
    vec[7] = safe_div(equity, liabilities)       # 유동비율 proxy

    if previous is not None:
        prev_revenue = float(previous.get("revenue", 0) or 0)
        prev_op_income = float(previous.get("op_income", 0) or 0)
        vec[8] = safe_div(revenue - prev_revenue, prev_revenue)           # 매출 성장률
        vec[9] = safe_div(op_income - prev_op_income, prev_op_income)     # 영업이익 성장률
    else:
        vec[8] = 0.0
        vec[9] = 0.0

    logger.debug("Financial vector computed: %s", vec)
    return vec
