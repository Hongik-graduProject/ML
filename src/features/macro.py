"""매크로 감도 Beta 산출 모듈: 월별 주가 수익률 vs 거시지표 변동률 다중선형회귀."""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import config

logger = logging.getLogger(__name__)

_MIN_DATA_POINTS = 12  # 최소 데이터포인트 수


def compute_macro_betas(
    stock_returns: pd.Series,
    macro_df: pd.DataFrame,
) -> np.ndarray:
    """기업별 매크로 감도 Beta 3D 벡터 산출.

    월별 주가 수익률을 종속변수로, 월별 거시지표 변동률(금리/환율/유가)을
    독립변수로 하는 다중선형회귀를 수행한다.
    config.feature.rolling_window_months(36개월) 롤링 윈도우를 적용하여
    가장 최근 윈도우의 Beta를 반환한다.

    Args:
        stock_returns: 월별 주가 수익률 pd.Series (index: DatetimeIndex 또는 period).
        macro_df: 월별 거시지표 변동률 pd.DataFrame.
            컬럼: config.feature.macro_factors = ["interest_rate", "fx_rate", "oil_price"].
            index는 stock_returns와 동일한 타입이어야 한다.

    Returns:
        np.ndarray: shape (3,) float32 벡터 [β_interest, β_fx, β_oil].
        데이터 포인트가 12개 미만이면 제로 벡터 반환.
    """
    macro_dim = config.feature.macro_dim  # 3
    zero_vec = np.zeros(macro_dim, dtype=np.float32)

    if stock_returns is None or macro_df is None:
        logger.warning("stock_returns or macro_df is None; returning zero vector.")
        return zero_vec

    macro_factors = config.feature.macro_factors
    window = config.feature.rolling_window_months

    # 공통 인덱스 정렬
    common_idx = stock_returns.index.intersection(macro_df.index)
    if len(common_idx) < _MIN_DATA_POINTS:
        logger.warning(
            "Insufficient data points (%d < %d); returning zero vector.",
            len(common_idx),
            _MIN_DATA_POINTS,
        )
        return zero_vec

    y = stock_returns.loc[common_idx].values.astype(float)

    # macro_df에서 필요한 컬럼만 추출 (없는 컬럼은 0으로 대체)
    X_parts = []
    for factor in macro_factors:
        if factor in macro_df.columns:
            X_parts.append(macro_df.loc[common_idx, factor].values.astype(float))
        else:
            logger.warning("Macro factor '%s' not found in macro_df; using zeros.", factor)
            X_parts.append(np.zeros(len(common_idx), dtype=float))

    X = np.column_stack(X_parts)  # shape (N, 3)

    # 롤링 윈도우: 가장 최근 window 개월
    if len(common_idx) >= window:
        y = y[-window:]
        X = X[-window:]

    # NaN 제거
    valid_mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y_clean = y[valid_mask]
    X_clean = X[valid_mask]

    if len(y_clean) < _MIN_DATA_POINTS:
        logger.warning(
            "After NaN removal, insufficient data points (%d < %d); returning zero vector.",
            len(y_clean),
            _MIN_DATA_POINTS,
        )
        return zero_vec

    try:
        model = LinearRegression(fit_intercept=True)
        model.fit(X_clean, y_clean)
        betas = model.coef_.astype(np.float32)
        logger.debug("Macro betas computed: %s", betas)
        return betas
    except Exception as e:
        logger.error("Error computing macro betas: %s", e)
        return zero_vec
