"""Walk-forward validation 백테스트."""
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.config import config
from src.prediction.model import evaluate, predict, train_model

logger = logging.getLogger(__name__)

# 피처/타깃 컬럼에서 제외할 메타 컬럼
_META_COLS = {"corp_name", "date"}
_TARGET_COL = "next_month_return"


def _split_df(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """데이터프레임을 train / val / test 구간으로 분할한다.

    분할 기준 (config.prediction):
    - train : date.year <= train_end_year
    - val   : date.year == val_year
    - test  : date.year == test_year

    Args:
        df: 'date' 컬럼이 포함된 피처 데이터프레임

    Returns:
        {'train': df, 'val': df, 'test': df}
    """
    date_col = pd.to_datetime(df["date"])
    train_mask = date_col.dt.year <= config.prediction.train_end_year
    val_mask = date_col.dt.year == config.prediction.val_year
    test_mask = date_col.dt.year == config.prediction.test_year

    return {
        "train": df[train_mask].copy(),
        "val": df[val_mask].copy(),
        "test": df[test_mask].copy(),
    }


def _extract_Xy(df: pd.DataFrame) -> tuple:
    """데이터프레임에서 피처 행렬과 타깃 벡터를 추출한다.

    Args:
        df: 피처 + 타깃 데이터프레임

    Returns:
        (X, y) — np.ndarray 쌍
    """
    drop_cols = _META_COLS | {_TARGET_COL}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].fillna(0.0).to_numpy(dtype=float)
    y = df[_TARGET_COL].to_numpy(dtype=float)
    return X, y


def walk_forward_backtest(
    df: pd.DataFrame,
    model_type: str = "ridge",
) -> Dict[str, Any]:
    """Walk-forward validation을 수행한다.

    Train: config.prediction.train_end_year 이전 연도까지
    Val  : config.prediction.val_year
    Test : config.prediction.test_year

    Args:
        df: build_prediction_features() 출력 데이터프레임.
            'date', 'corp_name', 'next_month_return' 컬럼 포함 필수.
        model_type: "ridge" 또는 "lightgbm"

    Returns:
        {
          'train': {rmse, mae, directional_accuracy, sharpe_ratio},
          'val':   {rmse, mae, directional_accuracy, sharpe_ratio},
          'test':  {rmse, mae, directional_accuracy, sharpe_ratio},
          'model_type': str,
          'n_train': int,
          'n_val': int,
          'n_test': int,
        }
    """
    splits = _split_df(df)

    n_train = len(splits["train"])
    n_val = len(splits["val"])
    n_test = len(splits["test"])

    logger.info(
        "Walk-forward 분할 — train=%d, val=%d, test=%d",
        n_train, n_val, n_test,
    )

    if n_train == 0:
        raise ValueError("학습 데이터가 비어 있습니다.")

    X_train, y_train = _extract_Xy(splits["train"])

    # NaN 타깃 행 제거
    valid_train = ~np.isnan(y_train)
    X_train, y_train = X_train[valid_train], y_train[valid_train]

    model = train_model(X_train, y_train, model_type=model_type)

    results: Dict[str, Any] = {
        "model_type": model_type,
        "n_train": int(valid_train.sum()),
        "n_val": n_val,
        "n_test": n_test,
    }

    for split_name in ("train", "val", "test"):
        split_df = splits[split_name]
        if len(split_df) == 0:
            logger.warning("%s 분할 데이터가 비어 있습니다.", split_name)
            results[split_name] = {
                "rmse": np.nan,
                "mae": np.nan,
                "directional_accuracy": np.nan,
                "sharpe_ratio": np.nan,
            }
            continue

        X_split, y_split = _extract_Xy(split_df)
        y_pred = predict(model, X_split)
        metrics = evaluate(y_split, y_pred)
        results[split_name] = metrics
        logger.info(
            "[%s] RMSE=%.4f, DA=%.4f, Sharpe=%.4f",
            split_name,
            metrics.get("rmse", np.nan),
            metrics.get("directional_accuracy", np.nan),
            metrics.get("sharpe_ratio", np.nan),
        )

    return results
