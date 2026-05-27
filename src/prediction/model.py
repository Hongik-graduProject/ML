"""Ridge Regression (baseline) + LightGBM (main) 예측 모델."""
import logging
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import config

logger = logging.getLogger(__name__)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "ridge",
) -> Any:
    """예측 모델을 학습한다.

    Args:
        X_train: 학습 피처 배열 (n_samples, n_features)
        y_train: 학습 타깃 배열 (n_samples,)
        model_type: "ridge" 또는 "lightgbm"

    Returns:
        학습된 모델 객체
    """
    if model_type == "ridge":
        model = Ridge(alpha=1.0, random_state=config.cluster.random_state)
        model.fit(X_train, y_train)
        logger.info("Ridge 모델 학습 완료: 샘플수=%d, 피처수=%d", *X_train.shape)

    elif model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError(
                "lightgbm 패키지가 설치되지 않았습니다. "
                "`pip install lightgbm`으로 설치하세요."
            ) from exc

        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=config.cluster.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        logger.info(
            "LightGBM 모델 학습 완료: 샘플수=%d, 피처수=%d", *X_train.shape
        )

    else:
        raise ValueError(
            f"지원하지 않는 model_type: '{model_type}'. "
            "지원 목록: 'ridge', 'lightgbm'"
        )

    return model


def predict(model: Any, X_test: np.ndarray) -> np.ndarray:
    """학습된 모델로 예측값을 반환한다.

    Args:
        model: 학습된 모델 (Ridge 또는 LGBMRegressor)
        X_test: 테스트 피처 배열 (n_samples, n_features)

    Returns:
        예측 수익률 배열 (n_samples,)
    """
    preds: np.ndarray = model.predict(X_test)
    logger.debug("예측 완료: %d 샘플", len(preds))
    return preds


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """예측 성능 지표를 계산한다.

    지표:
    - rmse: Root Mean Squared Error
    - mae: Mean Absolute Error
    - directional_accuracy: 방향 일치율 (예측 부호 == 실제 부호)
    - sharpe_ratio: (평균 예측 수익률) / (표준편차 + eps) × sqrt(12)
                    월별 기준으로 연환산

    Args:
        y_true: 실제 수익률 배열 (n_samples,)
        y_pred: 예측 수익률 배열 (n_samples,)

    Returns:
        {metric_name: value} 딕셔너리
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # NaN 행 제거
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        logger.warning("유효한 평가 샘플이 없습니다.")
        return {"rmse": np.nan, "mae": np.nan,
                "directional_accuracy": np.nan, "sharpe_ratio": np.nan}

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    # 방향 정확도: 0 수익률은 양수 방향으로 처리
    directional_accuracy = float(
        (np.sign(y_pred) == np.sign(y_true)).mean()
    )

    # Sharpe Ratio: 예측 수익률 기준 (포트폴리오 시뮬레이션 아님)
    eps = 1e-8
    sharpe_ratio = float(
        (y_pred.mean() / (y_pred.std() + eps)) * np.sqrt(12)
    )

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": directional_accuracy,
        "sharpe_ratio": sharpe_ratio,
    }
    logger.info(
        "평가 지표 — RMSE=%.4f, MAE=%.4f, DA=%.4f, Sharpe=%.4f",
        rmse, mae, directional_accuracy, sharpe_ratio,
    )
    return metrics
