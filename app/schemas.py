from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Signal(str, Enum):
    bottom_10 = "Bottom 10%"
    below_average = "Below Average"
    neutral = "Neutral"
    above_average = "Above Average"
    top_10 = "Top 10%"


class PredictionSort(str, Enum):
    percentile = "percentile"
    expected_return = "expected_return"
    agreement = "agreement"


class FeaturePredictionRequest(StrictModel):
    code: str = Field(description="6자리 종목코드", examples=["005930"])
    as_of_date: date | None = Field(default=None, description="Feature 기준일")
    close: float | None = Field(default=None, gt=0, description="시나리오 가격 계산용 현재가")
    name: str | None = None
    features: dict[str, float | None] = Field(
        description="feature_schema_v24.json에 정의된 31개 Feature"
    )

    @field_validator("code", mode="before")
    @classmethod
    def normalize_code(cls, value: Any) -> str:
        text = str(value).strip()
        if not text.isdigit() or len(text) > 6:
            raise ValueError("code는 최대 6자리 숫자여야 합니다.")
        return text.zfill(6)


class BatchPredictionRequest(StrictModel):
    items: list[FeaturePredictionRequest] = Field(min_length=1)
    percentile_scope: str = Field(
        default="batch",
        pattern="^(batch|latest_reference)$",
        description="batch: 요청 내 상대순위, latest_reference: 최신 저장 예측분포 대비 순위",
    )


class InferencePrediction(StrictModel):
    code: str
    name: str | None
    as_of_date: date | None
    predicted_excess_log_return_60d: float
    predicted_excess_return_60d: float
    predicted_percentile: float
    signal: Signal
    model_agreement_score: float
    model_disagreement: float
    ridge_prediction: float
    xgb_prediction: float
    feature_coverage: float
    missing_features: list[str]
    scenario_price_if_kospi_flat: float | None


class BatchPredictionResponse(StrictModel):
    model_version: str
    percentile_scope: str
    count: int
    items: list[InferencePrediction]


class LatestPredictionItem(StrictModel):
    date: date
    code: str
    name: str | None
    sector: str | None
    close: float
    predicted_excess_return_60d: float
    predicted_percentile: float
    signal: Signal
    model_agreement_score: float
    model_disagreement: float
    scenario_price_if_kospi_flat: float
    avg_trading_value_20: float | None
    feature_coverage: float
    disclosure_date: date | None
    financial_age_days: float | None
    financial_source: str | None
    macro_as_of_date: date | None
    dna_age_days: float | None


class PredictionPage(StrictModel):
    as_of_date: date
    total: int
    offset: int
    limit: int
    items: list[LatestPredictionItem]


class ModelInfo(StrictModel):
    api_version: str
    model_version: str
    bundle_metadata_version: str | None
    target: str
    target_definition: str
    feature_count: int
    features: list[str]
    ensemble_xgb_weight: float
    target_clip: list[float]
    latest_prediction_date: date
    latest_prediction_count: int
    notes: list[str]


class HealthResponse(StrictModel):
    status: str
    api_version: str
    model_loaded: bool
    predictions_loaded: bool
    detail: str | None = None
