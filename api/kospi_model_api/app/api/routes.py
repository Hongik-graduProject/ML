from __future__ import annotations

from datetime import date
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_runtime_settings, get_store, verify_api_key
from app.config import Settings
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeaturePredictionRequest,
    InferencePrediction,
    LatestPredictionItem,
    ModelInfo,
    PredictionPage,
    PredictionSort,
    Signal,
)
from app.services.artifact_store import ArtifactStore
from app.services.predictor import FeatureValidationError, Predictor

router = APIRouter(prefix="/api/v1", dependencies=[Depends(verify_api_key)])
StoreDep = Annotated[ArtifactStore, Depends(get_store)]
SettingsDep = Annotated[Settings, Depends(get_runtime_settings)]


def _optional_date(value) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def _optional_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _latest_item(row: pd.Series) -> LatestPredictionItem:
    return LatestPredictionItem(
        date=pd.Timestamp(row["Date"]).date(),
        code=str(row["Code"]),
        name=None if pd.isna(row.get("Name")) else str(row.get("Name")),
        sector=None if pd.isna(row.get("Sector")) else str(row.get("Sector")),
        close=float(row["Close"]),
        predicted_excess_return_60d=float(row["Predicted_Excess_Return_60D"]),
        predicted_percentile=float(row["Predicted_Percentile"]),
        signal=Signal(str(row["Signal"])),
        model_agreement_score=float(row["Confidence_Score"]),
        model_disagreement=float(row["Model_Disagreement"]),
        scenario_price_if_kospi_flat=float(row["Scenario_Price_If_KOSPI_Flat"]),
        avg_trading_value_20=_optional_float(row.get("Avg_Trading_Value_20")),
        feature_coverage=float(row["Feature_Coverage"]),
        disclosure_date=_optional_date(row.get("disclosure_date")),
        financial_age_days=_optional_float(row.get("Financial_Age_Days")),
        financial_source=None if pd.isna(row.get("financial_source")) else str(row.get("financial_source")),
        macro_as_of_date=_optional_date(row.get("AsOfDate")),
        dna_age_days=_optional_float(row.get("DNA_Age_Days")),
    )


@router.get("/model/info", response_model=ModelInfo, tags=["model"])
def model_info(
    store: StoreDep,
    settings: SettingsDep,
) -> ModelInfo:
    if store.bundle is None:
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")
    latest_date = pd.Timestamp(store.predictions["Date"].max()).date()
    return ModelInfo(
        api_version=settings.app_version,
        model_version=settings.app_version,
        bundle_metadata_version=(
            str(store.bundle["version"]) if store.bundle.get("version") is not None else None
        ),
        target=str(store.bundle["target"]),
        target_definition=str(store.feature_schema.get(
            "target_definition",
            "stock 60-observation log return minus KOSPI log return over the same dates",
        )),
        feature_count=len(store.features),
        features=store.features,
        ensemble_xgb_weight=float(store.bundle["ensemble_xgb_weight"]),
        target_clip=[float(value) for value in store.bundle["target_clip"]],
        latest_prediction_date=latest_date,
        latest_prediction_count=len(store.predictions),
        notes=[
            "예측값은 KOSPI 대비 60거래일 초과수익률입니다.",
            "model_agreement_score는 확률이 아니라 Ridge/XGBoost 일치도 순위입니다.",
            "Top 10%는 양의 수익률 보장이 아니라 동일 시점 종목 간 상대순위입니다.",
            "bundle_metadata_version은 Colab artifact 내부의 레거시 표기이며 API release와 구분합니다.",
        ],
    )


@router.get("/predictions/latest", response_model=PredictionPage, tags=["predictions"])
def latest_predictions(
    store: StoreDep,
    signal: Signal | None = None,
    search: str | None = Query(default=None, min_length=1, max_length=40),
    min_agreement: float | None = Query(default=None, ge=0.0, le=1.0),
    sort: PredictionSort = PredictionSort.percentile,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
) -> PredictionPage:
    frame = store.predictions.copy()
    if signal is not None:
        frame = frame[frame["Signal"].astype(str).eq(signal.value)]
    if search:
        token = search.strip()
        name = frame.get("Name", pd.Series("", index=frame.index)).fillna("").astype(str)
        frame = frame[frame["Code"].str.contains(token, regex=False) | name.str.contains(token, case=False, regex=False)]
    if min_agreement is not None:
        frame = frame[frame["Confidence_Score"].astype(float).ge(min_agreement)]
    sort_map = {
        PredictionSort.percentile: "Predicted_Percentile",
        PredictionSort.expected_return: "Predicted_Excess_Return_60D",
        PredictionSort.agreement: "Confidence_Score",
    }
    frame = frame.sort_values(sort_map[sort], ascending=False)
    total = len(frame)
    page = frame.iloc[offset: offset + limit]
    return PredictionPage(
        as_of_date=pd.Timestamp(store.predictions["Date"].max()).date(),
        total=total,
        offset=offset,
        limit=limit,
        items=[_latest_item(row) for _, row in page.iterrows()],
    )


@router.get("/predictions/{code}", response_model=LatestPredictionItem, tags=["predictions"])
def prediction_by_code(code: str, store: StoreDep) -> LatestPredictionItem:
    normalized = str(code).strip().zfill(6)
    if not normalized.isdigit() or len(normalized) != 6:
        raise HTTPException(status_code=422, detail="종목코드는 6자리 숫자여야 합니다.")
    selected = store.predictions[store.predictions["Code"].eq(normalized)]
    if selected.empty:
        raise HTTPException(status_code=404, detail="해당 종목의 최신 예측이 없습니다.")
    return _latest_item(selected.iloc[0])


@router.post("/predict", response_model=InferencePrediction, tags=["inference"])
def predict_one(
    payload: FeaturePredictionRequest,
    store: StoreDep,
    settings: SettingsDep,
) -> InferencePrediction:
    predictor = Predictor(store, settings.min_feature_coverage)
    try:
        raw = predictor.predict_raw([payload])
    except FeatureValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return InferencePrediction(**predictor.format(raw, "latest_reference")[0])


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(
    payload: BatchPredictionRequest,
    store: StoreDep,
    settings: SettingsDep,
) -> BatchPredictionResponse:
    if len(payload.items) > settings.max_batch_size:
        raise HTTPException(status_code=413, detail=f"최대 batch 크기는 {settings.max_batch_size}입니다.")
    predictor = Predictor(store, settings.min_feature_coverage)
    try:
        raw = predictor.predict_raw(payload.items)
    except FeatureValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    items = [InferencePrediction(**item) for item in predictor.format(raw, payload.percentile_scope)]
    return BatchPredictionResponse(
        model_version=settings.app_version,
        percentile_scope=payload.percentile_scope,
        count=len(items),
        items=items,
    )
