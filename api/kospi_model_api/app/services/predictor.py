from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.schemas import FeaturePredictionRequest, Signal
from app.services.artifact_store import ArtifactStore


class FeatureValidationError(ValueError):
    pass


@dataclass
class RawPrediction:
    request: FeaturePredictionRequest
    ridge: float
    xgb: float
    ensemble: float
    disagreement: float
    coverage: float
    missing: list[str]


class Predictor:
    def __init__(self, store: ArtifactStore, min_feature_coverage: float):
        if not store.ready or store.bundle is None:
            raise RuntimeError("ArtifactStore가 준비되지 않았습니다.")
        self.store = store
        self.bundle = store.bundle
        self.features = store.features
        self.min_feature_coverage = min_feature_coverage
        self.lower, self.upper = map(float, self.bundle["target_clip"])
        self.weight = float(self.bundle["ensemble_xgb_weight"])
        latest = store.predictions
        self.reference_log_returns = np.log1p(
            latest["Predicted_Excess_Return_60D"].astype(float).clip(lower=-0.999999)
        ).to_numpy()
        self.reference_disagreements = latest["Model_Disagreement"].astype(float).to_numpy()

    def predict_raw(self, items: list[FeaturePredictionRequest]) -> list[RawPrediction]:
        records: list[dict[str, float]] = []
        metadata: list[tuple[float, list[str]]] = []
        allowed = set(self.features)
        for item in items:
            unknown = sorted(set(item.features) - allowed)
            if unknown:
                raise FeatureValidationError(f"허용되지 않은 Feature: {unknown}")
            row = {name: item.features.get(name, np.nan) for name in self.features}
            frame = pd.Series(row, dtype="float64").replace([np.inf, -np.inf], np.nan)
            missing = frame[frame.isna()].index.tolist()
            coverage = 1.0 - len(missing) / len(self.features)
            if coverage < self.min_feature_coverage:
                raise FeatureValidationError(
                    f"{item.code} Feature coverage {coverage:.1%}가 최소 "
                    f"{self.min_feature_coverage:.1%}보다 낮습니다. 누락: {missing}"
                )
            records.append(frame.to_dict())
            metadata.append((coverage, missing))

        x = pd.DataFrame(records, columns=self.features)
        ridge_values = np.clip(self.bundle["ridge"].predict(x), self.lower, self.upper)
        xgb_values = np.clip(self.bundle["xgb"].predict(x), self.lower, self.upper)
        ensemble_values = self.weight * xgb_values + (1.0 - self.weight) * ridge_values

        result = []
        for request, ridge, xgb, ensemble, meta in zip(
            items, ridge_values, xgb_values, ensemble_values, metadata
        ):
            result.append(RawPrediction(
                request=request,
                ridge=float(ridge),
                xgb=float(xgb),
                ensemble=float(ensemble),
                disagreement=float(abs(xgb - ridge)),
                coverage=meta[0],
                missing=meta[1],
            ))
        return result

    def format(self, raw: list[RawPrediction], percentile_scope: str) -> list[dict]:
        if percentile_scope == "batch" and len(raw) >= 2:
            return_values = np.array([item.ensemble for item in raw])
            disagreements = np.array([item.disagreement for item in raw])
        else:
            return_values = self.reference_log_returns
            disagreements = self.reference_disagreements

        output = []
        for item in raw:
            percentile = self._percentile(return_values, item.ensemble)
            agreement = 1.0 - self._percentile(disagreements, item.disagreement)
            close = item.request.close
            output.append({
                "code": item.request.code,
                "name": item.request.name,
                "as_of_date": item.request.as_of_date,
                "predicted_excess_log_return_60d": item.ensemble,
                "predicted_excess_return_60d": float(np.expm1(item.ensemble)),
                "predicted_percentile": percentile,
                "signal": self._signal(percentile),
                "model_agreement_score": agreement,
                "model_disagreement": item.disagreement,
                "ridge_prediction": item.ridge,
                "xgb_prediction": item.xgb,
                "feature_coverage": item.coverage,
                "missing_features": item.missing,
                "scenario_price_if_kospi_flat": (
                    float(close * np.exp(item.ensemble)) if close is not None else None
                ),
            })
        return output

    @staticmethod
    def _percentile(reference: np.ndarray, value: float) -> float:
        clean = reference[np.isfinite(reference)]
        if clean.size == 0:
            raise RuntimeError("Percentile 기준 분포가 비어 있습니다.")
        return float(np.mean(clean <= value))

    @staticmethod
    def _signal(percentile: float) -> Signal:
        if percentile <= 0.10:
            return Signal.bottom_10
        if percentile <= 0.30:
            return Signal.below_average
        if percentile <= 0.70:
            return Signal.neutral
        if percentile <= 0.90:
            return Signal.above_average
        return Signal.top_10
