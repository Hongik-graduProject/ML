from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import Settings

REQUIRED_BUNDLE_KEYS = {
    "ridge", "xgb", "ensemble_xgb_weight", "target_clip", "features", "target"
}

REQUIRED_PREDICTION_COLUMNS = {
    "Date", "Code", "Close", "Predicted_Excess_Return_60D",
    "Predicted_Percentile", "Signal", "Confidence_Score", "Model_Disagreement",
    "Scenario_Price_If_KOSPI_Flat", "Feature_Coverage",
}


class ArtifactValidationError(RuntimeError):
    pass


class ArtifactStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.bundle: dict[str, Any] | None = None
        self.predictions = pd.DataFrame()
        self.feature_schema: dict[str, Any] = {}
        self.manifest: dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        return self.bundle is not None and not self.predictions.empty

    @property
    def features(self) -> list[str]:
        if self.bundle is None:
            return []
        return list(self.bundle["features"])

    def load(self) -> None:
        self.predictions = self._load_predictions(self.settings.predictions_path)
        self.feature_schema = self._load_optional_json(self.settings.schema_path)
        self.manifest = self._load_optional_json(self.settings.manifest_path)
        # 최신 예측 조회는 모델 역직렬화 실패와 독립적으로 사용할 수 있도록 CSV를 먼저 로드한다.
        self.bundle = self._load_model(self.settings.model_path)
        self._cross_validate()

    def _load_model(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise ArtifactValidationError(f"모델 파일이 없습니다: {path}")
        try:
            bundle = joblib.load(path)
        except Exception as exc:
            raise ArtifactValidationError(
                "모델 로드에 실패했습니다. Colab과 API의 scikit-learn/xgboost 버전을 확인하십시오."
            ) from exc
        if not isinstance(bundle, dict):
            raise ArtifactValidationError("모델 artifact는 dict 형식이어야 합니다.")
        missing = REQUIRED_BUNDLE_KEYS - set(bundle)
        if missing:
            raise ArtifactValidationError(f"모델 bundle 필수 키 누락: {sorted(missing)}")
        features = bundle["features"]
        if not isinstance(features, list) or not features or len(features) != len(set(features)):
            raise ArtifactValidationError("모델 features가 비어 있거나 중복되었습니다.")
        clip = bundle["target_clip"]
        if not isinstance(clip, (list, tuple)) or len(clip) != 2 or clip[0] >= clip[1]:
            raise ArtifactValidationError("target_clip 형식이 올바르지 않습니다.")
        return bundle

    def _load_predictions(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise ArtifactValidationError(f"최신 예측 파일이 없습니다: {path}")
        frame = pd.read_csv(path, dtype={"Code": "string"})
        missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
        if missing:
            raise ArtifactValidationError(f"최신 예측 컬럼 누락: {sorted(missing)}")
        if frame.empty:
            raise ArtifactValidationError("최신 예측 파일이 비어 있습니다.")
        frame["Code"] = frame["Code"].str.replace(r"\.0$", "", regex=True).str.zfill(6)
        if not frame["Code"].str.fullmatch(r"\d{6}").all():
            raise ArtifactValidationError("최신 예측에 잘못된 종목코드가 있습니다.")
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        if frame["Date"].isna().any():
            raise ArtifactValidationError("최신 예측 Date 변환에 실패했습니다.")
        if frame["Code"].duplicated().any():
            raise ArtifactValidationError("최신 예측에 중복 종목코드가 있습니다.")
        return frame.sort_values("Predicted_Percentile", ascending=False).reset_index(drop=True)

    @staticmethod
    def _load_optional_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ArtifactValidationError(f"JSON artifact는 object여야 합니다: {path}")
        return payload

    def _cross_validate(self) -> None:
        if self.bundle is None:
            raise ArtifactValidationError("모델이 로드되지 않았습니다.")
        if self.feature_schema:
            schema_features = self.feature_schema.get("features")
            if schema_features and list(schema_features) != self.features:
                raise ArtifactValidationError("모델 bundle과 feature schema의 Feature 순서가 다릅니다.")
        numeric = [
            "Close", "Predicted_Excess_Return_60D", "Predicted_Percentile",
            "Confidence_Score", "Model_Disagreement", "Feature_Coverage",
        ]
        values = self.predictions[numeric].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any() or np.isinf(values.to_numpy()).any():
            raise ArtifactValidationError("최신 예측 핵심 수치에 NaN/무한대가 있습니다.")
