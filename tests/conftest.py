from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

FEATURES = [f"feature_{idx:02d}" for idx in range(31)]


class ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


@pytest.fixture
def artifact_dir(tmp_path: Path) -> Path:
    bundle = {
        "version": "test",
        "ridge": ConstantModel(0.01),
        "xgb": ConstantModel(0.03),
        "ensemble_xgb_weight": 0.7,
        "target_clip": [-0.5, 0.5],
        "features": FEATURES,
        "target": "Excess_Log_Return_60D",
    }
    joblib.dump(bundle, tmp_path / "ensemble_pipeline_v24.joblib")
    rows = []
    for idx in range(10):
        rows.append({
            "Date": "2026-10-02", "Code": f"{idx+1:06d}", "Name": f"기업{idx+1}",
            "Sector": None, "Close": 10000 + idx, "Predicted_Excess_Return_60D": -0.05 + idx * 0.01,
            "Predicted_Percentile": (idx + 1) / 10, "Signal": "Top 10%" if idx == 9 else "Neutral",
            "Confidence_Score": 0.5, "Model_Disagreement": 0.02,
            "Scenario_Price_If_KOSPI_Flat": 10000, "Avg_Trading_Value_20": 1e9,
            "Feature_Coverage": 1.0, "disclosure_date": "2026-03-20",
            "Financial_Age_Days": 196, "financial_source": "dart", "AsOfDate": "2026-10-02",
            "DNA_Age_Days": 0,
        })
    pd.DataFrame(rows).to_csv(tmp_path / "latest_kospi_predictions_v24.csv", index=False)
    (tmp_path / "feature_schema_v24.json").write_text(
        json.dumps({"features": FEATURES, "target_definition": "test"}), encoding="utf-8"
    )
    return tmp_path
