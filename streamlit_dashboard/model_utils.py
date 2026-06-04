
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FEATURES = [
    "roe", "roe_diff",
    "op_margin", "op_margin_diff",
    "debt_ratio", "debt_ratio_diff",
    "op_income", "op_income_diff",
    "Interest_stock", "FX_stock", "Oil_stock", "Gold_stock",
    "Interest_theme", "FX_theme", "Oil_theme", "Gold_theme",
]
TARGET = "Target_Return"
TARGET_DAYS = 60
BASE_FIN_COLS = ["roe", "op_margin", "debt_ratio", "op_income"]
SENSITIVITY_COLS = [
    "Interest_stock", "FX_stock", "Oil_stock", "Gold_stock",
    "Interest_theme", "FX_theme", "Oil_theme", "Gold_theme",
]
NUMERIC_COLS = [
    "Close", "Year", "Theme",
    "revenue", "op_income", "net_income", "assets", "liabilities", "equity",
    "op_margin", "roe", "debt_ratio", "net_margin",
] + SENSITIVITY_COLS

@dataclass
class PredictionResult:
    name: str
    code: str
    date: pd.Timestamp
    year: int
    current_price: float
    predicted_return: float
    predicted_price: float
    actual_return: float | None
    actual_price: float | None
    error_pctp: float | None
    sample: pd.DataFrame


def _normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일이 없습니다: {path}")
    df = pd.read_csv(path, dtype={"Code": str}, low_memory=False)
    required = {"Date", "Code", "Name", "Close", "Year"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Code"] = _normalize_code(df["Code"])
    df = _coerce_numeric(df)
    df = df.dropna(subset=["Date", "Code", "Name", "Close", "Year"]).copy()
    df["Year"] = df["Year"].astype(int)
    df = df.sort_values(["Code", "Date"]).reset_index(drop=True)
    return df


def prepare_model_frames(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw_df.copy().sort_values(["Code", "Date"]).reset_index(drop=True)
    df = _coerce_numeric(df)
    for col in BASE_FIN_COLS + SENSITIVITY_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # 재무/민감도 값 보정: 기업별 앞/뒤 채움, 그래도 없으면 중앙값/0 처리
    fill_cols = BASE_FIN_COLS + SENSITIVITY_COLS
    df[fill_cols] = df.groupby("Code", group_keys=False)[fill_cols].ffill()
    df[fill_cols] = df.groupby("Code", group_keys=False)[fill_cols].bfill()
    for col in fill_cols:
        if df[col].isna().all():
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(df[col].median())

    for col in BASE_FIN_COLS:
        df[f"{col}_diff"] = df.groupby("Code")[col].diff().fillna(0.0)

    df[TARGET] = df.groupby("Code")["Close"].shift(-TARGET_DAYS) / df["Close"] - 1
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    inference_df = df.dropna(subset=["Date", "Code", "Name", "Close", "Year"]).copy()
    train_df = df.dropna(subset=[TARGET]).copy()
    return inference_df, train_df


def get_available_train_years(train_df: pd.DataFrame) -> list[int]:
    return sorted(train_df["Year"].dropna().astype(int).unique())


def train_model(train_df: pd.DataFrame, train_years: Iterable[int] | None = None, max_rows: int = 120_000):
    if train_df.empty:
        raise ValueError("학습 가능한 행이 없습니다. 기업별로 60일 뒤 주가가 있어야 Target_Return을 만들 수 있습니다.")
    years = get_available_train_years(train_df)
    if train_years is None:
        train_years = years[-3:] if len(years) >= 3 else years
    train_years = list(map(int, train_years))
    use_df = train_df[train_df["Year"].isin(train_years)].copy()
    if use_df.empty:
        use_df = train_df.copy()
        train_years = years
    if len(use_df) > max_rows:
        use_df = use_df.sample(max_rows, random_state=42).sort_values(["Code", "Date"])
    X = use_df[FEATURES]
    y = use_df[TARGET]

    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=60,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=1,
            random_state=42,
        )
        model.fit(X, y)
        model._model_name_for_ui = "XGBoostRegressor"
        return model, train_years, len(use_df)
    except Exception as e:
        # XGBoost가 Mac 환경에서 충돌하면 과제 시연이 중단되지 않도록 sklearn 모델로 대체
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.05, random_state=42)
        model.fit(X, y)
        model._model_name_for_ui = f"Fallback: HistGradientBoostingRegressor (XGBoost 오류: {type(e).__name__})"
        return model, train_years, len(use_df)


def predict_company(model, inference_df: pd.DataFrame, name: str, year: int, as_of: str | None = None) -> PredictionResult:
    target_data = inference_df[(inference_df["Name"] == name) & (inference_df["Year"] == int(year))].sort_values("Date")
    if as_of:
        as_of_dt = pd.to_datetime(as_of)
        target_data = target_data[target_data["Date"] <= as_of_dt]
    if target_data.empty:
        raise ValueError(f"{name} / {year} 조건에 맞는 데이터가 없습니다.")
    sample = target_data.tail(1).copy()
    pred_return = float(model.predict(sample[FEATURES])[0])
    current_price = float(sample["Close"].iloc[0])
    actual_return = sample[TARGET].iloc[0] if TARGET in sample.columns else np.nan
    actual_return_val = None if pd.isna(actual_return) else float(actual_return)
    actual_price = None if actual_return_val is None else current_price * (1 + actual_return_val)
    error_pctp = None if actual_return_val is None else abs(pred_return - actual_return_val) * 100
    return PredictionResult(
        name=name, code=str(sample["Code"].iloc[0]), date=pd.to_datetime(sample["Date"].iloc[0]),
        year=int(year), current_price=current_price, predicted_return=pred_return,
        predicted_price=current_price * (1 + pred_return), actual_return=actual_return_val,
        actual_price=actual_price, error_pctp=error_pctp, sample=sample,
    )


def build_financial_summary(raw_df: pd.DataFrame, name: str) -> pd.DataFrame:
    cols = ["Year", "revenue", "op_income", "net_income", "assets", "liabilities", "equity", "op_margin", "roe", "net_margin", "debt_ratio"]
    available = [c for c in cols if c in raw_df.columns]
    if not available:
        return pd.DataFrame()
    out = raw_df[raw_df["Name"] == name][available].copy()
    return out.drop_duplicates(subset=["Year"]).sort_values("Year")


def build_stock_history(raw_df: pd.DataFrame, name: str) -> pd.DataFrame:
    return raw_df[raw_df["Name"] == name][["Date", "Close", "Year"]].dropna(subset=["Date", "Close"]).sort_values("Date")


def feature_contribution_table(model) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    else:
        return pd.DataFrame({"Feature": FEATURES, "Importance": [0.0] * len(FEATURES)})
    return pd.DataFrame({"Feature": FEATURES, "Importance": vals}).sort_values("Importance", ascending=False).reset_index(drop=True)
