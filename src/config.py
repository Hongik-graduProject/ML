import os
from dataclasses import dataclass, field
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()


@dataclass
class DartConfig:
    api_key: str = field(default_factory=lambda: os.environ.get("DART_API_KEY", ""))
    request_interval: float = 0.6
    max_retries: int = 3
    report_code: str = "11011"


@dataclass
class DataConfig:
    years: List[int] = field(default_factory=lambda: list(range(2015, 2025)))
    db_path: str = "data/panel.db"
    cache_dir: str = "data/cache"


@dataclass
class FeatureConfig:
    text_model_name: str = "jhgan/ko-sbert-sts"
    text_dim: int = 768
    financial_dim: int = 10
    macro_dim: int = 3
    macro_factors: List[str] = field(
        default_factory=lambda: ["interest_rate", "fx_rate", "oil_price"]
    )
    rolling_window_months: int = 36


@dataclass
class ClusterConfig:
    weight_text: float = 0.6
    weight_financial: float = 0.2
    weight_macro: float = 0.2
    k_range: Tuple[int, int] = (2, 15)
    random_state: int = 42
    n_init: int = 10


@dataclass
class PredictionConfig:
    train_end_year: int = 2022
    val_year: int = 2023
    test_year: int = 2024
    momentum_window_days: int = 20
    models: List[str] = field(default_factory=lambda: ["ridge", "lightgbm"])


@dataclass
class GCPConfig:
    project_id: str = field(
        default_factory=lambda: os.environ.get("GCP_PROJECT_ID", "")
    )
    region: str = field(
        default_factory=lambda: os.environ.get("GCP_REGION", "asia-northeast3")
    )
    gcs_bucket: str = field(
        default_factory=lambda: os.environ.get("GCS_BUCKET", "")
    )


@dataclass
class Config:
    dart: DartConfig = field(default_factory=DartConfig)
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    gcp: GCPConfig = field(default_factory=GCPConfig)


config = Config()
