from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KOSPI_API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "KOSPI DART ML API"
    app_version: str = "2.4.0"
    environment: str = "development"
    artifact_dir: Path = PROJECT_ROOT / "artifacts"
    model_filename: str = "ensemble_pipeline_v24.joblib"
    predictions_filename: str = "latest_kospi_predictions_v24.csv"
    schema_filename: str = "feature_schema_v24.json"
    manifest_filename: str = "data_manifest_v24.json"
    api_key: str | None = None
    allowed_origins: str = "http://localhost:3000,http://localhost:8080"
    min_feature_coverage: float = Field(default=0.75, ge=0.0, le=1.0)
    max_batch_size: int = Field(default=500, ge=1, le=5000)

    @property
    def model_path(self) -> Path:
        return self.artifact_dir / self.model_filename

    @property
    def predictions_path(self) -> Path:
        return self.artifact_dir / self.predictions_filename

    @property
    def schema_path(self) -> Path:
        return self.artifact_dir / self.schema_filename

    @property
    def manifest_path(self) -> Path:
        return self.artifact_dir / self.manifest_filename

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
