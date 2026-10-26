"""Colab 프로젝트의 API 필수 artifact를 하나의 ZIP으로 내보낸다."""
from __future__ import annotations

import importlib.metadata
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/KOSPI_DART_ML")
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
EXPORT_DIR = PROJECT_ROOT / "api_export_v24"
REQUIRED = [
    "ensemble_pipeline_v24.joblib",
    "latest_kospi_predictions_v24.csv",
    "feature_schema_v24.json",
    "data_manifest_v24.json",
]
PACKAGES = ["python", "numpy", "pandas", "scikit-learn", "xgboost", "joblib"]


def version_of(name: str) -> str:
    if name == "python":
        import platform
        return platform.python_version()
    return importlib.metadata.version(name)


def main() -> None:
    missing = [name for name in REQUIRED if not (ARTIFACT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"먼저 V2.4 노트북을 끝까지 실행하십시오. 누락: {missing}")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        shutil.copy2(ARTIFACT_DIR / name, EXPORT_DIR / name)
    versions = {
        "created_at": datetime.now(UTC).isoformat(),
        "packages": {name: version_of(name) for name in PACKAGES},
    }
    (EXPORT_DIR / "runtime_versions_v24.json").write_text(
        json.dumps(versions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive = shutil.make_archive(str(PROJECT_ROOT / "kospi_api_artifacts_v24"), "zip", EXPORT_DIR)
    print("API artifact ZIP:", archive)


if __name__ == "__main__":
    main()
