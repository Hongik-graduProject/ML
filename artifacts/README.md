# 필요한 모델 산출물

이 폴더에 Colab V2.4 실행 후 생성되는 다음 파일을 복사합니다.

1. `ensemble_pipeline_v24.joblib` - Ridge/XGBoost 모델 bundle
2. `latest_kospi_predictions_v24.csv` - 최신 종목별 예측
3. `feature_schema_v24.json` - Feature 순서 및 Target 정의
4. `data_manifest_v24.json` - 학습 데이터·설정 정보
5. `runtime_versions_v24.json` - 선택 사항이지만 배포 시 강력 권장

Colab에서 `scripts/export_api_artifacts.py`를 실행하면 필수 파일과 런타임 버전이
`kospi_api_artifacts_v24.zip`으로 생성됩니다.

`joblib` 파일은 pickle 기반이므로 신뢰할 수 있는 본인 산출물만 로드해야 합니다.
또한 `runtime_versions_v24.json`의 Python, numpy, scikit-learn, xgboost 버전을
배포 환경과 동일하게 맞추는 것이 안전합니다.
