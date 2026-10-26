# KOSPI DART ML API V2.4

KOSPI 전체 기업의 DART 재무정보, 가격 Feature, Macro DNA를 이용하여 향후
60거래일 KOSPI 대비 초과수익률을 제공하는 FastAPI 서비스입니다.

## 제공 기능

- Colab에서 미리 계산한 최신 664개 종목 예측 조회
- 종목코드·이름·Signal 검색 및 정렬
- 저장된 Ridge/XGBoost ensemble을 이용한 단건·배치 추론
- 31개 Feature 순서, Target, 모델 버전 조회
- artifact 시작 검증과 readiness/liveness health check
- 선택형 `X-API-Key` 인증, CORS 환경변수, Docker 실행

## API 구조

| Method | Path | 설명 |
|---|---|---|
| GET | `/health/live` | 프로세스 생존 확인 |
| GET | `/health/ready` | 모델·최신 예측 로드 확인 |
| GET | `/api/v1/model/info` | Feature 목록과 모델 메타데이터 |
| GET | `/api/v1/predictions/latest` | 최신 예측 목록·필터·페이지네이션 |
| GET | `/api/v1/predictions/{code}` | 종목코드별 최신 예측 |
| POST | `/api/v1/predict` | 31개 Feature 단건 추론 |
| POST | `/api/v1/predict/batch` | 최대 500건 배치 추론 |

Swagger UI는 `/docs`, ReDoc은 `/redoc`, OpenAPI JSON은 `/openapi.json`입니다.

## 1. Colab에서 artifact 내보내기

V2.4 노트북을 마지막 셀까지 실행한 뒤 `scripts/export_api_artifacts.py` 내용을
Colab 셀에서 실행합니다. 생성된 `kospi_api_artifacts_v24.zip`의 파일을 이 프로젝트의
`artifacts/` 폴더에 넣습니다.

필수 파일은 다음 네 개입니다.

```text
artifacts/
├── ensemble_pipeline_v24.joblib
├── latest_kospi_predictions_v24.csv
├── feature_schema_v24.json
└── data_manifest_v24.json
```

## 2. 로컬 실행

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env
PYTHONPATH=. python scripts/validate_artifacts.py
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 3. Docker 실행

```bash
cp .env.example .env
docker compose up --build -d
curl http://localhost:8000/health/ready
```

운영환경에서는 `.env`의 `KOSPI_API_API_KEY`를 반드시 임의의 긴 값으로 바꾸고,
백엔드에서 모든 `/api/v1/*` 요청에 `X-API-Key` 헤더를 넣습니다.

## 4. 최신 예측 조회 예시

```bash
curl -H "X-API-Key: $KOSPI_API_API_KEY" \
  "http://localhost:8000/api/v1/predictions/latest?signal=Top%2010%25&limit=20"

curl -H "X-API-Key: $KOSPI_API_API_KEY" \
  "http://localhost:8000/api/v1/predictions/005930"
```

응답의 `model_agreement_score`는 성공확률이 아닙니다. 노트북의
`Confidence_Score`를 오해하지 않도록 API에서는 Ridge와 XGBoost 예측의 일치도
순위라는 의미의 이름으로 변경했습니다.

V2.4 노트북이 저장한 model bundle 내부에는 레거시 `version: 2.1` 표기가 남아 있을 수
있습니다. API의 `model_version`은 배포 release인 `2.4.0`을 반환하고, 원본 값은
`bundle_metadata_version`으로 별도 공개합니다.

## 5. 직접 추론 예시

먼저 `/api/v1/model/info`에서 정확한 Feature 이름과 순서를 확인합니다.

```json
{
  "code": "005930",
  "as_of_date": "2026-10-02",
  "close": 275500,
  "name": "삼성전자",
  "features": {
    "roe_rank": 0.811665,
    "roe_diff_rank": 0.703067,
    "op_margin_rank": 0.854878
  }
}
```

누락 Feature는 학습 pipeline의 median imputer가 처리하지만 전체 31개 중 75% 이상이
전달되어야 합니다. 원시 재무제표·종가를 `/predict`에 바로 넣으면 안 됩니다.
노트북과 같은 point-in-time Feature 생성 로직을 별도 batch/feature 서비스에서 실행한
뒤 API에 전달해야 합니다.

## 6. 버전 호환성

`joblib`은 Python 객체 직렬화이므로 학습 환경과 서빙 환경의 Python,
scikit-learn, xgboost 버전이 다르면 로드 실패 또는 결과 차이가 날 수 있습니다.
`runtime_versions_v24.json`의 버전을 확인해 운영 이미지의 버전을 고정하십시오.

## 7. 운영 권장 구조

```text
[일 1회 Colab/Batch Pipeline]
  └─ DART·가격·Macro 수집 → Feature 생성 → 모델 학습·최신 예측
       └─ versioned artifacts 저장
            └─ API container 재배포/재시작
                 └─ Backend → Frontend
```

실시간 HTTP 요청 안에서 DART 전체 수집과 Feature Store 재생성을 실행하지 않습니다.
수집·학습은 batch로 분리하고 API는 검증된 artifact의 조회·추론만 담당합니다.
