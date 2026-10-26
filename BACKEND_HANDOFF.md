# 백엔드 개발자 인수인계서

## 1. 서비스 목적

이 API는 향후 60거래일의 **KOSPI 대비 종목별 초과수익률**을 제공합니다.
절대 주가 상승확률 API가 아니며, 동일 기준일의 종목 간 상대순위를 주된 용도로 합니다.

## 2. 백엔드에서 우선 연동할 Endpoint

초기 서비스에서는 Feature를 백엔드가 직접 만들지 말고 아래 조회 API를 우선 사용합니다.

1. `GET /api/v1/predictions/latest` - 종목 목록 화면
2. `GET /api/v1/predictions/{code}` - 종목 상세 화면
3. `GET /api/v1/model/info` - 모델 기준일·버전 표시
4. `GET /health/ready` - 배포 readiness probe

`POST /predict`와 `/predict/batch`는 노트북과 동일한 31개 Feature를 생성할 수 있는
batch pipeline이 서버에 구축된 이후 사용합니다.

## 3. 필드 의미

| API 필드 | 의미 | UI 표시 권장 |
|---|---|---|
| `predicted_excess_return_60d` | 60거래일 KOSPI 대비 예상 초과수익률 | 퍼센트 |
| `predicted_percentile` | 동일 기준일 예측 대상 내 상대순위 | 상위 n% |
| `signal` | percentile 구간 | Top 10%, Above Average 등 |
| `model_agreement_score` | Ridge/XGBoost 예측 일치도 순위 | 모델 일치도 |
| `scenario_price_if_kospi_flat` | KOSPI 수익률 0% 가정 가격 | 참고 시나리오 가격 |
| `feature_coverage` | 31개 Feature 중 유효값 비율 | 데이터 충족률 |

`model_agreement_score`를 신뢰확률 또는 적중확률로 표기하지 마십시오.

## 4. HTTP 상태코드

| 코드 | 상황 |
|---|---|
| 200 | 정상 |
| 401 | API key 누락·불일치 |
| 404 | 해당 종목 예측 없음 |
| 413 | batch 최대 크기 초과 |
| 422 | 요청 스키마·Feature coverage 오류 |
| 503 | 모델 또는 예측 artifact 로드 실패 |

## 5. 배포 전 확인

- `artifacts/`에 필수 파일 4개가 있는지 확인
- Colab `runtime_versions_v24.json`에 맞춰 ML 라이브러리 버전 고정
- `python scripts/validate_artifacts.py` 성공 확인
- `pytest -q` 통과 확인
- `/health/ready`가 HTTP 200인지 확인
- 운영 API key와 CORS origin 설정
- 모델·CSV 업데이트 시 API container 재시작

## 6. Artifact 갱신 계약

파일을 하나씩 덮어쓰면 모델과 예측의 버전이 잠시 어긋날 수 있습니다. 새 artifact를
별도 version directory에 모두 업로드하고 검증한 뒤, 배포 설정의 artifact 경로를
새 directory로 원자적으로 전환하는 방식을 권장합니다.

예시:

```text
/models/v2.4.20261002/
/models/v2.4.20261101/
/models/current -> /models/v2.4.20261101/
```

## 7. 제한사항

- 현재 최신 예측은 노트북 실행 시점의 snapshot입니다.
- 현재 Universe를 과거로 확장했으므로 과거 검증에 생존편향 가능성이 있습니다.
- 연간 재무정보 기반이며 분기 TTM Feature는 아직 포함하지 않았습니다.
- API는 투자 권유나 수익 보장을 제공하지 않습니다.
