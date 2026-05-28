🏗️ Model Pipeline: From Raw Data to Prediction
1단계: 데이터 수집 및 전처리 (Data Ingestion & Preprocessing)
서로 다른 소스에서 온 6개의 데이터셋을 하나로 통합하는 단계입니다.

금융 지표: KRX 금리, 환율, 유가, 금 시세 데이터를 수집합니다.

재무 DNA: KOSPI 200 기업들의 ROE, 영업이익, 부채비율 등 재무제표 데이터를 추출합니다.

정렬: 모든 데이터는 '날짜(연도)'와 '종목코드(Code)'를 기준으로 병합(Merge)되며, 결측치는 모델의 안정성을 위해 전처리됩니다.

2단계: 피처 엔지니어링 (Feature Engineering)
모델이 시장의 '변화'를 읽을 수 있도록 원천 데이터를 가공합니다.

상태 지표 (State): 현재의 ROE, 영업이익률 등 절대 수치.

변화 지표 (Momentum): roe_diff, op_income_diff 등 전년 대비 증감폭을 계산하여 실적 개선세를 포착합니다.

민감도 지표 (Sensitivity): 각 종목/테마가 거시 경제 변수(금리, 환율 등)에 얼마나 민감하게 반응하는지 상관계수(Correlation) 피처를 생성합니다.

3단계: 슬라이딩 윈도우 기반 학습 (Sliding Window Training)
주식 시장의 시시각각 변하는 특성을 반영하기 위해 고정된 학습이 아닌 '롤링(Rolling)' 방식을 사용합니다.

학습 창(Train Window): 과거 3개년 데이터를 묶어 모델을 학습시킵니다.

예측 창(Test Window): 학습 직후의 1개년 데이터를 대상으로 모델의 예측력을 검증합니다.

반복: 윈도우를 1년씩 뒤로 밀며 이 과정을 반복하여, 모델이 '코로나19 팬데믹'이나 '고금리 시대' 같은 특수 국면을 스스로 학습하게 합니다.

4단계: XGBoost 모델 추론 (Inference via XGBoost)
가공된 피처들을 XGBoost 알고리즘에 투입합니다.

비선형 관계 파악: 금리 인상이 어떤 기업에는 독이 되고 어떤 기업에는 약이 되는 복잡한 비선형적 관계를 트리 기반 알고리즘이 찾아냅니다.

중요도 산출: 매크로 변수와 재무 변수 중 현재 주가에 가장 큰 영향을 미치는 '핵심 동인(Key Driver)'을 매 연도별로 추출합니다.

5단계: 성과 평가 및 종목 추출 (Evaluation & Selection)
최종적으로 학습된 모델을 통해 실전 의사결정을 내립니다.

수익률 예측: 3개월(60일) 뒤의 예상 누적 수익률을 종목별로 산출합니다.

오차 분석: 실제 수익률과의 차이(RMSE)를 분석하여 모델의 신뢰도를 평가합니다.

Top Pick 추출: 예측 수익률이 가장 높은 상위 종목 리스트를 생성하여 투자 전략에 활용합니다.

**🚀 Phase 2: 매크로 유전자 기반 테마 분류 엔진 (2014-2023)이 단계에서는 단순히 업종별(IT, 금융 등)로 기업을 나누는 것이 아니라, 거시경제 지표(금리, 환율, 유가, 금)에 반응하는 기업별 '매크로 유전자'를 추출하여 8개의 동적 테마로 분류합니다.**


1. 분석 목적매크로 민감도 수치화: 각 기업의 주가가 거시경제 변수에 얼마나 민감하게 반응하는지 상관계수로 산출합니다.동적 테마 구축: 고정된 섹터가 아닌, 매년 변화하는 경제 환경 속에서 실제로 같이 움직이는 기업군을 포착합니다.3차 예측 가중치 생성: 향후 AI 주가 예측 모델이 참고할 지표별 중요도(Feature Weight)를 DB화합니다
우리가 사용하는 코드(price.corr(macro_df))는 내부적으로 피어슨 상관계수 공식을 사용합니다.$$r = \frac{\sum (X - \bar{X})(Y - \bar{Y})}{\sqrt{\sum (X - \bar{X})^2 \sum (Y - \bar{Y})^2}}$$

2.   핵심 알고리즘 및 프로세스데이터 정밀화 (Weekly Data): 월간 데이터의 노이즈를 줄이기 위해 주간 수익률 단위로 분석하여 1년당 약 52개의 데이터 포인트를 확보합니다.유전자 추출 (Sensitivity Analysis): 4대 매크로 지표($Interest, FX, Oil, Gold$)와 개별 종목 주가 간의 상관계수를 계산하여 기업별 '민감도 벡터'를 생성합니다.군집화 (K-Means Clustering): 생성된 민감도 벡터를 바탕으로 거시경제 반응 체계가 가장 유사한 기업들을 8개 테마로 묶습니다.표준 가중치 확정 (Intelligence): 각 테마에 속한 기업들의 평균 수치를 계산하여, 해당 테마의 '대표 매크로 유전자'를 확정합니다
3.   **생성되는 결과물 (3차 파이프라인 입력값)company_yearly_sensitivities.csv: 기업별 연도별 매크로 반응도 (Raw Data)theme_yearly_weights.csv: 테마별 표준 가중치 (AI 모델의 직접적 가중치로 활용)company_theme_mapping.csv: 기업별 연도별 테마 소속 기록**

<img width="572" height="246" alt="Image" src="https://github.com/user-attachments/assets/3592fade-abae-4a29-aa5d-9444c1c07a56" />

<img width="588" height="411" alt="Image" src="https://github.com/user-attachments/assets/a461e0dc-3602-4710-af64-b08fc3abbe9e" />

<img width="589" height="279" alt="Image" src="https://github.com/user-attachments/assets/a82dd566-8e05-4ce9-b914-d77c5bdbee4e" />

<img width="598" height="403" alt="Image" src="https://github.com/user-attachments/assets/c2f64ab9-7ddc-4abf-8e9f-aa4b60e3a0db" />


XGBoost (Extreme Gradient Boosting)

Training Strategy: Sliding Window 과거의 데이터가 현재의 시장 질서를 대변하지 못하는 주식 시장의 특성을 반영하여 '슬라이딩 윈도우' 방식을 채택했습니다.

Window Structure: 3-Year Training → 1-Year Prediction

Learning Process: 1. 2014-2016년 데이터를 학습하여 2017년을 예측 및 검증합니다. 2. 윈도우를 1년 뒤로 밀어 2015-2017년 데이터를 학습하고 2018년을 예측합니다. 3. 이 과정을 반복하여 모델이 최신 시장 트렌드(예: 고금리 국면, 팬데믹 등)를 지속적으로 업데이트하도록 설계되었습니다.

<img width="596" height="504" alt="Image" src="https://github.com/user-attachments/assets/c3c592fd-cf98-4222-844f-f9b4f03c2a9c" />

<img width="524" height="327" alt="Image" src="https://github.com/user-attachments/assets/70f7a3f8-e401-47b3-994d-cbf5634b2479" />
