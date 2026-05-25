### 🛠 다시 복사해서 붙여넣을 코드 (검증됨)

```mermaid
graph TD
    subgraph Data_Collection
        A[DART API: 공시/재무]
        B[FinanceDataReader: 주가/거시지표]
    end

    subgraph Preprocessing
        C{금융권 제외 필터링}
        D[재무/텍스트 벡터화]
    end

    subgraph Modeling
        E[1차: 거시지표 군집화]
        F[2차: 하이브리드 정밀 분류]
        G[AI 테마 라벨링]
        H[주가 예측 모델]
    end

    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H


### 🔍 단계별 상세 설계

1. **Data Collection & Filtering**:
   - DART API를 통해 상위 25개 기업의 재무제표 및 비정형 사업개황 데이터를 추출합니다.
   - 분석의 일관성을 위해 계정 과목 체계가 다른 은행/금융 기업을 데이터셋에서 분리하였습니다.

2. **Multi-Modal Vectorization**:
   - **Numerical Vector**: 재무 수치와 거시 경제 지표를 정규화하여 수치적 특징을 추출합니다.
   - **Textual Vector**: NLP 기술을 사용하여 사업개황 텍스트를 고차원 벡터로 변환, 기업의 실질적인 사업 도메인 유사성을 반영합니다.

3. **Hybrid Clustering (Two-Step)**:
   - 거시 지표로 시장 내 거동이 유사한 그룹을 1차 선별한 뒤, 정적 데이터(재무/텍스트)를 결합하여 테마의 경계선을 정밀하게 조정합니다.
   - 최종적으로 LLM 기반의 라벨링을 통해 각 테마의 정체성을 정의합니다.

4. **Prediction Strategy**:
   - 정적 데이터로 분류된 '테마 경향성'을 사전 지식(Prior Knowledge)으로 활용하여, 신규 공시 발생 시 주가에 미칠 충격과 방향성을 예측합니다.




## 📂 AI 기반 테마 분류 결과 (AI Automated Theme Labeling)

거시 지표와 공시 데이터(비정형 텍스트), 재무 제표를 결합하여 AI가 생성한 5개의 핵심 테마입니다. 각 테마는 유사한 비즈니스 모델과 시장 변동성을 공유하는 기업군으로 구성되었습니다.

| 테마명 (AI Generated) | 구성 종목 (Key Members) | 주요 키워드 (Keywords) |
| :--- | :--- | :--- |
| **✨ GLOBAL CDMO & SET ✨** | 삼성전자, SK하이닉스, 삼성바이오로직스, 셀트리온, NAVER, 카카오 등 | CDMO, SET, USA, MATERIAL |
| **✨ GLOBAL LG & INFRASTRUCTURE ✨** | LG에너지솔루션, 삼성SDI, LG화학, SK이노베이션, HD현대중공업 | LG, SK, ESS, ISO |
| **✨ GLOBAL EV & LCD ✨** | 현대자동차, 기아, POSCO홀딩스, 현대모비스, LG전자, LG디스플레이 | EV, LCD, OLED, STEEL |
| **✨ GLOBAL EPC & ENGINEERING ✨** | 한화오션, 한국전력, 두산에너빌리티 | EPC, ENGINEERING, SMR, CONSTRUCTION |
| **✨ SPECIALIZED SECTOR ✨** | 한화에어로스페이스 | UTILITY |

---

## 📊 데이터 시각화 분석 (Data Visualization)

DART 공시 데이터와 재무 데이터를 벡터화하여 각 테마별 군집 정도를 시각화한 결과입니다. 거시적 지표로 분류되지 않던 기업 간의 관계를 텍스트 및 재무 데이터를 통해 정교하게 재분류하였습니다.

<p align="center">
  <img width="48%" alt="Theme Visualization 1" src="https://github.com/user-attachments/assets/40073939-8425-4329-99ba-b5367a988036" />
  <img width="48%" alt="Theme Visualization 2" src="https://github.com/user-attachments/assets/46a34af1-a169-47ae-988d-e4108b9833c7" />
</p>
<p align="center">
  <img width="48%" alt="Theme Visualization 3" src="https://github.com/user-attachments/assets/7efc2394-a2e8-4f50-b110-dd315cb5fe5f" />
  <img width="48%" alt="Theme Visualization 4" src="https://github.com/user-attachments/assets/f9460550-f0c5-4a44-8e70-8e818a08a0cc" />
</p>

> **💡 시각화 인사이트**
> 1. **하이브리드 데이터 활용:** 거시 지표(주가 변동)와 정적 데이터(공시 텍스트/재무)를 결합하여 기업의 본질적인 비즈니스 유사성을 포착했습니다.
> 2. **군집의 명확성:** 텍스트 벡터 주입 후, 단순 산업 분류를 넘어선 테마별 경향성이 뚜렷하게 나타남을 확인하였습니다.
> 3. **향후 활용:** 본 분류 모델은 실시간 공시 데이터 발생 시 테마별 변동성을 반영한 주가 예측 모델의 기반으로 활용됩니다.
