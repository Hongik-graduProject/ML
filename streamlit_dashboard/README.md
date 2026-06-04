# KOSPI200 AI 재무·주가 분석 대시보드 - Safe Version

이 버전은 앱 시작 시 무거운 XGBoost 학습을 바로 실행하지 않고, CSV 로드 후 사용자가 버튼을 눌렀을 때 모델을 학습합니다.

## 실행

```bash
cd ~/Downloads/kospi200_ai_dashboard_safe
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py --browser.gatherUsageStats false
```
python 명령어가 안된다면 이걸로 하고 activate 부터는 동일
```bash
/opt/homebrew/bin/python3.12 -m venv .venv
```
CSV 파일은 다음 위치에 넣으세요.

```text
data/kospi200_train_dataset_2014_2025.csv
```
