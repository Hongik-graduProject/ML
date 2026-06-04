
from __future__ import annotations

from pathlib import Path
import traceback

import pandas as pd
import plotly.express as px
import streamlit as st

from model_utils import (
    FEATURES,
    build_financial_summary,
    build_stock_history,
    feature_contribution_table,
    get_available_train_years,
    load_dataset,
    predict_company,
    prepare_model_frames,
    train_model,
)

st.set_page_config(page_title="KOSPI200 AI 분석 대시보드", page_icon="📈", layout="wide")

DATA_PATH = Path("data/kospi200_train_dataset_2014_2025.csv")

st.title("📈 KOSPI200 AI 재무·주가 분석 대시보드")
st.caption("DART 재무정보, KRX/주가 데이터, 매크로 민감도, AI 예측 모델을 연결한 Streamlit UI")

with st.sidebar:
    st.header("⚙️ 설정")
    st.caption("기본 CSV 경로")
    st.code(str(DATA_PATH), language="text")
    uploaded = st.file_uploader("다른 CSV 업로드", type=["csv"])

@st.cache_data(show_spinner="CSV 데이터를 불러오는 중...")
def cached_load_default(path: str):
    return load_dataset(path)

@st.cache_data(show_spinner="업로드 CSV를 불러오는 중...")
def cached_load_uploaded(file_bytes: bytes):
    tmp = Path(".uploaded_dataset.csv")
    tmp.write_bytes(file_bytes)
    return load_dataset(tmp)

@st.cache_data(show_spinner="모델용 데이터 전처리 중...")
def cached_prepare(path_marker: str, uploaded_bytes: bytes | None = None):
    if uploaded_bytes is None:
        raw = cached_load_default(path_marker)
    else:
        raw = cached_load_uploaded(uploaded_bytes)
    inference, train = prepare_model_frames(raw)
    return raw, inference, train

try:
    if uploaded is None:
        raw_df, inference_df, train_df = cached_prepare(str(DATA_PATH), None)
    else:
        raw_df, inference_df, train_df = cached_prepare(uploaded.name, uploaded.getvalue())
except Exception as e:
    st.error("CSV를 불러오거나 전처리하는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

companies = sorted(inference_df["Name"].dropna().unique())
train_year_options = get_available_train_years(train_df)

with st.sidebar:
    st.success(f"데이터 로드 완료: {len(raw_df):,}행")
    st.write(f"학습 가능 데이터: **{len(train_df):,}행**")
    selected_company = st.selectbox("기업 선택", companies, index=companies.index("삼성전자") if "삼성전자" in companies else 0)
    years = sorted(inference_df[inference_df["Name"] == selected_company]["Year"].dropna().astype(int).unique())
    selected_year = st.selectbox("분석 기준 연도", years, index=len(years)-1)

    default_train_years = train_year_options[-3:] if len(train_year_options) >= 3 else train_year_options
    selected_train_years = st.multiselect("모델 학습 연도", train_year_options, default=default_train_years)
    train_clicked = st.button("AI 모델 학습/새로고침", type="primary")

company_year_df = inference_df[(inference_df["Name"] == selected_company) & (inference_df["Year"] == selected_year)].sort_values("Date")
if company_year_df.empty:
    st.warning("선택한 기업/연도에 사용할 수 있는 데이터가 없습니다.")
    st.stop()

as_of_date = st.sidebar.date_input(
    "분석 기준일",
    value=company_year_df["Date"].max().date(),
    min_value=company_year_df["Date"].min().date(),
    max_value=company_year_df["Date"].max().date(),
)

# 앱 시작 즉시 무거운 학습을 하지 않고, 사용자가 버튼을 눌렀을 때만 학습합니다.
if "model" not in st.session_state or train_clicked:
    with st.spinner("AI 모델 학습 중입니다. 처음 실행 시 시간이 조금 걸릴 수 있습니다..."):
        try:
            model, used_years, used_rows = train_model(train_df, selected_train_years)
            st.session_state["model"] = model
            st.session_state["used_years"] = used_years
            st.session_state["used_rows"] = used_rows
        except Exception as e:
            st.error("모델 학습 중 오류가 발생했습니다.")
            st.exception(e)
            st.stop()

model = st.session_state["model"]
used_years = st.session_state["used_years"]
used_rows = st.session_state["used_rows"]

st.info(f"현재 모델: {getattr(model, '_model_name_for_ui', type(model).__name__)} / 학습 연도: {', '.join(map(str, used_years))} / 학습 행 수: {used_rows:,}")

try:
    result = predict_company(model, inference_df, selected_company, selected_year, str(as_of_date))
except Exception as e:
    st.error("예측 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("기준일", result.date.strftime("%Y-%m-%d"))
col2.metric("현재가", f"{result.current_price:,.0f}원")
col3.metric("AI 예측 수익률", f"{result.predicted_return * 100:+.2f}%")
col4.metric("AI 목표 주가", f"{result.predicted_price:,.0f}원")

if result.actual_return is not None:
    c1, c2, c3 = st.columns(3)
    c1.metric("실제 60일 뒤 수익률", f"{result.actual_return * 100:+.2f}%")
    c2.metric("실제 60일 뒤 주가", f"{result.actual_price:,.0f}원")
    c3.metric("예측 오차", f"{result.error_pctp:.2f}%p")
else:
    st.info("선택 기준일 이후 60일 실제 수익률은 데이터셋에 없거나 계산할 수 없습니다.")

st.divider()
left, right = st.columns([1.45, 1])
with left:
    st.subheader(f"{selected_company} 주가 추이")
    history = build_stock_history(raw_df, selected_company)

    fig = px.line(
        history,
        x="Date",
        y="Close",
        title=f"{selected_company} Close Price"
    )

    기준일 = result.date.strftime("%Y-%m-%d") if hasattr(result.date, "strftime") else str(result.date)

    fig.add_vline(
        x=기준일,
        line_dash="dash"
    )

    if not history.empty and "Close" in history.columns:
        y_pos = history["Close"].max()
    else:
        y_pos = result.current_price

    fig.add_annotation(
        x=기준일,
        y=y_pos,
        text="기준일",
        showarrow=False,
        yshift=10
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("모델 입력값")
    sample = result.sample.iloc[0]
    show_cols = [c for c in ["roe", "roe_diff", "op_margin", "op_margin_diff", "debt_ratio", "debt_ratio_diff", "op_income", "op_income_diff"] if c in sample.index]
    st.dataframe(pd.DataFrame({"Feature": show_cols, "Value": [sample[c] for c in show_cols]}), use_container_width=True, hide_index=True)
    st.subheader("피처 중요도")
    importance = feature_contribution_table(model).head(10)
    st.bar_chart(importance.set_index("Feature"))

st.divider()
st.subheader("재무제표 주요 지표")
financial_summary = build_financial_summary(raw_df, selected_company)
if financial_summary.empty:
    st.warning("재무제표 컬럼을 찾지 못했습니다.")
else:
    st.dataframe(financial_summary, use_container_width=True, hide_index=True)

with st.expander("디버그 정보 보기"):
    st.write("컬럼 목록", list(raw_df.columns))
    st.write("학습 데이터 행 수", len(train_df))
    st.write("선택 기준일 입력 데이터")
    st.dataframe(result.sample[[c for c in ["Date", "Code", "Name", "Year", "Close"] + FEATURES if c in result.sample.columns]], use_container_width=True)

st.caption("주의: 본 대시보드는 수업 과제용/분석 실험용 예시이며, 투자 권유나 실제 매매 판단 자료가 아닙니다.")
