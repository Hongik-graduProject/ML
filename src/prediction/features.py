"""예측 모델용 피처 엔지니어링."""
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import config

logger = logging.getLogger(__name__)


def build_prediction_features(
    panel_df: pd.DataFrame,
    theme_labels: Dict[str, int],
    stock_returns: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """예측 모델에 사용할 피처 데이터프레임을 구성한다.

    피처 목록:
    - theme_* : 테마 one-hot 인코딩
    - theme_avg_return : 테마 내 다른 기업들의 최근 N일 평균 수익률
    - macro_interest_chg : 금리 당일 변동
    - macro_fx_chg      : 환율 당일 변동
    - macro_oil_chg     : 유가 당일 변동
    - momentum          : 해당 기업의 최근 N일 수익률

    타깃:
    - next_month_return : 다음 월 수익률

    Args:
        panel_df: 패널 데이터프레임.
            필수 컬럼: ['corp_name', 'date']
        theme_labels: {corp_name: cluster_id} 매핑
        stock_returns: 일별 주가 수익률 데이터프레임.
            인덱스: DatetimeIndex, 컬럼: corp_name
        macro_df: 거시지표 데이터프레임.
            인덱스: DatetimeIndex,
            컬럼: ['interest_rate', 'fx_rate', 'oil_price']

    Returns:
        피처 + 타깃 컬럼이 포함된 데이터프레임.
        타깃 컬럼: 'next_month_return'
    """
    momentum_window = config.prediction.momentum_window_days

    # --- 테마 레이블 매핑 ---
    panel_df = panel_df.copy()
    panel_df["theme_id"] = panel_df["corp_name"].map(theme_labels)

    # 테마 one-hot 인코딩
    theme_dummies = pd.get_dummies(
        panel_df["theme_id"], prefix="theme", dtype=float
    )
    panel_df = pd.concat([panel_df, theme_dummies], axis=1)

    # --- 거시지표 당일 변동 ---
    macro_chg = macro_df.diff()
    macro_chg.columns = [
        "macro_interest_chg",
        "macro_fx_chg",
        "macro_oil_chg",
    ]

    # --- 행별 피처 계산 ---
    records: List[dict] = []

    for _, row in panel_df.iterrows():
        corp = row["corp_name"]
        date = pd.Timestamp(row["date"])
        theme_id = row.get("theme_id")

        # 해당 기업의 수익률 시계열
        if corp not in stock_returns.columns:
            logger.warning("%s 주가 데이터 없음, 스킵", corp)
            continue

        corp_ret = stock_returns[corp].dropna()
        past_ret = corp_ret.loc[:date]

        # momentum: 최근 N일 수익률 합산
        if len(past_ret) >= momentum_window:
            momentum = past_ret.iloc[-momentum_window:].sum()
        elif len(past_ret) > 0:
            momentum = past_ret.sum()
        else:
            momentum = 0.0

        # theme_avg_return: 테마 내 다른 기업의 평균 수익률
        peers = [
            c for c, tid in theme_labels.items()
            if tid == theme_id and c != corp and c in stock_returns.columns
        ]
        if peers:
            peer_rets = stock_returns[peers].dropna(how="all")
            past_peer = peer_rets.loc[:date]
            if len(past_peer) >= momentum_window:
                theme_avg_return = (
                    past_peer.iloc[-momentum_window:].mean().mean()
                )
            elif len(past_peer) > 0:
                theme_avg_return = past_peer.mean().mean()
            else:
                theme_avg_return = 0.0
        else:
            theme_avg_return = 0.0

        # 거시지표 변동
        macro_row = (
            macro_chg.loc[date] if date in macro_chg.index
            else pd.Series({"macro_interest_chg": 0.0,
                            "macro_fx_chg": 0.0,
                            "macro_oil_chg": 0.0})
        )

        # next_month_return: 다음 월 수익률 (타깃)
        next_month_start = date + pd.offsets.MonthBegin(1)
        next_month_end = date + pd.offsets.MonthEnd(1)
        future_ret = corp_ret.loc[next_month_start:next_month_end]
        next_month_return = future_ret.sum() if len(future_ret) > 0 else np.nan

        # theme one-hot 컬럼 수집
        theme_cols = {
            c: row[c]
            for c in row.index
            if str(c).startswith("theme_") and c != "theme_id"
        }

        record = {
            "corp_name": corp,
            "date": date,
            "momentum": momentum,
            "theme_avg_return": theme_avg_return,
            "macro_interest_chg": macro_row.get("macro_interest_chg", 0.0),
            "macro_fx_chg": macro_row.get("macro_fx_chg", 0.0),
            "macro_oil_chg": macro_row.get("macro_oil_chg", 0.0),
            "next_month_return": next_month_return,
            **theme_cols,
        }
        records.append(record)

    result_df = pd.DataFrame(records)
    logger.info(
        "피처 빌드 완료: %d 행, %d 컬럼", len(result_df), len(result_df.columns)
    )
    return result_df
