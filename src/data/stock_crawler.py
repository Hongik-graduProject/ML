"""
stock_crawler.py

KRX 주가 데이터(월별 수익률) 및 FRED 거시경제 지표 수집.
"""

import logging
from typing import Optional

import FinanceDataReader as fdr
import pandas as pd
import pandas_datareader.data as web

from src.config import config

logger = logging.getLogger(__name__)

# FRED 시리즈 코드
FRED_SERIES = {
    "interest_rate": "DFII10",   # 미국 10년 실질국채금리 (TIPS)
    "fx_rate": "DEXKOUS",        # 원/달러 환율
    "oil_price": "DCOILWTICO",   # WTI 유가
}


def fetch_monthly_stock_returns(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    KRX 주가 데이터를 수집하고 월별 수익률을 산출한다.

    Parameters
    ----------
    ticker : str
        KRX 종목코드 (예: "005930")
    start : str
        시작일 (예: "2015-01-01")
    end : str
        종료일 (예: "2024-12-31")

    Returns
    -------
    pd.DataFrame
        인덱스: 월별 DatetimeIndex (월말 기준)
        컬럼: ['return_pct']
        빈 DataFrame 반환 시 수집 실패를 의미.
    """
    try:
        df = fdr.DataReader(ticker, start=start, end=end)
        if df is None or df.empty:
            logger.warning("주가 데이터 없음: ticker=%s", ticker)
            return pd.DataFrame(columns=["return_pct"])

        # 종가 컬럼 탐색 ('Close' 또는 'close')
        close_col: Optional[str] = None
        for col in ["Close", "close", "Adj Close", "adj close"]:
            if col in df.columns:
                close_col = col
                break

        if close_col is None:
            logger.warning("종가 컬럼 미발견: ticker=%s, 컬럼=%s", ticker, df.columns.tolist())
            return pd.DataFrame(columns=["return_pct"])

        close = df[close_col].dropna()

        # 월별 마지막 거래일 종가로 리샘플링
        monthly_close = close.resample("ME").last().dropna()

        # 월별 수익률 (%)
        monthly_return = monthly_close.pct_change().dropna() * 100
        monthly_return.name = "return_pct"

        result = monthly_return.to_frame()
        result.index.name = "date"
        logger.debug(
            "월별 수익률 산출 완료: ticker=%s, 기간=%s~%s, 행 수=%d",
            ticker,
            start,
            end,
            len(result),
        )
        return result

    except Exception as exc:
        logger.warning("fetch_monthly_stock_returns 실패: ticker=%s, 오류=%s", ticker, exc)
        return pd.DataFrame(columns=["return_pct"])


def fetch_macro_indicators(
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    FRED에서 거시경제 지표를 수집하고 월별 변동률로 변환한다.

    수집 지표:
      - interest_rate : 미국 10년 실질국채금리 (DFII10)
      - fx_rate       : 원/달러 환율 (DEXKOUS)
      - oil_price     : WTI 유가 (DCOILWTICO)

    Parameters
    ----------
    start : str
        시작일 (예: "2015-01-01")
    end : str
        종료일 (예: "2024-12-31")

    Returns
    -------
    pd.DataFrame
        인덱스: 월별 DatetimeIndex (월말 기준)
        컬럼: ['interest_rate', 'fx_rate', 'oil_price']
        각 값은 월별 변동률(%).
        수집 실패한 지표는 NaN으로 채워짐.
    """
    frames = {}

    for indicator_name, series_code in FRED_SERIES.items():
        try:
            raw = web.DataReader(series_code, "fred", start=start, end=end)
            if raw is None or raw.empty:
                logger.warning("FRED 데이터 없음: series=%s", series_code)
                frames[indicator_name] = pd.Series(dtype=float, name=indicator_name)
                continue

            series = raw.iloc[:, 0].dropna()

            # 월별 평균으로 리샘플링
            monthly = series.resample("ME").mean().dropna()

            # 월별 변동률 (%)
            monthly_change = monthly.pct_change().dropna() * 100
            monthly_change.name = indicator_name
            frames[indicator_name] = monthly_change

            logger.debug(
                "FRED 지표 수집 완료: %s (%s), 행 수=%d",
                indicator_name,
                series_code,
                len(monthly_change),
            )

        except Exception as exc:
            logger.warning(
                "FRED 지표 수집 실패: %s (%s), 오류=%s", indicator_name, series_code, exc
            )
            frames[indicator_name] = pd.Series(dtype=float, name=indicator_name)

    if not frames:
        return pd.DataFrame(columns=list(FRED_SERIES.keys()))

    result = pd.concat(frames.values(), axis=1)
    result.index.name = "date"

    logger.info(
        "거시경제 지표 수집 완료: 기간=%s~%s, 행 수=%d, 컬럼=%s",
        start,
        end,
        len(result),
        result.columns.tolist(),
    )
    return result
