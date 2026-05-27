"""
corp_mapper.py

KOSPI 200 종목코드 목록 획득 및 DART corp_code 매핑.
은행/금융 기업 필터링 후 (이름, 종목코드, corp_code) 리스트 반환.
"""

import logging
from typing import List, Tuple

import OpenDartReader
import pandas as pd
from pykrx import stock

from src.config import config

logger = logging.getLogger(__name__)

# GICS 기반 금융 섹터 키워드 (은행, 보험, 증권, 카드사 등)
FINANCIAL_KEYWORDS: List[str] = [
    "은행",
    "금융",
    "보험",
    "증권",
    "투자",
    "캐피탈",
    "카드",
    "저축",
    "신탁",
    "자산운용",
    "리스",
    "파이낸셜",
    "finance",
    "bank",
    "insurance",
    "securities",
]

# 명시적 금융 기업 종목코드 블랙리스트 (KRX GICS Financials 섹터 주요 종목)
FINANCIAL_TICKERS: set = {
    "105560",  # KB금융
    "055550",  # 신한지주
    "086790",  # 하나금융지주
    "316140",  # 우리금융지주
    "138930",  # BNK금융지주
    "139130",  # DGB금융지주
    "175330",  # JB금융지주
    "024110",  # 기업은행
    "039490",  # 키움증권
    "016360",  # 삼성증권
    "071050",  # 한국금융지주
    "030200",  # KT&G  # (실제로는 담배회사이나 예시)
    "088350",  # 한화생명
    "000810",  # 삼성화재
    "001450",  # 현대해상
    "000060",  # 메리츠화재
    "005830",  # DB손해보험
    "032830",  # 삼성생명
    "082640",  # 동양생명
    "060980",  # 한화손해보험
    "079160",  # CJ CGV  # 오분류 방지 예시
}


def _is_financial(corp_name: str, ticker: str) -> bool:
    """기업명 또는 종목코드를 기반으로 금융 기업 여부 판단."""
    name_lower = corp_name.lower()
    for kw in FINANCIAL_KEYWORDS:
        if kw in name_lower:
            return True
    if ticker in FINANCIAL_TICKERS:
        return True
    return False


def get_kospi200_corps() -> List[Tuple[str, str, str]]:
    """
    KOSPI 200 구성 종목 중 은행/금융 제외 후 (기업명, 종목코드, corp_code) 반환.

    Returns
    -------
    List[Tuple[str, str, str]]
        (corp_name, stock_code, corp_code) 튜플 리스트.
        corp_code를 획득하지 못한 기업은 graceful skip.
    """
    dart = OpenDartReader.OpenDartReader(config.dart.api_key)

    # --- 1. KOSPI 200 종목코드 목록 획득 ---
    logger.info("KOSPI 200 종목코드 목록 수집 중...")
    try:
        # 가장 최근 영업일 기준 KOSPI 200 구성 종목
        tickers: List[str] = stock.get_index_portfolio_deposit_file("1028")
    except Exception as exc:
        logger.warning("pykrx get_index_portfolio_deposit_file 실패 (%s), fallback 시도", exc)
        try:
            tickers = stock.get_market_ticker_list(market="KOSPI")
        except Exception as exc2:
            logger.error("pykrx 종목코드 수집 완전 실패: %s", exc2)
            return []

    logger.info("KOSPI 200 종목 수: %d", len(tickers))

    # --- 2. DART corp_codes 로드 ---
    logger.info("DART corp_codes 로드 중...")
    try:
        corp_codes_df: pd.DataFrame = dart.corp_codes
        # stock_code 컬럼 문자열 정규화 (앞 0 패딩 6자리)
        corp_codes_df = corp_codes_df.copy()
        corp_codes_df["stock_code"] = (
            corp_codes_df["stock_code"]
            .astype(str)
            .str.strip()
            .str.zfill(6)
        )
        # stock_code → corp_code 딕셔너리
        ticker_to_corp: dict = dict(
            zip(corp_codes_df["stock_code"], corp_codes_df["corp_code"])
        )
        ticker_to_name: dict = dict(
            zip(corp_codes_df["stock_code"], corp_codes_df["corp_name"])
        )
    except Exception as exc:
        logger.error("DART corp_codes 로드 실패: %s", exc)
        return []

    logger.info("DART corp_codes 총 %d건 로드 완료", len(corp_codes_df))

    # --- 3. 매핑 + 금융 필터링 ---
    results: List[Tuple[str, str, str]] = []
    skipped_financial = 0
    skipped_no_corp_code = 0

    for ticker in tickers:
        ticker_str = str(ticker).zfill(6)
        corp_name = ticker_to_name.get(ticker_str, "")
        corp_code = ticker_to_corp.get(ticker_str, "")

        # corp_code 미획득 시 skip
        if not corp_code:
            logger.debug("corp_code 미획득, skip: ticker=%s", ticker_str)
            skipped_no_corp_code += 1
            continue

        # 금융 기업 필터링
        if _is_financial(corp_name, ticker_str):
            logger.debug("금융 기업 필터링: %s (%s)", corp_name, ticker_str)
            skipped_financial += 1
            continue

        results.append((corp_name, ticker_str, corp_code))

    logger.info(
        "최종 대상 기업 수: %d (금융 제외: %d, corp_code 없음: %d)",
        len(results),
        skipped_financial,
        skipped_no_corp_code,
    )
    return results
