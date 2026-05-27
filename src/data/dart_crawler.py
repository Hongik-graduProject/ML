"""
dart_crawler.py

OpenDartReader를 사용하여 연도별 사업보고서(11011) 사업의 개요 텍스트 및
재무제표 데이터를 크롤링한다.
"""

import logging
import time
from typing import Any, Optional

import OpenDartReader
from bs4 import BeautifulSoup

from src.config import config

logger = logging.getLogger(__name__)

# 사업보고서에서 탐색할 섹션 이름 후보 (우선순위 순)
BUSINESS_SECTION_CANDIDATES = [
    "사업의 내용",
    "사업의 개요",
    "사업 내용",
    "사업개요",
    "영업의 내용",
]

# 재무제표 계정명 매핑 (DART 보고서 내 다양한 표기 통합)
ACCOUNT_MAP = {
    "매출액": ["매출액", "영업수익", "수익(매출액)", "매출", "순매출액"],
    "영업이익": ["영업이익", "영업이익(손실)", "영업손익"],
    "당기순이익": [
        "당기순이익",
        "당기순이익(손실)",
        "당기순손익",
        "연결당기순이익",
        "당기순이익(지배)",
    ],
    "자산총계": ["자산총계", "자산 총계", "총자산"],
    "자본총계": ["자본총계", "자본 총계", "총자본", "자본합계"],
    "부채총계": ["부채총계", "부채 총계", "총부채", "부채합계"],
}


def _safe_sleep() -> None:
    """DART API 속도 제한 대응 sleep."""
    time.sleep(config.dart.request_interval)


def _extract_text_from_html(html: str) -> str:
    """HTML 원문에서 가시 텍스트만 추출한다."""
    soup = BeautifulSoup(html, "html.parser")
    # script, style 태그 제거
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # 연속 공백 정리
    import re
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _find_section_dcm_no(sub_docs: Any, section_candidates: list) -> Optional[str]:
    """
    sub_docs DataFrame에서 후보 섹션명과 일치하는 dcm_no를 반환한다.
    title 컬럼을 기준으로 부분 문자열 매칭.
    """
    if sub_docs is None or len(sub_docs) == 0:
        return None

    for candidate in section_candidates:
        mask = sub_docs["title"].str.contains(candidate, na=False)
        matched = sub_docs[mask]
        if not matched.empty:
            return str(matched.iloc[0]["dcm_no"])
    return None


def _normalize_account_value(value_str: str) -> Optional[float]:
    """DART 재무 금액 문자열을 float으로 변환 (단위: 원)."""
    if not value_str or value_str in ("-", "", "nan"):
        return None
    try:
        cleaned = str(value_str).replace(",", "").replace(" ", "").strip()
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _extract_account(finstate_df: Any, account_name: str) -> Optional[float]:
    """
    재무제표 DataFrame에서 특정 계정의 값을 추출한다.
    account_nm 또는 account_id 컬럼 기준으로 탐색.
    """
    if finstate_df is None or len(finstate_df) == 0:
        return None

    candidates = ACCOUNT_MAP.get(account_name, [account_name])
    for candidate in candidates:
        mask = finstate_df["account_nm"].str.contains(candidate, na=False)
        matched = finstate_df[mask]
        if not matched.empty:
            row = matched.iloc[0]
            # thstrm_amount: 당기, frmtrm_amount: 전기
            for col in ["thstrm_amount", "thstrm_amt"]:
                if col in row.index:
                    val = _normalize_account_value(str(row[col]))
                    if val is not None:
                        return val
    return None


def fetch_business_overview(
    dart: OpenDartReader.OpenDartReader,
    corp_code: str,
    year: int,
) -> str:
    """
    DART 연도별 사업보고서에서 "사업의 개요" 섹션 텍스트를 추출한다.

    Parameters
    ----------
    dart : OpenDartReader
    corp_code : str
        DART 기업 고유 코드
    year : int
        사업연도 (예: 2023)

    Returns
    -------
    str
        사업의 개요 텍스트. 실패 시 빈 문자열 반환.
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    try:
        # 1. 사업보고서(11011) 접수번호 획득
        _safe_sleep()
        report_list = dart.list(
            corp_code,
            start=start_date,
            end=end_date,
            kind="A",
            final=True,
        )

        if report_list is None or len(report_list) == 0:
            logger.debug("사업보고서 없음: corp_code=%s, year=%d", corp_code, year)
            return ""

        # 사업보고서(11011)만 필터
        annual_reports = report_list[
            report_list["report_nm"].str.contains("사업보고서", na=False)
        ]
        if annual_reports.empty:
            annual_reports = report_list

        rcept_no = str(annual_reports.iloc[0]["rcept_no"])
        logger.debug("rcept_no 획득: %s (corp_code=%s, year=%d)", rcept_no, corp_code, year)

        # 2. 목차 조회
        _safe_sleep()
        sub_docs = dart.sub_docs(rcept_no)

        if sub_docs is None or len(sub_docs) == 0:
            logger.debug("sub_docs 없음: rcept_no=%s", rcept_no)
            return ""

        # 3. 사업의 개요 섹션 dcm_no 추출
        dcm_no = _find_section_dcm_no(sub_docs, BUSINESS_SECTION_CANDIDATES)
        if dcm_no is None:
            logger.debug(
                "사업의 개요 섹션 미발견: rcept_no=%s, 가용 섹션=%s",
                rcept_no,
                sub_docs["title"].tolist()[:10],
            )
            return ""

        # 4. HTML 원문 획득 및 텍스트 추출
        _safe_sleep()
        html_content = dart.document(rcept_no, dcm_no)
        if not html_content:
            logger.debug("HTML 원문 없음: dcm_no=%s", dcm_no)
            return ""

        text = _extract_text_from_html(html_content)
        logger.debug(
            "텍스트 추출 성공: corp_code=%s, year=%d, 길이=%d", corp_code, year, len(text)
        )
        return text

    except Exception as exc:
        logger.warning(
            "fetch_business_overview 실패: corp_code=%s, year=%d, 오류=%s",
            corp_code,
            year,
            exc,
        )
        return ""


def fetch_business_overview_with_fallback(
    dart: OpenDartReader.OpenDartReader,
    corp_code: str,
    year: int,
    corp_name: str = "",
    prev_text: str = "",
) -> str:
    """
    fetch_business_overview 래퍼. 실패 시 prev_text 캐리오버, 최종 fallback은 기업명 기본 문구.

    Parameters
    ----------
    dart : OpenDartReader
    corp_code : str
    year : int
    corp_name : str
        Fallback 문구 생성에 사용할 기업명
    prev_text : str
        이전 연도 텍스트 (캐리오버용)

    Returns
    -------
    str
    """
    text = fetch_business_overview(dart, corp_code, year)
    if text:
        return text

    if prev_text:
        logger.info(
            "텍스트 추출 실패, 이전 연도 캐리오버: corp_code=%s, year=%d", corp_code, year
        )
        return prev_text

    fallback = f"{corp_name}은(는) 다양한 사업 영역에서 활동하는 기업입니다." if corp_name else "사업 정보를 불러오지 못했습니다."
    logger.info(
        "텍스트 추출 실패, 기본 문구 사용: corp_code=%s, year=%d", corp_code, year
    )
    return fallback


def fetch_financials(
    dart: OpenDartReader.OpenDartReader,
    corp_code: str,
    year: int,
) -> dict:
    """
    DART 연도별 재무제표에서 주요 재무 항목을 추출한다.

    Parameters
    ----------
    dart : OpenDartReader
    corp_code : str
    year : int

    Returns
    -------
    dict
        키: 매출액, 영업이익, 당기순이익, 자산총계, 자본총계, 부채총계
        값: float 또는 None (추출 실패 시)
    """
    result: dict = {
        "매출액": None,
        "영업이익": None,
        "당기순이익": None,
        "자산총계": None,
        "자본총계": None,
        "부채총계": None,
    }

    try:
        _safe_sleep()
        # 연결재무제표(CFS) 우선, 없으면 별도재무제표(OFS)
        finstate = None
        for fs_div in ["CFS", "OFS"]:
            try:
                finstate = dart.finstate_all(corp_code, year, reprt_code=config.dart.report_code)
                if finstate is not None and len(finstate) > 0:
                    # fs_div 필터
                    if "fs_div" in finstate.columns:
                        fs_filtered = finstate[finstate["fs_div"] == fs_div]
                        if not fs_filtered.empty:
                            finstate = fs_filtered
                            break
                    else:
                        break
            except Exception:
                pass

        if finstate is None or len(finstate) == 0:
            # finstate_all 실패 시 finstate 시도
            try:
                _safe_sleep()
                finstate = dart.finstate(corp_code, year, reprt_code=config.dart.report_code)
            except Exception as exc2:
                logger.warning(
                    "finstate 조회 실패: corp_code=%s, year=%d, 오류=%s",
                    corp_code,
                    year,
                    exc2,
                )
                return result

        if finstate is None or len(finstate) == 0:
            logger.debug("재무제표 없음: corp_code=%s, year=%d", corp_code, year)
            return result

        for account_name in result.keys():
            result[account_name] = _extract_account(finstate, account_name)

        logger.debug(
            "재무제표 추출 성공: corp_code=%s, year=%d, 결과=%s",
            corp_code,
            year,
            {k: v for k, v in result.items() if v is not None},
        )

    except Exception as exc:
        logger.warning(
            "fetch_financials 실패: corp_code=%s, year=%d, 오류=%s",
            corp_code,
            year,
            exc,
        )

    return result
