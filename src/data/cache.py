"""
cache.py

SQLite 기반 캐싱 레이어.
API 재호출 방지를 위해 결과를 로컬 DB에 저장하고 조회한다.
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional

from src.config import config

logger = logging.getLogger(__name__)

# DDL 정의
_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS business_texts (
    corp_code  TEXT    NOT NULL,
    year       INTEGER NOT NULL,
    text       TEXT    NOT NULL,
    created_at TEXT    DEFAULT (datetime('now')),
    PRIMARY KEY (corp_code, year)
);

CREATE TABLE IF NOT EXISTS financials (
    corp_code  TEXT    NOT NULL,
    year       INTEGER NOT NULL,
    data_json  TEXT    NOT NULL,
    created_at TEXT    DEFAULT (datetime('now')),
    PRIMARY KEY (corp_code, year)
);

CREATE TABLE IF NOT EXISTS stock_returns (
    ticker      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    return_pct  REAL,
    created_at  TEXT    DEFAULT (datetime('now')),
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS macro_indicators (
    date       TEXT    NOT NULL,
    indicator  TEXT    NOT NULL,
    value      REAL,
    created_at TEXT    DEFAULT (datetime('now')),
    PRIMARY KEY (date, indicator)
);
"""


def _ensure_db_dir(db_path: str) -> None:
    """DB 파일이 위치할 디렉토리를 생성한다."""
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)


def init_db(db_path: Optional[str] = None) -> None:
    """
    SQLite DB를 초기화하고 테이블을 생성한다.

    Parameters
    ----------
    db_path : str, optional
        DB 파일 경로. None이면 config.data.db_path 사용.
    """
    path = db_path or config.data.db_path
    _ensure_db_dir(path)
    with sqlite3.connect(path) as conn:
        conn.executescript(_CREATE_TABLES_SQL)
        conn.commit()
    logger.info("SQLite DB 초기화 완료: %s", path)


@contextmanager
def _get_conn(db_path: Optional[str] = None) -> Generator[sqlite3.Connection, None, None]:
    """DB 연결 컨텍스트 매니저."""
    path = db_path or config.data.db_path
    _ensure_db_dir(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# business_texts
# ---------------------------------------------------------------------------

def get_business_text(corp_code: str, year: int) -> Optional[str]:
    """캐시에서 사업의 개요 텍스트를 조회한다."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT text FROM business_texts WHERE corp_code=? AND year=?",
            (corp_code, year),
        ).fetchone()
        return row["text"] if row else None


def set_business_text(corp_code: str, year: int, text: str) -> None:
    """사업의 개요 텍스트를 캐시에 저장한다."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO business_texts (corp_code, year, text) VALUES (?,?,?)",
            (corp_code, year, text),
        )


def get_or_fetch_business_text(
    corp_code: str,
    year: int,
    fetch_fn: Callable[[], str],
) -> str:
    """
    캐시에서 사업 텍스트를 조회하고 없으면 fetch_fn을 호출한 뒤 저장한다.

    Parameters
    ----------
    corp_code : str
    year : int
    fetch_fn : Callable[[], str]
        캐시 미스 시 호출할 함수. 인자 없이 호출 가능해야 함.

    Returns
    -------
    str
    """
    cached = get_business_text(corp_code, year)
    if cached is not None:
        logger.debug("캐시 히트 (business_texts): corp_code=%s, year=%d", corp_code, year)
        return cached

    logger.debug("캐시 미스 (business_texts): corp_code=%s, year=%d", corp_code, year)
    value = fetch_fn()
    if value:
        set_business_text(corp_code, year, value)
    return value


# ---------------------------------------------------------------------------
# financials
# ---------------------------------------------------------------------------

def get_financials(corp_code: str, year: int) -> Optional[dict]:
    """캐시에서 재무제표 데이터를 조회한다."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT data_json FROM financials WHERE corp_code=? AND year=?",
            (corp_code, year),
        ).fetchone()
        if row:
            return json.loads(row["data_json"])
        return None


def set_financials(corp_code: str, year: int, data: dict) -> None:
    """재무제표 데이터를 캐시에 저장한다."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO financials (corp_code, year, data_json) VALUES (?,?,?)",
            (corp_code, year, json.dumps(data, ensure_ascii=False)),
        )


def get_or_fetch_financials(
    corp_code: str,
    year: int,
    fetch_fn: Callable[[], dict],
) -> dict:
    """
    캐시에서 재무 데이터를 조회하고 없으면 fetch_fn을 호출한 뒤 저장한다.

    Parameters
    ----------
    corp_code : str
    year : int
    fetch_fn : Callable[[], dict]

    Returns
    -------
    dict
    """
    cached = get_financials(corp_code, year)
    if cached is not None:
        logger.debug("캐시 히트 (financials): corp_code=%s, year=%d", corp_code, year)
        return cached

    logger.debug("캐시 미스 (financials): corp_code=%s, year=%d", corp_code, year)
    value = fetch_fn()
    if value:
        set_financials(corp_code, year, value)
    return value


# ---------------------------------------------------------------------------
# stock_returns
# ---------------------------------------------------------------------------

def get_stock_returns(ticker: str) -> Optional[list]:
    """캐시에서 특정 종목의 모든 월별 수익률 레코드를 조회한다."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT date, return_pct FROM stock_returns WHERE ticker=? ORDER BY date",
            (ticker,),
        ).fetchall()
        if rows:
            return [{"date": r["date"], "return_pct": r["return_pct"]} for r in rows]
        return None


def set_stock_returns(ticker: str, records: list) -> None:
    """
    월별 수익률 레코드 리스트를 캐시에 저장한다.

    Parameters
    ----------
    ticker : str
    records : list of dict
        각 dict: {"date": "YYYY-MM-DD", "return_pct": float}
    """
    with _get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO stock_returns (ticker, date, return_pct) VALUES (?,?,?)",
            [(ticker, r["date"], r["return_pct"]) for r in records],
        )


def get_or_fetch_stock_returns(
    ticker: str,
    fetch_fn: Callable[[], Any],
) -> Any:
    """
    캐시에서 주가 수익률을 조회하고 없으면 fetch_fn 호출 후 저장한다.

    Parameters
    ----------
    ticker : str
    fetch_fn : Callable[[], pd.DataFrame or list]
        DataFrame 반환 시 index가 DatetimeIndex, 컬럼 'return_pct' 가정.

    Returns
    -------
    캐시 히트 시: list of dict
    캐시 미스 시: fetch_fn의 반환값
    """
    cached = get_stock_returns(ticker)
    if cached is not None:
        logger.debug("캐시 히트 (stock_returns): ticker=%s, 레코드 수=%d", ticker, len(cached))
        return cached

    logger.debug("캐시 미스 (stock_returns): ticker=%s", ticker)
    value = fetch_fn()

    # DataFrame이면 레코드 리스트로 변환하여 저장
    try:
        import pandas as pd
        if isinstance(value, pd.DataFrame) and not value.empty:
            records = [
                {"date": str(idx.date()), "return_pct": row["return_pct"]}
                for idx, row in value.iterrows()
            ]
            set_stock_returns(ticker, records)
    except Exception as exc:
        logger.debug("stock_returns 캐시 저장 실패: %s", exc)

    return value


# ---------------------------------------------------------------------------
# macro_indicators
# ---------------------------------------------------------------------------

def get_macro_indicators(indicator: str) -> Optional[list]:
    """캐시에서 특정 거시지표의 모든 월별 레코드를 조회한다."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT date, value FROM macro_indicators WHERE indicator=? ORDER BY date",
            (indicator,),
        ).fetchall()
        if rows:
            return [{"date": r["date"], "value": r["value"]} for r in rows]
        return None


def set_macro_indicators(indicator: str, records: list) -> None:
    """
    거시지표 레코드 리스트를 캐시에 저장한다.

    Parameters
    ----------
    indicator : str
    records : list of dict
        각 dict: {"date": "YYYY-MM-DD", "value": float}
    """
    with _get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO macro_indicators (date, indicator, value) VALUES (?,?,?)",
            [(r["date"], indicator, r["value"]) for r in records],
        )


def get_or_fetch_macro_indicators(
    indicator: str,
    fetch_fn: Callable[[], Any],
) -> Any:
    """
    캐시에서 거시지표를 조회하고 없으면 fetch_fn 호출 후 저장한다.

    Parameters
    ----------
    indicator : str
    fetch_fn : Callable[[], pd.DataFrame or list]
        DataFrame 반환 시 index DatetimeIndex, 컬럼 이름이 indicator 가정.

    Returns
    -------
    캐시 히트 시: list of dict
    캐시 미스 시: fetch_fn의 반환값
    """
    cached = get_macro_indicators(indicator)
    if cached is not None:
        logger.debug(
            "캐시 히트 (macro_indicators): indicator=%s, 레코드 수=%d", indicator, len(cached)
        )
        return cached

    logger.debug("캐시 미스 (macro_indicators): indicator=%s", indicator)
    value = fetch_fn()

    try:
        import pandas as pd
        if isinstance(value, pd.DataFrame) and not value.empty and indicator in value.columns:
            series = value[indicator].dropna()
            records = [
                {"date": str(idx.date()), "value": float(v)}
                for idx, v in series.items()
            ]
            set_macro_indicators(indicator, records)
        elif isinstance(value, pd.Series) and not value.empty:
            records = [
                {"date": str(idx.date()), "value": float(v)}
                for idx, v in value.items()
            ]
            set_macro_indicators(indicator, records)
    except Exception as exc:
        logger.debug("macro_indicators 캐시 저장 실패: %s", exc)

    return value


# ---------------------------------------------------------------------------
# 범용 get_or_fetch 패턴
# ---------------------------------------------------------------------------

def get_or_fetch(cache_key: str, fetch_fn: Callable[[], Any]) -> Any:
    """
    범용 캐싱 패턴. cache_key를 기반으로 business_texts 테이블에 JSON 직렬화하여 저장.

    cache_key 형식 예:
      "business_text:{corp_code}:{year}"
      "financials:{corp_code}:{year}"

    Parameters
    ----------
    cache_key : str
    fetch_fn : Callable[[], Any]

    Returns
    -------
    Any
    """
    # 범용 캐시는 business_texts 테이블을 활용하여 cache_key 앞 50자를 corp_code로 사용
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT text FROM business_texts WHERE corp_code=? AND year=?",
            (f"__cache__{cache_key[:50]}", 0),
        ).fetchone()
        if row:
            logger.debug("캐시 히트 (generic): key=%s", cache_key)
            try:
                return json.loads(row["text"])
            except json.JSONDecodeError:
                return row["text"]

    logger.debug("캐시 미스 (generic): key=%s", cache_key)
    value = fetch_fn()

    if value is not None:
        try:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            serialized = str(value)

        with _get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO business_texts (corp_code, year, text) VALUES (?,?,?)",
                (f"__cache__{cache_key[:50]}", 0, serialized),
            )

    return value
