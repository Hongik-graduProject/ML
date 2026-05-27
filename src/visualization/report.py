"""결과 리포트 모듈: 테마 분류 및 예측 결과를 콘솔/파일로 출력."""

import json
import logging
import os
from typing import Dict, List, Optional

from src.config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 콘솔 출력 헬퍼
# ---------------------------------------------------------------------------

_SEP = "=" * 70
_SUB_SEP = "-" * 70


def _print_section(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _print_subsection(title: str) -> None:
    print(f"\n{_SUB_SEP}")
    print(f"  {title}")
    print(_SUB_SEP)


# ---------------------------------------------------------------------------
# 테마 분류 리포트
# ---------------------------------------------------------------------------

def print_theme_report(
    theme_summary: Dict[int, List[str]],
    keywords: Dict[int, List[str]],
    macro_profiles: Dict[int, Dict[str, float]],
) -> None:
    """테마별 구성 기업, 키워드, 매크로 감도 평균을 콘솔에 출력한다.

    Args:
        theme_summary: {theme_id: [corp_name, ...]} 테마별 기업 목록.
        keywords: {theme_id: [keyword, ...]} 테마별 대표 키워드.
        macro_profiles: {theme_id: {"beta_interest": float, "beta_fx": float, "beta_oil": float}}
            테마별 매크로 감도 평균.
    """
    _print_section("테마 분류 결과 리포트")

    theme_ids = sorted(theme_summary.keys())
    print(f"  총 테마 수: {len(theme_ids)}")
    print(f"  총 기업 수: {sum(len(v) for v in theme_summary.values())}")

    for theme_id in theme_ids:
        corps = theme_summary.get(theme_id, [])
        kws = keywords.get(theme_id, [])
        macro = macro_profiles.get(theme_id, {})

        _print_subsection(f"테마 {theme_id}  ({len(corps)}개 기업)")

        # 구성 기업
        print(f"  [구성 기업]")
        for i, corp in enumerate(sorted(corps), start=1):
            print(f"    {i:>3}. {corp}")

        # 키워드
        if kws:
            print(f"\n  [대표 키워드]")
            print(f"    {', '.join(kws)}")

        # 매크로 감도
        if macro:
            b_interest = macro.get("beta_interest", macro.get("interest_rate", 0.0))
            b_fx = macro.get("beta_fx", macro.get("fx_rate", 0.0))
            b_oil = macro.get("beta_oil", macro.get("oil_price", 0.0))
            print(f"\n  [매크로 감도 평균]")
            print(f"    금리 Beta  : {b_interest:+.4f}")
            print(f"    환율 Beta  : {b_fx:+.4f}")
            print(f"    유가 Beta  : {b_oil:+.4f}")

    print(f"\n{_SEP}\n")


# ---------------------------------------------------------------------------
# 예측 성능 리포트
# ---------------------------------------------------------------------------

def print_prediction_report(backtest_results: dict) -> None:
    """모델별 백테스트 성능(RMSE, 방향 정확도, Sharpe Ratio)을 콘솔에 출력한다.

    Args:
        backtest_results: {
            model_name: {
                "rmse": float,
                "directional_accuracy": float,  # 0~1
                "sharpe_ratio": float,
                "n_predictions": int,           # optional
                "theme_results": {...}           # optional, 테마별 상세
            },
            ...
        }
    """
    _print_section("예측 모델 백테스트 성능 리포트")

    if not backtest_results:
        print("  [백테스트 결과 없음]")
        print(f"\n{_SEP}\n")
        return

    # 헤더
    header = f"  {'모델':<20} {'RMSE':>10} {'방향 정확도':>14} {'Sharpe':>10} {'예측 건수':>10}"
    print(header)
    print(f"  {'-'*20} {'-'*10} {'-'*14} {'-'*10} {'-'*10}")

    for model_name, metrics in sorted(backtest_results.items()):
        rmse = metrics.get("rmse", float("nan"))
        da = metrics.get("directional_accuracy", float("nan"))
        sharpe = metrics.get("sharpe_ratio", float("nan"))
        n_pred = metrics.get("n_predictions", "-")

        da_str = f"{da * 100:.2f}%" if not _isnan(da) else "N/A"
        rmse_str = f"{rmse:.6f}" if not _isnan(rmse) else "N/A"
        sharpe_str = f"{sharpe:.4f}" if not _isnan(sharpe) else "N/A"
        n_str = str(n_pred)

        print(f"  {model_name:<20} {rmse_str:>10} {da_str:>14} {sharpe_str:>10} {n_str:>10}")

    # 기준 충족 여부 표시
    print(f"\n  [검증 기준: 방향 정확도 >= 55%]")
    for model_name, metrics in sorted(backtest_results.items()):
        da = metrics.get("directional_accuracy", 0.0)
        passed = da >= 0.55 if not _isnan(da) else False
        mark = "PASS" if passed else "FAIL"
        print(f"    {model_name:<20}: {mark}  ({da * 100:.2f}%)" if not _isnan(da) else f"    {model_name:<20}: N/A")

    # 테마별 상세 (있는 경우)
    for model_name, metrics in sorted(backtest_results.items()):
        theme_results = metrics.get("theme_results")
        if not theme_results:
            continue
        _print_subsection(f"테마별 성능 상세 [{model_name}]")
        t_header = f"  {'테마':<10} {'RMSE':>10} {'방향 정확도':>14} {'Sharpe':>10}"
        print(t_header)
        print(f"  {'-'*10} {'-'*10} {'-'*14} {'-'*10}")
        for theme_id, t_metrics in sorted(theme_results.items()):
            t_rmse = t_metrics.get("rmse", float("nan"))
            t_da = t_metrics.get("directional_accuracy", float("nan"))
            t_sharpe = t_metrics.get("sharpe_ratio", float("nan"))
            t_da_str = f"{t_da * 100:.2f}%" if not _isnan(t_da) else "N/A"
            t_rmse_str = f"{t_rmse:.6f}" if not _isnan(t_rmse) else "N/A"
            t_sharpe_str = f"{t_sharpe:.4f}" if not _isnan(t_sharpe) else "N/A"
            print(f"  {str(theme_id):<10} {t_rmse_str:>10} {t_da_str:>14} {t_sharpe_str:>10}")

    print(f"\n{_SEP}\n")


def _isnan(value) -> bool:
    """float NaN 체크 (import math 없이)."""
    try:
        return value != value
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# 파일 저장
# ---------------------------------------------------------------------------

def save_report_to_file(report_data: dict, output_path: str) -> None:
    """리포트 데이터를 JSON 파일로 저장한다.

    Args:
        report_data: 저장할 리포트 딕셔너리 (JSON 직렬화 가능해야 함).
        output_path: 저장할 파일 경로 (.json 확장자 권장).

    Raises:
        OSError: 디렉터리 생성 또는 파일 쓰기 실패 시.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=_json_default)
        logger.info("리포트 저장 완료: %s", output_path)
    except OSError as e:
        logger.error("리포트 저장 실패: %s", e)
        raise


def _json_default(obj):
    """JSON 직렬화 불가 타입 처리 (numpy float/int 등)."""
    import numpy as np  # lazy import
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"JSON 직렬화 불가 타입: {type(obj)}")
