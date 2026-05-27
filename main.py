"""전체 ML 파이프라인 엔트리포인트.

실행 예시:
    python main.py
    python main.py --skip-crawl
    python main.py --model-type lightgbm --output-dir results/
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import config

# ---------------------------------------------------------------------------
# 로깅 설정
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI 인자 파싱
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KOSPI 200 테마 분류 + 주가 예측 ML 파이프라인"
    )
    parser.add_argument(
        "--skip-crawl",
        action="store_true",
        default=False,
        help="크롤링 단계를 건너뛰고 캐시 데이터만 사용합니다.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["ridge", "lightgbm"],
        default="ridge",
        help="예측 모델 종류 (ridge 또는 lightgbm). 기본값: ridge",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="결과 저장 디렉토리. 기본값: results/",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 환경 체크
# ---------------------------------------------------------------------------

def _check_environment() -> None:
    """필수 환경변수 및 설정 유효성 검사."""
    logger.info("[Step 0] 환경 체크 시작")

    if not config.dart.api_key:
        logger.warning(
            "DART_API_KEY 환경변수가 설정되지 않았습니다. "
            "크롤링 단계에서 오류가 발생할 수 있습니다."
        )

    # 결과 디렉토리는 main()에서 인자를 받아 생성하므로 여기서는 data 디렉토리만 확인
    os.makedirs(config.data.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.data.db_path), exist_ok=True)

    logger.info(
        "환경 체크 완료 — 대상 연도: %s, 텍스트 모델: %s",
        config.data.years,
        config.feature.text_model_name,
    )


# ---------------------------------------------------------------------------
# Step 1: KOSPI 200 종목 매핑
# ---------------------------------------------------------------------------

def _step_corp_mapping() -> List[Tuple[str, str, str]]:
    """KOSPI 200 기업 목록(기업명, 종목코드, corp_code)을 반환."""
    logger.info("[Step 1] KOSPI 200 종목 매핑 시작")
    from src.data.corp_mapper import get_kospi200_corps

    corps = get_kospi200_corps()
    logger.info("[Step 1] 종목 매핑 완료: %d개 기업", len(corps))
    return corps


# ---------------------------------------------------------------------------
# Step 2: 데이터 수집
# ---------------------------------------------------------------------------

def _step_crawl(
    corps: List[Tuple[str, str, str]],
    skip_crawl: bool,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """DART 사업보고서/재무제표 + 주가/거시지표 수집.

    Returns:
        dart_data: {(corp_code, year): {"text": str, "financials": dict}}
        stock_returns: DataFrame (index=DatetimeIndex, columns=corp_name)
        macro_df: DataFrame (index=DatetimeIndex, columns=[interest_rate, fx_rate, oil_price])
    """
    logger.info("[Step 2] 데이터 수집 시작 (skip_crawl=%s)", skip_crawl)

    import OpenDartReader
    from src.data.dart_crawler import (
        fetch_business_overview_with_fallback,
        fetch_financials,
    )
    from src.data.stock_crawler import fetch_macro_indicators, fetch_monthly_stock_returns

    start_year = min(config.data.years)
    end_year = max(config.data.years)
    start_str = f"{start_year}-01-01"
    end_str = f"{end_year}-12-31"

    dart_data: Dict = {}
    stock_returns_dict: Dict[str, pd.Series] = {}

    if not skip_crawl:
        dart = OpenDartReader.OpenDartReader(config.dart.api_key)

        total = len(corps) * len(config.data.years)
        done = 0
        for corp_name, ticker, corp_code in corps:
            prev_financials = None
            for year in config.data.years:
                key = (corp_code, year)
                logger.debug("크롤링: %s (%s) %d년", corp_name, corp_code, year)

                text = fetch_business_overview_with_fallback(dart, corp_code, year)
                financials = fetch_financials(dart, corp_code, year)
                dart_data[key] = {
                    "corp_name": corp_name,
                    "text": text,
                    "financials": financials,
                    "prev_financials": prev_financials,
                }
                prev_financials = financials
                done += 1
                if done % 50 == 0:
                    logger.info("크롤링 진행: %d / %d", done, total)

        logger.info("[Step 2] DART 크롤링 완료: %d건", len(dart_data))

        for corp_name, ticker, corp_code in corps:
            ret_df = fetch_monthly_stock_returns(ticker, start_str, end_str)
            if not ret_df.empty:
                stock_returns_dict[corp_name] = ret_df["return_pct"]

        logger.info("[Step 2] 주가 수집 완료: %d개 기업", len(stock_returns_dict))

    else:
        logger.info("[Step 2] 크롤링 건너뜀 — 캐시 데이터 사용 (dart_data 비어있음)")

    # 거시지표는 skip_crawl 여부와 무관하게 수집 (가벼운 FRED 호출)
    logger.info("[Step 2] 거시지표 수집 중...")
    macro_df = fetch_macro_indicators(start_str, end_str)
    logger.info("[Step 2] 거시지표 수집 완료: %d행", len(macro_df))

    # 주가 수익률을 단일 DataFrame으로 합치기
    if stock_returns_dict:
        stock_returns = pd.DataFrame(stock_returns_dict)
    else:
        stock_returns = pd.DataFrame()

    return dart_data, stock_returns, macro_df


# ---------------------------------------------------------------------------
# Step 3: 피처 생성
# ---------------------------------------------------------------------------

def _step_feature_engineering(
    corps: List[Tuple[str, str, str]],
    dart_data: Dict,
    stock_returns: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[int]]:
    """텍스트/재무/매크로 벡터를 생성한다.

    Returns:
        text_vecs: (N, 768) float32
        financial_vecs: (N, 10) float32
        macro_vecs: (N, 3) float32
        corp_names_flat: N개 기업명 (연도별 전개)
        years_flat: N개 연도 (연도별 전개)
    """
    logger.info("[Step 3] 피처 생성 시작")

    from src.features.financial import compute_financial_vector
    from src.features.macro import compute_macro_betas
    from src.features.text_encoder import encode_text

    text_vecs_list = []
    financial_vecs_list = []
    macro_vecs_list = []
    corp_names_flat = []
    years_flat = []

    for corp_name, ticker, corp_code in corps:
        # 해당 기업의 주가 수익률 시계열
        corp_stock = (
            stock_returns[corp_name]
            if (not stock_returns.empty and corp_name in stock_returns.columns)
            else None
        )

        # 매크로 Beta (기업 전체 기간 기준, 연도와 무관하게 1회 산출)
        if corp_stock is not None and not macro_df.empty:
            macro_vec = compute_macro_betas(corp_stock, macro_df)
        else:
            macro_vec = np.zeros(config.feature.macro_dim, dtype=np.float32)

        for year in config.data.years:
            key = (corp_code, year)
            entry = dart_data.get(key, {})

            # 텍스트 벡터
            text = entry.get("text", "")
            text_vec = encode_text(text)

            # 재무 벡터
            financials = entry.get("financials") or {}
            prev_financials = entry.get("prev_financials")
            financial_vec = compute_financial_vector(financials, prev_financials)

            text_vecs_list.append(text_vec)
            financial_vecs_list.append(financial_vec)
            macro_vecs_list.append(macro_vec)
            corp_names_flat.append(corp_name)
            years_flat.append(year)

    text_vecs = np.stack(text_vecs_list).astype(np.float32)
    financial_vecs = np.stack(financial_vecs_list).astype(np.float32)
    macro_vecs = np.stack(macro_vecs_list).astype(np.float32)

    logger.info(
        "[Step 3] 피처 생성 완료: N=%d, text=%s, financial=%s, macro=%s",
        len(corp_names_flat),
        text_vecs.shape,
        financial_vecs.shape,
        macro_vecs.shape,
    )
    return text_vecs, financial_vecs, macro_vecs, corp_names_flat, years_flat


# ---------------------------------------------------------------------------
# Step 4: 하이브리드 벡터 결합
# ---------------------------------------------------------------------------

def _step_hybrid(
    text_vecs: np.ndarray,
    financial_vecs: np.ndarray,
    macro_vecs: np.ndarray,
) -> np.ndarray:
    """가중 하이브리드 벡터 행렬 (N, 781)을 생성한다."""
    logger.info("[Step 4] 하이브리드 벡터 결합 시작")
    from src.features.hybrid import build_hybrid_matrix

    X, scalers = build_hybrid_matrix(text_vecs, financial_vecs, macro_vecs)
    logger.info("[Step 4] 하이브리드 벡터 결합 완료: shape=%s", X.shape)
    return X


# ---------------------------------------------------------------------------
# Step 5: 최적 K 탐색 + 클러스터링
# ---------------------------------------------------------------------------

def _step_clustering(
    X: np.ndarray,
    corp_names_flat: List[str],
    years_flat: List[int],
) -> Tuple[np.ndarray, int, float, List]:
    """최적 K 탐색 후 K-Means 클러스터링을 수행한다.

    Returns:
        labels: 클러스터 레이블 배열 (N,)
        best_k: 최적 K
        best_score: 최적 실루엣 스코어
        all_scores: [(k, score), ...] 전체 탐색 결과
    """
    logger.info("[Step 5] 최적 K 탐색 시작")
    from src.clustering.optimizer import find_optimal_k
    from src.clustering.theme_cluster import cluster_themes

    best_k, best_score, all_scores = find_optimal_k(X)
    logger.info("[Step 5] 최적 K=%d, 실루엣=%.4f", best_k, best_score)

    labels, _kmeans_model = cluster_themes(X, best_k)
    logger.info("[Step 5] 클러스터링 완료")

    return labels, best_k, best_score, all_scores


# ---------------------------------------------------------------------------
# Step 6: 테마 라벨링
# ---------------------------------------------------------------------------

def _step_labeling(
    labels: np.ndarray,
    corp_names_flat: List[str],
    years_flat: List[int],
    dart_data: Dict,
    corps: List[Tuple[str, str, str]],
) -> Tuple[Dict, Dict, Dict]:
    """TF-IDF 기반 테마 키워드 추출 및 라벨링.

    Returns:
        theme_summary: {theme_id: [corp_name_year_str, ...]}
        keywords: {theme_id: [keyword, ...]}
        macro_profiles: {theme_id: {"beta_interest": float, ...}}
    """
    logger.info("[Step 6] 테마 라벨링 시작")
    from src.clustering.labeler import extract_theme_keywords, generate_theme_label
    from src.clustering.theme_cluster import get_theme_summary

    theme_summary = get_theme_summary(labels, corp_names_flat, years_flat)

    # 테마별 텍스트 수집 (키워드 추출용)
    corp_code_map = {cn: cc for cn, _, cc in corps}
    texts_by_cluster: Dict[int, List[str]] = {}

    for i, (corp_name, year) in enumerate(zip(corp_names_flat, years_flat)):
        cluster_id = int(labels[i])
        corp_code = corp_code_map.get(corp_name, "")
        entry = dart_data.get((corp_code, year), {})
        text = entry.get("text", "")
        texts_by_cluster.setdefault(cluster_id, [])
        if text:
            texts_by_cluster[cluster_id].append(text)

    keywords = extract_theme_keywords(texts_by_cluster, top_n=5)

    # 테마 라벨 문자열 출력
    for theme_id, kws in sorted(keywords.items()):
        label_str = generate_theme_label(kws, theme_summary.get(theme_id, []))
        logger.info("테마 %d 라벨: %s", theme_id, label_str)

    # 매크로 프로필: 현재는 더미 (실제 Beta는 기업별로 이미 macro_vecs에 포함)
    macro_profiles: Dict[int, Dict[str, float]] = {
        theme_id: {"beta_interest": 0.0, "beta_fx": 0.0, "beta_oil": 0.0}
        for theme_id in theme_summary
    }

    logger.info("[Step 6] 테마 라벨링 완료: %d개 테마", len(theme_summary))
    return theme_summary, keywords, macro_profiles


# ---------------------------------------------------------------------------
# Step 7: 예측 모델 학습 + 백테스트
# ---------------------------------------------------------------------------

def _step_prediction(
    corps: List[Tuple[str, str, str]],
    labels: np.ndarray,
    corp_names_flat: List[str],
    years_flat: List[int],
    stock_returns: pd.DataFrame,
    macro_df: pd.DataFrame,
    model_type: str,
) -> Dict:
    """예측 모델 학습 및 백테스트를 수행한다.

    Returns:
        backtest_results: {model_type: {"rmse": float, "directional_accuracy": float, ...}}
    """
    logger.info("[Step 7] 예측 모델 학습 + 백테스트 시작 (model_type=%s)", model_type)

    if stock_returns.empty:
        logger.warning("[Step 7] 주가 데이터가 없어 예측 단계를 건너뜁니다.")
        return {}

    from src.prediction.features import build_prediction_features
    from src.prediction.model import evaluate, predict, train_model

    # theme_labels: {corp_name: cluster_id} — 연도별 최빈 클러스터로 결정
    from collections import Counter

    corp_cluster_votes: Dict[str, List[int]] = {}
    for corp_name, year, label in zip(corp_names_flat, years_flat, labels):
        corp_cluster_votes.setdefault(corp_name, []).append(int(label))

    theme_labels: Dict[str, int] = {
        corp: Counter(votes).most_common(1)[0][0]
        for corp, votes in corp_cluster_votes.items()
    }

    # 패널 DataFrame 구성
    panel_records = []
    for corp_name, ticker, corp_code in corps:
        if corp_name not in stock_returns.columns:
            continue
        for date in stock_returns.index:
            panel_records.append({"corp_name": corp_name, "date": date})

    if not panel_records:
        logger.warning("[Step 7] 패널 데이터가 비어 있어 예측 단계를 건너뜁니다.")
        return {}

    panel_df = pd.DataFrame(panel_records)

    logger.info("[Step 7] 예측 피처 빌드 중...")
    feature_df = build_prediction_features(panel_df, theme_labels, stock_returns, macro_df)

    if feature_df.empty:
        logger.warning("[Step 7] 피처 DataFrame이 비어 있어 예측 단계를 건너뜁니다.")
        return {}

    # Walk-forward 분할
    train_end = config.prediction.train_end_year
    val_year = config.prediction.val_year

    feature_col = [
        c for c in feature_df.columns
        if c not in ("corp_name", "date", "next_month_return")
    ]
    target_col = "next_month_return"

    if target_col not in feature_df.columns:
        logger.warning("[Step 7] 타깃 컬럼(%s) 없음, 예측 단계 건너뜀", target_col)
        return {}

    feature_df = feature_df.dropna(subset=[target_col])

    train_mask = feature_df["date"].dt.year <= train_end
    test_mask = feature_df["date"].dt.year == val_year

    X_train = feature_df.loc[train_mask, feature_col].values
    y_train = feature_df.loc[train_mask, target_col].values
    X_test = feature_df.loc[test_mask, feature_col].values
    y_test = feature_df.loc[test_mask, target_col].values

    if len(X_train) == 0 or len(X_test) == 0:
        logger.warning("[Step 7] 학습 또는 테스트 데이터 부족, 예측 단계 건너뜀")
        return {}

    logger.info(
        "[Step 7] 모델 학습: X_train=%s, X_test=%s", X_train.shape, X_test.shape
    )
    model = train_model(X_train, y_train, model_type=model_type)
    preds = predict(model, X_test)
    metrics = evaluate(y_test, preds)
    metrics["n_predictions"] = len(preds)

    backtest_results = {model_type: metrics}
    logger.info("[Step 7] 백테스트 완료: %s", metrics)
    return backtest_results


# ---------------------------------------------------------------------------
# Step 8: 시각화 + 리포트
# ---------------------------------------------------------------------------

def _step_visualization(
    X: np.ndarray,
    labels: np.ndarray,
    corp_names_flat: List[str],
    years_flat: List[int],
    all_scores: List,
    best_k: int,
    theme_summary: Dict,
    keywords: Dict,
    macro_profiles: Dict,
    backtest_results: Dict,
    output_dir: str,
) -> None:
    """t-SNE 시각화 + 콘솔/파일 리포트를 생성한다."""
    logger.info("[Step 8] 시각화 + 리포트 시작")

    import matplotlib
    matplotlib.use("Agg")  # 헤드리스 환경 대응

    from src.visualization.report import (
        print_prediction_report,
        print_theme_report,
        save_report_to_file,
    )
    from src.visualization.tsne_plot import plot_silhouette_scores, plot_theme_clusters

    os.makedirs(output_dir, exist_ok=True)

    # t-SNE 클러스터 시각화
    try:
        tsne_path = os.path.join(output_dir, "tsne_clusters.png")
        fig = plot_theme_clusters(
            X, labels, corp_names_flat, years_flat, save_path=tsne_path
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        logger.info("[Step 8] t-SNE 시각화 저장: %s", tsne_path)
    except Exception as exc:
        logger.warning("[Step 8] t-SNE 시각화 실패: %s", exc)

    # 실루엣 스코어 시각화
    if all_scores:
        try:
            sil_path = os.path.join(output_dir, "silhouette_scores.png")
            import matplotlib.pyplot as plt
            fig = plot_silhouette_scores(all_scores, best_k, save_path=sil_path)
            plt.close(fig)
            logger.info("[Step 8] 실루엣 스코어 시각화 저장: %s", sil_path)
        except Exception as exc:
            logger.warning("[Step 8] 실루엣 시각화 실패: %s", exc)

    # 콘솔 리포트
    print_theme_report(theme_summary, keywords, macro_profiles)
    print_prediction_report(backtest_results)

    # JSON 파일 저장
    report_data = {
        "best_k": best_k,
        "silhouette_scores": all_scores,
        "theme_summary": {str(k): v for k, v in theme_summary.items()},
        "keywords": {str(k): v for k, v in keywords.items()},
        "backtest_results": backtest_results,
    }
    report_path = os.path.join(output_dir, "report.json")
    save_report_to_file(report_data, report_path)

    logger.info("[Step 8] 시각화 + 리포트 완료")


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("KOSPI 200 테마 분류 + 주가 예측 파이프라인 시작")
    logger.info("  skip_crawl  : %s", args.skip_crawl)
    logger.info("  model_type  : %s", args.model_type)
    logger.info("  output_dir  : %s", output_dir)
    logger.info("=" * 60)

    # Step 0: 환경 체크
    _check_environment()

    # Step 1: KOSPI 200 종목 매핑
    corps = _step_corp_mapping()
    if not corps:
        logger.error("종목 매핑 결과가 없습니다. 종료합니다.")
        sys.exit(1)

    # Step 2: 데이터 수집
    dart_data, stock_returns, macro_df = _step_crawl(corps, args.skip_crawl)

    # Step 3: 피처 생성
    text_vecs, financial_vecs, macro_vecs, corp_names_flat, years_flat = (
        _step_feature_engineering(corps, dart_data, stock_returns, macro_df)
    )

    if len(corp_names_flat) == 0:
        logger.error("피처 생성 결과가 없습니다. 종료합니다.")
        sys.exit(1)

    # Step 4: 하이브리드 벡터 결합
    X = _step_hybrid(text_vecs, financial_vecs, macro_vecs)

    # Step 5: 최적 K 탐색 + 클러스터링
    labels, best_k, best_score, all_scores = _step_clustering(
        X, corp_names_flat, years_flat
    )

    # Step 6: 테마 라벨링
    theme_summary, keywords, macro_profiles = _step_labeling(
        labels, corp_names_flat, years_flat, dart_data, corps
    )

    # Step 7: 예측 모델 학습 + 백테스트
    backtest_results = _step_prediction(
        corps,
        labels,
        corp_names_flat,
        years_flat,
        stock_returns,
        macro_df,
        args.model_type,
    )

    # Step 8: 시각화 + 리포트
    _step_visualization(
        X,
        labels,
        corp_names_flat,
        years_flat,
        all_scores,
        best_k,
        theme_summary,
        keywords,
        macro_profiles,
        backtest_results,
        output_dir,
    )

    logger.info("=" * 60)
    logger.info("파이프라인 완료. 결과 디렉토리: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
