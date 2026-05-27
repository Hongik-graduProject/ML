"""t-SNE 클러스터 시각화: 하이브리드 벡터를 2D로 투영하여 테마별 군집 시각화."""

import logging
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from src.config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 한글 폰트 설정
# ---------------------------------------------------------------------------

def _setup_korean_font() -> None:
    """NanumBarunGothic 우선, 없으면 사용 가능한 한글 폰트로 fallback."""
    candidate_fonts = ["NanumBarunGothic", "NanumGothic", "AppleGothic", "Malgun Gothic"]
    available = {f.name for f in fm.fontManager.ttflist}

    selected = None
    for font in candidate_fonts:
        if font in available:
            selected = font
            break

    if selected:
        matplotlib.rc("font", family=selected)
        logger.info("한글 폰트 설정: %s", selected)
    else:
        matplotlib.rc("font", family="sans-serif")
        logger.warning(
            "한글 폰트(%s)를 찾지 못했습니다. sans-serif로 fallback.",
            candidate_fonts,
        )

    matplotlib.rcParams["axes.unicode_minus"] = False


_setup_korean_font()


# ---------------------------------------------------------------------------
# 주요 기업 라벨 표시 (상위 N개 기업만)
# ---------------------------------------------------------------------------

_MAX_LABEL_CORPS = 20  # 텍스트 라벨을 표시할 최대 기업 수


def plot_theme_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    corp_names: List[str],
    years: Optional[List[int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """하이브리드 벡터를 t-SNE로 2D 투영하여 테마별 군집을 시각화한다.

    Args:
        X: 하이브리드 벡터 행렬, shape (n_samples, n_features).
        labels: 클러스터 레이블 배열, shape (n_samples,).
        corp_names: 각 샘플에 대응하는 기업명 리스트 (len == n_samples).
        years: 각 샘플에 대응하는 연도 리스트 (len == n_samples). None이면 미표시.
        save_path: 저장 경로 (확장자 포함). None이면 저장하지 않음.

    Returns:
        matplotlib.Figure 객체.
    """
    if X.shape[0] < 2:
        raise ValueError(f"t-SNE를 위한 샘플 수가 부족합니다: {X.shape[0]}")

    # t-SNE 투영
    perplexity = min(30, X.shape[0] - 1)
    logger.info(
        "t-SNE 시작: n_samples=%d, n_features=%d, perplexity=%d",
        X.shape[0],
        X.shape[1],
        perplexity,
    )
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=config.cluster.random_state,
        n_iter=1000,
    )
    X_2d = tsne.fit_transform(X)
    logger.info("t-SNE 완료.")

    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)
    palette = sns.color_palette("tab20", n_colors=max(n_clusters, 1))

    fig, ax = plt.subplots(figsize=(14, 10))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        color = palette[idx % len(palette)]
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=[color],
            label=f"테마 {label}",
            alpha=0.7,
            s=40,
            edgecolors="white",
            linewidths=0.3,
        )

    # 주요 기업 텍스트 라벨 (샘플 수가 많으면 일부만 표시)
    n_samples = len(corp_names)
    step = max(1, n_samples // _MAX_LABEL_CORPS)
    label_indices = list(range(0, n_samples, step))[:_MAX_LABEL_CORPS]

    for i in label_indices:
        name = corp_names[i]
        year_str = f"\n({years[i]})" if years is not None else ""
        ax.annotate(
            f"{name}{year_str}",
            xy=(X_2d[i, 0], X_2d[i, 1]),
            fontsize=7,
            alpha=0.8,
            ha="center",
            va="bottom",
        )

    ax.set_title("테마별 기업 군집 (t-SNE 2D 투영)", fontsize=15, pad=15)
    ax.set_xlabel("t-SNE 축 1", fontsize=11)
    ax.set_ylabel("t-SNE 축 2", fontsize=11)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.15, 1.0),
        fontsize=9,
        title="테마",
        title_fontsize=10,
        framealpha=0.8,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("t-SNE 시각화 저장: %s", save_path)

    return fig


def plot_silhouette_scores(
    scores: List[Tuple[int, float]],
    best_k: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """K별 실루엣 스코어를 막대 그래프로 시각화한다.

    Args:
        scores: [(k, silhouette_score), ...] 리스트.
        best_k: 최적 K값 (강조 표시).
        save_path: 저장 경로 (확장자 포함). None이면 저장하지 않음.

    Returns:
        matplotlib.Figure 객체.
    """
    if not scores:
        raise ValueError("scores 리스트가 비어 있습니다.")

    ks = [s[0] for s in scores]
    sil = [s[1] for s in scores]

    colors = ["#e74c3c" if k == best_k else "#3498db" for k in ks]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ks, sil, color=colors, edgecolor="white", linewidth=0.8)

    # 최적 K 강조
    best_score = dict(scores).get(best_k, 0.0)
    ax.axhline(y=best_score, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.annotate(
        f"최적 K={best_k}\n(score={best_score:.4f})",
        xy=(best_k, best_score),
        xytext=(best_k + 0.5, best_score + 0.01),
        fontsize=9,
        color="#e74c3c",
        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2),
    )

    # 막대 위 값 표시
    for bar, score in zip(bars, sil):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.002,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title("K별 실루엣 스코어 (최적 K 탐색)", fontsize=14, pad=12)
    ax.set_xlabel("클러스터 수 K", fontsize=11)
    ax.set_ylabel("실루엣 스코어", fontsize=11)
    ax.set_xticks(ks)
    ax.set_ylim(0, max(sil) * 1.15 if sil else 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("실루엣 스코어 시각화 저장: %s", save_path)

    return fig
