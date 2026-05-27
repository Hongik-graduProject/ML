"""클러스터 구성원 정보를 기반으로 테마 라벨 생성."""
import logging
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import config  # noqa: F401  (향후 확장용 설정 참조)

logger = logging.getLogger(__name__)


def extract_theme_keywords(
    texts_by_cluster: Dict[int, List[str]],
    top_n: int = 5,
) -> Dict[int, List[str]]:
    """클러스터별 사업개황 텍스트에서 TF-IDF 기반 상위 키워드를 추출한다.

    각 클러스터의 텍스트를 하나의 문서로 합산하여 TF-IDF를 계산한 뒤,
    IDF 가중치가 높은 상위 키워드를 반환한다.

    Args:
        texts_by_cluster: {cluster_id: [text1, text2, ...]} 딕셔너리
        top_n: 클러스터당 추출할 키워드 수

    Returns:
        {cluster_id: [keyword1, keyword2, ...]} 딕셔너리
    """
    if not texts_by_cluster:
        return {}

    cluster_ids = sorted(texts_by_cluster.keys())

    # 클러스터별 텍스트를 하나의 문서로 합침
    documents = [
        " ".join(texts_by_cluster[cid]) for cid in cluster_ids
    ]

    vectorizer = TfidfVectorizer(
        max_features=300,
        min_df=1,
        token_pattern=r"(?u)\b\w{2,}\b",  # 2글자 이상 토큰
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError as exc:
        logger.warning("TF-IDF 벡터화 실패: %s", exc)
        return {cid: [] for cid in cluster_ids}

    feature_names = vectorizer.get_feature_names_out()
    result: Dict[int, List[str]] = {}

    for idx, cid in enumerate(cluster_ids):
        row = tfidf_matrix[idx].toarray().flatten()
        top_indices = row.argsort()[::-1][:top_n]
        keywords = [feature_names[i] for i in top_indices if row[i] > 0]
        result[cid] = keywords
        logger.debug("테마 %d 키워드: %s", cid, keywords)

    return result


def generate_theme_label(
    keywords: List[str],
    members: Optional[List[str]] = None,
) -> str:
    """TF-IDF 키워드 조합으로 테마 라벨 문자열을 생성한다.

    Args:
        keywords: TF-IDF 상위 키워드 리스트
        members: 클러스터 구성 기업명 리스트 (선택, 현재는 참고용)

    Returns:
        테마 라벨 문자열 (예: "반도체_전자_부품")
    """
    if not keywords:
        return "미분류"

    label = "_".join(keywords[:3])
    logger.debug("생성된 테마 라벨: %s", label)
    return label


# ---------------------------------------------------------------------------
# LLM 기반 테마명 생성 — 인터페이스만 정의 (향후 구현)
# ---------------------------------------------------------------------------

def generate_theme_label_llm(
    keywords: List[str],
    members: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> str:
    """LLM API를 호출하여 테마 라벨을 생성한다 (인터페이스 정의).

    실제 LLM 호출은 향후 구현 예정이며, 현재는 TF-IDF 기반 라벨을 반환한다.

    Args:
        keywords: 상위 키워드 리스트
        members: 클러스터 구성 기업명 리스트
        api_key: LLM API 키 (미사용 시 None)

    Returns:
        테마 라벨 문자열
    """
    logger.info("LLM 기반 테마 라벨링은 아직 구현되지 않았습니다. TF-IDF 라벨을 반환합니다.")
    return generate_theme_label(keywords, members)
