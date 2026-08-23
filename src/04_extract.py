"""
stage 4 · TF-IDF 스코어링 → 상위 키워드 추출

02_tokenize.weighted_tf() 결과(TF)와 03_idf.fit() 결과(IDF)를 곱해 점수를 산출하고,
상위 top_n개 (keyword, score) 리스트를 반환한다.
"""

from __future__ import annotations

import math


def score_keywords(
    tf: dict[str, float],
    idf: dict[str, float],
    n_docs: int,
    nnp_weight: float = 2.0,  # 이미 02_tokenize에서 반영됐으면 1.0 전달
) -> list[tuple[str, float]]:
    """
    tf: {token: weighted_tf} — 02_tokenize.weighted_tf() 결과
    idf: {token: idf_value}  — 03_idf.fit() 결과
    returns: [(keyword, score), ...] 내림차순 정렬
    """
    _unseen = math.log((n_docs + 1) / 2) + 1.0
    scores = {
        term: count * idf.get(term, _unseen)
        for term, count in tf.items()
        if idf.get(term, _unseen) > 0
    }
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def top_keywords(
    tf: dict[str, float],
    idf: dict[str, float],
    n_docs: int,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """score_keywords() 결과에서 상위 top_n개만 반환."""
    return score_keywords(tf, idf, n_docs)[:top_n]
