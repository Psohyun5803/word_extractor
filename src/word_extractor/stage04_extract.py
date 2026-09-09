"""
stage 4 · TF-IDF 스코어링 → 상위 키워드 추출

stage02_tokenize.weighted_tf() 결과(TF)와 stage03_idf.fit() 결과(IDF)를 곱해
점수를 산출하고, 내림차순 (keyword, score) 리스트를 반환한다.
"""

from __future__ import annotations

from . import stage03_idf


def score_keywords(
    tf: dict[str, float],
    idf: dict[str, float],
    n_docs: int,
) -> list[tuple[str, float]]:
    """
    tf: {token: weighted_tf} — stage02_tokenize.weighted_tf() 결과
    idf: {token: idf_value}  — stage03_idf.fit() 결과
    returns: [(keyword, score), ...] 내림차순 정렬
    """
    _unseen = stage03_idf.unseen_idf(n_docs)
    scores = {
        term: count * idf.get(term, _unseen)
        for term, count in tf.items()
        if idf.get(term, _unseen) > 0
    }
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
