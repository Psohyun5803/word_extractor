"""
stage 5 · 문서 임베딩 유사도 기반 키워드 재순위 (선택 단계)

TF-IDF 상위 후보(pool)를 문장 임베딩으로 한 번 더 보정한다.
보정 강도는 alpha 범위 [1-alpha, 1+alpha] 배수로만 조정하므로
TF-IDF 순위를 뒤집을 만큼 세게 작용하지 않는다.

모델: jhgan/ko-sroberta-multitask (Korean RoBERTa Sentence-BERT, 768-dim)
의존: sentence-transformers

사용 예:
    from 05_rerank import Reranker
    rr = Reranker()
    ranked = rr.rerank(doc_text, tfidf_ranked, pool_size=15, alpha=0.3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class Reranker:
    DEFAULT_MODEL = "jhgan/ko-sroberta-multitask"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def rerank(
        self,
        doc_text: str,
        ranked: list[tuple[str, float]],
        pool_size: int = 15,
        alpha: float = 0.3,
    ) -> list[tuple[str, float]]:
        """
        doc_text: 문서 전체 텍스트 (01_preprocess.combined_text 결과)
        ranked:   [(keyword, tfidf_score), ...] 내림차순 — 04_extract.score_keywords 결과
        pool_size: 재순위 대상 상위 N개 (나머지는 그대로 뒤에 붙임)
        alpha:    보정 강도. 후보 점수를 [1-alpha, 1+alpha] 배수로만 조정

        returns: 재정렬된 [(keyword, adjusted_score), ...]
        """
        if not ranked:
            return ranked

        pool = ranked[:pool_size]
        tail = ranked[pool_size:]
        words = [w for w, _ in pool]

        doc_emb = self._model.encode([doc_text], normalize_embeddings=True)[0]
        cand_emb = self._model.encode(words, normalize_embeddings=True)
        sims = cand_emb @ doc_emb  # cosine similarity (이미 L2 정규화됨)

        lo, hi = float(sims.min()), float(sims.max())
        span = hi - lo

        adjusted = []
        for (word, score), sim in zip(pool, sims):
            # 풀 내 상대 위치 0~1로 정규화 → [1-alpha, 1+alpha] 배수로 사상
            norm = (float(sim) - lo) / span if span > 1e-9 else 0.5
            factor = (1 - alpha) + 2 * alpha * norm
            adjusted.append((word, score * factor))

        adjusted.sort(key=lambda kv: kv[1], reverse=True)
        return adjusted + tail
