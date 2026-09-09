"""
stage 5 · 문서 임베딩 유사도 기반 키워드 재순위 + 필터링 (선택 단계)

TF-IDF 상위 후보(pool)를 문장 임베딩으로 한 번 더 보정한다.
보정 강도는 alpha 범위 [1-alpha, 1+alpha] 배수로만 조정하므로
TF-IDF 순위를 뒤집을 만큼 세게 작용하지 않는다.

threshold > 0 이면 문서 임베딩과 cosine similarity가 threshold 미만인
키워드를 결과에서 제거한다 (맥락 무관 단어 필터링).

모델: jhgan/ko-sroberta-multitask (Korean RoBERTa Sentence-BERT, 768-dim)
의존: sentence-transformers

사용 예:
    from word_extractor.stage05_rerank import Reranker
    rr = Reranker()
    ranked = rr.rerank(doc_text, tfidf_ranked, pool_size=15, alpha=0.3)
    ranked = rr.rerank(doc_text, tfidf_ranked, threshold=0.1)  # 필터링만
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

    def encode_corpus(self, texts: list[str], batch_size: int = 256) -> "np.ndarray":
        """전체 문서 텍스트를 배치 인코딩."""
        import numpy as np
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def rerank(
        self,
        doc_text_or_emb,
        ranked: list[tuple[str, float]],
        pool_size: int = 15,
        alpha: float = 0.3,
        threshold: float = 0.0,
        word_embs: "dict[str, np.ndarray] | None" = None,
    ) -> list[tuple[str, float]]:
        """
        doc_text_or_emb: 문서 텍스트(str) 또는 사전 인코딩된 임베딩 벡터(np.ndarray)
        ranked:    [(keyword, tfidf_score), ...] 내림차순
        pool_size: 재순위 대상 상위 N개
        alpha:     보정 강도 [1-alpha, 1+alpha]
        threshold: cosine similarity 하한 (0.0=필터링 없음)
        word_embs: 사전 계산된 어휘 임베딩 {word: emb} — 제공 시 model.encode 생략
        """
        import numpy as np

        if not ranked:
            return ranked

        pool = ranked[:pool_size]
        tail = ranked[pool_size:]
        words = [w for w, _ in pool]

        if isinstance(doc_text_or_emb, str):
            doc_emb = self._model.encode([doc_text_or_emb], normalize_embeddings=True)[0]
        else:
            doc_emb = doc_text_or_emb  # 사전 계산된 벡터

        if word_embs is not None:
            available = [(item, word_embs[w]) for item, w in zip(pool, words) if w in word_embs]
            if not available:
                return tail
            pool, emb_list = zip(*available)
            pool = list(pool)
            cand_emb = np.stack(emb_list)
        else:
            cand_emb = self._model.encode(words, normalize_embeddings=True)
        sims     = cand_emb @ doc_emb

        pool_filtered = [(item, float(sim)) for item, sim in zip(pool, sims)
                         if float(sim) >= threshold]

        if not pool_filtered:
            return tail

        items_f, sims_f = zip(*pool_filtered)
        sims_f = list(sims_f)

        lo, hi = min(sims_f), max(sims_f)
        span = hi - lo

        adjusted = []
        for (word, score), sim in zip(items_f, sims_f):
            norm = (sim - lo) / span if span > 1e-9 else 0.5
            factor = (1 - alpha) + 2 * alpha * norm
            adjusted.append((word, score * factor))

        adjusted.sort(key=lambda kv: kv[1], reverse=True)
        return adjusted + tail
