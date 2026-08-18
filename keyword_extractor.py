"""
유튜브 영상 title + description(본문)에서 핵심 단어를 추출하는 모듈.

- 형태소 분석(kiwipiepy)으로 명사류를 뽑아내고
- 코퍼스 전체의 IDF와 문서별 TF를 곱한 TF-IDF 점수로 순위를 매긴다.
- title과 description은 동일하게 취급한다 (가중치 없음).
- 고유명사(NNP)는 nnp_weight배로 가중해 TF-IDF 순위에 우선 반영한다.
- description에 반복되는 저작권 문구/타임코드 목차 줄 등 상투 문구는 토큰화 전에 제거한다.
- 코퍼스 내 문서빈도가 max_df 비율을 넘는 단어(채널 브랜드 해시태그 등)는 자동으로 제외한다.
- (옵션) use_doc_similarity=True면 문장임베딩으로 "문서 전체 의미와 이 단어가 얼마나
  가까운가"를 약한 보정치로 곱해 재순위한다. TF-IDF 순위를 뒤집을 만큼 세게는 아니고,
  doc_similarity_alpha 범위 안에서만 살짝 흔든다.

사용 흐름:
    extractor = KeywordExtractor()
    extractor.fit(list_of_texts)                  # 코퍼스 전체 1회 통과, IDF 계산
    extractor.save_idf("idf.json")                 # 재사용을 위해 저장
    ...
    extractor.load_idf("idf.json")                 # 다음 실행에서 재계산 없이 로드
    keywords = extractor.extract_keywords(title, description, top_n=10)
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from kiwipiepy import Kiwi

# 키워드로 유의미한 형태소 품사만 사용 (일반명사, 고유명사, 외국어, 숫자+단위 등)
_KEEP_TAGS = {"NNG", "NNP", "SL"}

_URL_RE = re.compile(r"https?://\S+")
_HASHTAG_MARK_RE = re.compile(r"#")
_WHITESPACE_RE = re.compile(r"\s+")

# description에 반복적으로 붙는 상투 문구 줄 제거용 패턴
_COPYRIGHT_LINE_RE = re.compile(
    r"^.*(ⓒ|©|무단\s*전재|재배포|저작권).*$", re.MULTILINE
)
_TIMECODE_LINE_RE = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s+.*$", re.MULTILINE)
_FILLER_CHAR_RE = re.compile(r"[ㅤ​]")  # 한글 채움 문자, zero-width space

DEFAULT_STOPWORDS = {
    "영상", "오늘", "이번", "구독", "채널", "뉴스", "속보", "생방송", "라이브",
    "출처", "저작권", "무단", "전재", "재배포", "금지", "이용", "학습",
    "제공", "촬영", "편집", "댓글", "좋아요", "알림", "설정", "안내", "공지",
}


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = _URL_RE.sub(" ", text)
    text = _HASHTAG_MARK_RE.sub(" ", text)  # '#' 기호만 제거, 뒤 단어는 남김
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _strip_boilerplate(description: str | None) -> str:
    """description에서 저작권 고지, 타임코드 목차 줄, 채움 문자 등 상투 문구를 제거."""
    if not description:
        return ""
    text = _FILLER_CHAR_RE.sub(" ", description)
    text = _COPYRIGHT_LINE_RE.sub(" ", text)
    text = _TIMECODE_LINE_RE.sub(" ", text)
    return text


@dataclass
class KeywordExtractor:
    stopwords: set[str] = field(default_factory=lambda: set(DEFAULT_STOPWORDS))
    min_token_len: int = 2
    max_df: float = 0.3  # 이 비율보다 많은 문서에 등장하는 단어는 자동 제외 (채널 브랜드 해시태그 등)
    nnp_weight: float = 2.0  # 고유명사(NNP)에 줄 가중치. 일반명사(NNG)는 1.0 그대로.

    # ---- 문서 유사도 보정 (옵션, 무거움) ----
    use_doc_similarity: bool = False
    doc_similarity_model: str = "jhgan/ko-sroberta-multitask"
    doc_similarity_alpha: float = 0.3  # 보정 강도. 후보 점수를 [1-alpha, 1+alpha] 배수로만 살짝 조정
    rerank_pool_size: int = 15  # TF-IDF 상위 몇 개까지를 재순위 대상으로 볼지

    idf_: dict[str, float] = field(default_factory=dict, repr=False)
    n_docs_: int = 0

    def __post_init__(self) -> None:
        self._kiwi = Kiwi()
        self._embed_model = None
        if self.use_doc_similarity:
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(self.doc_similarity_model)

    # ---- 토큰화 ----

    def _tokenize_with_tags(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        tokens = []
        for token in self._kiwi.tokenize(text):
            if token.tag not in _KEEP_TAGS:
                continue
            form = token.form.strip()
            if len(form) < self.min_token_len:
                continue
            if form in self.stopwords:
                continue
            tokens.append((form, token.tag))
        return tokens

    def _tokenize_text(self, text: str) -> list[str]:
        return [form for form, _tag in self._tokenize_with_tags(text)]

    @staticmethod
    def _combined_text(title: str | None, description: str | None) -> str:
        return f"{_clean_text(title)} {_clean_text(_strip_boilerplate(description))}".strip()

    def tokenize(self, title: str | None, description: str | None) -> list[str]:
        """title+description을 합쳐 명사류 토큰 리스트로 변환 (IDF 학습용, 가중치 없음)."""
        return self._tokenize_text(self._combined_text(title, description))

    # ---- IDF 학습 ----

    def fit(self, texts: Iterable[tuple[str | None, str | None]]) -> "KeywordExtractor":
        """
        texts: (title, description) 튜플들의 iterable.
        코퍼스를 한 번 순회하며 document frequency를 세고 IDF를 계산한다.
        문서빈도가 max_df 비율을 넘는 단어는 idf=0으로 처리해 키워드 후보에서 사실상 제외한다.
        """
        df: Counter[str] = Counter()
        n_docs = 0
        for title, description in texts:
            tokens = set(self.tokenize(title, description))
            if not tokens:
                continue
            n_docs += 1
            df.update(tokens)

        self.n_docs_ = n_docs
        max_df_count = self.max_df * n_docs
        # 스무딩 IDF: log((N+1)/(df+1)) + 1  -> 항상 양수, 흔한 단어일수록 낮은 값
        self.idf_ = {
            term: 0.0 if freq > max_df_count else math.log((n_docs + 1) / (freq + 1)) + 1.0
            for term, freq in df.items()
        }
        return self

    def save_idf(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"n_docs": self.n_docs_, "idf": self.idf_},
                f,
                ensure_ascii=False,
            )

    def load_idf(self, path: str) -> "KeywordExtractor":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.n_docs_ = data["n_docs"]
        self.idf_ = data["idf"]
        return self

    # ---- 키워드 추출 ----

    def _unseen_idf(self) -> float:
        """학습 코퍼스에 없던 단어에 줄 IDF (df=1로 취급, 최댓값에 가까운 희귀도)."""
        return math.log((self.n_docs_ + 1) / 2) + 1.0

    def extract_keywords(
        self,
        title: str | None,
        description: str | None,
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """title+description에서 top_n개의 (키워드, 점수)를 반환. 고유명사(NNP)는 nnp_weight배로 가중.

        use_doc_similarity=True면 TF-IDF 상위 rerank_pool_size개를 문서 임베딩 유사도로
        한 번 더 (약하게) 보정한 뒤 top_n을 뽑는다.
        """
        doc_text = self._combined_text(title, description)
        tagged = self._tokenize_with_tags(doc_text)
        if not tagged:
            return []

        tf: Counter[str] = Counter()
        for form, tag in tagged:
            tf[form] += self.nnp_weight if tag == "NNP" else 1.0

        unseen_idf = self._unseen_idf()
        scores = {
            term: count * self.idf_.get(term, unseen_idf)
            for term, count in tf.items()
        }
        ranked = sorted(
            (kv for kv in scores.items() if kv[1] > 0),
            key=lambda kv: kv[1],
            reverse=True,
        )

        if self._embed_model is not None and ranked:
            ranked = self._rerank_by_doc_similarity(doc_text, ranked)

        return ranked[:top_n]

    def _rerank_by_doc_similarity(
        self, doc_text: str, ranked: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        pool = ranked[: self.rerank_pool_size]
        words = [w for w, _ in pool]

        doc_emb = self._embed_model.encode([doc_text], normalize_embeddings=True)[0]
        cand_emb = self._embed_model.encode(words, normalize_embeddings=True)
        sims = cand_emb @ doc_emb

        lo, hi = float(sims.min()), float(sims.max())
        span = hi - lo
        alpha = self.doc_similarity_alpha
        adjusted = []
        for (word, score), sim in zip(pool, sims):
            norm = (sim - lo) / span if span > 1e-9 else 0.5  # 후보 풀 내 상대 위치, 0~1
            factor = (1 - alpha) + 2 * alpha * norm  # [1-alpha, 1+alpha] 범위로 사상
            adjusted.append((word, score * factor))

        adjusted.sort(key=lambda kv: kv[1], reverse=True)
        return adjusted + ranked[self.rerank_pool_size :]
