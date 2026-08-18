"""
stage 2 · 형태소 분석 → 명사류 토큰 추출

kiwipiepy로 NNG(일반명사) / NNP(고유명사) / SL(외국어)만 추출.
고유명사는 nnp_weight 배로 가중해 TF 산정 시 반영.
"""

from __future__ import annotations

from kiwipiepy import Kiwi

KEEP_TAGS = frozenset({"NNG", "NNP", "SL"})

_kiwi: Kiwi | None = None


def _get_kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def tokenize(
    text: str,
    stopwords: set[str] = frozenset(),
    min_len: int = 2,
) -> list[tuple[str, str]]:
    """텍스트 → [(token, tag), ...]. NNG/NNP/SL 품사만 반환."""
    if not text:
        return []
    result = []
    for token in _get_kiwi().tokenize(text):
        if token.tag not in KEEP_TAGS:
            continue
        form = token.form.strip()
        if len(form) < min_len or form in stopwords:
            continue
        result.append((form, str(token.tag)))
    return result


def weighted_tf(
    tagged: list[tuple[str, str]],
    nnp_weight: float = 2.0,
) -> dict[str, float]:
    """(token, tag) 리스트 → {token: weighted_tf}. NNP는 nnp_weight 배."""
    tf: dict[str, float] = {}
    for form, tag in tagged:
        tf[form] = tf.get(form, 0.0) + (nnp_weight if tag == "NNP" else 1.0)
    return tf
