"""
stage 2 · 형태소 분석 → 명사류 토큰 추출

kiwipiepy로 NNG(일반명사) / NNP(고유명사) / SL(외국어)만 추출한다.
고유명사는 nnp_weight 배로 가중해 TF 산정 시 반영.

수식어 비율은 태깅과 같은 kiwi 패스에서 함께 센다 — 코퍼스를 두 번 훑지
않기 위함. 어떤 단어 뒤에 명사류가 오면 수식어, 조사·어미류가 오면 head 로
세어 mod / (mod + head) 를 비율로 쓴다.
"""

from __future__ import annotations

import os
from collections import defaultdict

KEEP_TAGS = frozenset({"NNG", "NNP", "SL"})
MIN_TOKEN_LEN = 2

# 뒤따르면 앞 단어가 head 였다는 뜻인 품사 (조사/접사/어미/문장부호)
_HEAD_NEXT_PREFIXES = ("J", "X", "E", "SF", "SP")


def default_workers() -> int:
    """kiwi 배치 워커 수 — 코어의 절반, 4~32개로 제한."""
    return min(32, max(4, (os.cpu_count() or 8) // 2))


def tag_corpus(
    texts: list[str],
    stopwords: set[str],
    *,
    count_modifiers: bool = False,
    modifier_min_count: int = 50,
    num_workers: int | None = None,
) -> tuple[list[list[tuple[str, str]]], dict[str, float]]:
    """텍스트 배치 → (문서별 [(token, tag)] 리스트, 단어별 수식어 비율).

    count_modifiers=False면 수식어 비율은 빈 dict로 돌아온다.
    modifier_min_count 미만으로 등장한 단어는 비율에서 제외 (표본 부족).
    """
    # kiwipiepy 로드는 무거우므로 실제로 태깅할 때만 가져온다.
    from kiwipiepy import Kiwi

    kiwi = Kiwi(num_workers=default_workers() if num_workers is None else num_workers)

    tagged_seqs: list[list[tuple[str, str]]] = []
    modifier_count: defaultdict[str, int] = defaultdict(int)
    head_count: defaultdict[str, int] = defaultdict(int)

    for token_list in kiwi.tokenize(texts):
        tagged: list[tuple[str, str]] = []
        for i, tok in enumerate(token_list):
            tag = str(tok.tag)
            form = tok.form.strip()
            if tag not in KEEP_TAGS or len(form) < MIN_TOKEN_LEN or form in stopwords:
                continue

            if count_modifiers:
                if i + 1 < len(token_list):
                    next_tag = str(token_list[i + 1].tag)
                    if next_tag in KEEP_TAGS:
                        modifier_count[form] += 1
                    elif next_tag.startswith(_HEAD_NEXT_PREFIXES):
                        head_count[form] += 1
                else:
                    head_count[form] += 1

            tagged.append((form, tag))
        tagged_seqs.append(tagged)

    ratios = _modifier_ratios(modifier_count, head_count, modifier_min_count)
    return tagged_seqs, ratios


def _modifier_ratios(
    modifier_count: dict[str, int],
    head_count: dict[str, int],
    min_count: int,
) -> dict[str, float]:
    """{단어: 수식어로 쓰인 비율}. 등장 횟수가 min_count 미만이면 제외."""
    ratios: dict[str, float] = {}
    for word in set(modifier_count) | set(head_count):
        mod = modifier_count.get(word, 0)
        head = head_count.get(word, 0)
        if mod + head >= min_count:
            ratios[word] = mod / (mod + head)
    return ratios


def weighted_tf(
    tagged: list[tuple[str, str]],
    nnp_weight: float = 2.0,
) -> dict[str, float]:
    """(token, tag) 리스트 → {token: weighted_tf}. NNP/SL은 nnp_weight 배 가중."""
    tf: dict[str, float] = {}
    for form, tag in tagged:
        w = nnp_weight if tag in ("NNP", "SL") else 1.0
        tf[form] = tf.get(form, 0.0) + w
    return tf
