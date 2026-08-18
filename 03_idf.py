"""
stage 3 · 코퍼스 IDF 계산 / 저장 / 로드 / 증분 업데이트

스무딩 IDF: log((N+1) / (df+1)) + 1  →  항상 양수, 흔한 단어일수록 낮은 값.
max_df 비율을 초과하는 단어(채널 브랜드 등)는 idf=0 처리해 사실상 제외.

idf.json 저장 구조:
  {
    "n_docs": 259583,
    "max_df": 0.3,
    "df":  {"단어": 문서빈도, ...},   ← 증분 업데이트를 위해 raw df 보존
    "idf": {"단어": idf값, ...}
  }
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


def _compute_idf(df: dict[str, int], n_docs: int, max_df: float) -> dict[str, float]:
    max_df_count = max_df * n_docs
    return {
        term: (
            0.0
            if freq > max_df_count
            else math.log((n_docs + 1) / (freq + 1)) + 1.0
        )
        for term, freq in df.items()
    }


def fit(
    token_seqs: Iterable[list[str]],
    max_df: float = 0.3,
) -> tuple[dict[str, float], int, dict[str, int]]:
    """
    token_seqs: 문서당 토큰 리스트의 iterable
    returns: (idf_dict, n_docs, df_counts)
    """
    df: Counter[str] = Counter()
    n_docs = 0
    for tokens in token_seqs:
        unique = set(tokens)
        if not unique:
            continue
        n_docs += 1
        df.update(unique)

    idf = _compute_idf(df, n_docs, max_df)
    return idf, n_docs, dict(df)


def update(
    path: str | Path,
    new_token_seqs: Iterable[list[str]],
) -> tuple[dict[str, float], int]:
    """
    기존 idf.json에 새 문서들을 증분 반영해 IDF를 갱신한다.

    new_token_seqs: 새로 추가된 문서들의 토큰 리스트
    returns: (updated_idf, updated_n_docs)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    n_docs = data["n_docs"]
    max_df = data.get("max_df", 0.3)
    df: Counter[str] = Counter(data.get("df", {}))

    new_docs = 0
    for tokens in new_token_seqs:
        unique = set(tokens)
        if not unique:
            continue
        new_docs += 1
        df.update(unique)

    if new_docs == 0:
        return data["idf"], n_docs

    n_docs += new_docs
    idf = _compute_idf(df, n_docs, max_df)

    _write(path, idf, n_docs, max_df, dict(df))
    return idf, n_docs


def unseen_idf(n_docs: int) -> float:
    """코퍼스에 없던 단어에 줄 IDF (df=1 취급)."""
    return math.log((n_docs + 1) / 2) + 1.0


def _write(
    path: str | Path,
    idf: dict[str, float],
    n_docs: int,
    max_df: float,
    df: dict[str, int],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"n_docs": n_docs, "max_df": max_df, "df": df, "idf": idf},
            f,
            ensure_ascii=False,
        )


def save(
    idf: dict[str, float],
    n_docs: int,
    path: str | Path,
    max_df: float = 0.3,
    df: dict[str, int] | None = None,
) -> None:
    _write(path, idf, n_docs, max_df, df or {})


def load(path: str | Path) -> tuple[dict[str, float], int]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["idf"], data["n_docs"]
