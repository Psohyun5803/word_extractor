"""
총괄 파이프라인 실행 스크립트.

01_preprocess → 02_tokenize → 03_idf → 04_extract [→ 05_rerank (optional)]

실행 모드:
    python run_pipeline.py                          # 전체 실행 (idf.json 있으면 로드)
    python run_pipeline.py --refit-idf              # IDF 전체 재계산
    python run_pipeline.py --update-from 2026-08-10 # 새 문서만 증분 처리
    python run_pipeline.py --use-rerank             # 임베딩 재순위 적용 (느림)
    python run_pipeline.py --top-n 15               # 문서당 추출 키워드 수 변경
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# ── 숫자 시작 모듈 로드 ────────────────────────────────────────────────────────
_DIR = Path(__file__).parent


def _load_stage(stem: str):
    spec = importlib.util.spec_from_file_location(stem, _DIR / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


preprocess = _load_stage("01_preprocess")
tokenize   = _load_stage("02_tokenize")
idf_mod    = _load_stage("03_idf")
extract    = _load_stage("04_extract")

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="유튜브 영상 일별 키워드 추출 파이프라인")
parser.add_argument("--refit-idf",    action="store_true",
                    help="IDF 전체 재계산 (idf.json 무시)")
parser.add_argument("--update-from",  metavar="YYYY-MM-DD",
                    help="이 날짜 이후 신규 문서만 증분 처리 (IDF도 증분 갱신)")
parser.add_argument("--use-rerank",   action="store_true",
                    help="문서 임베딩 유사도 재순위 적용 (느림)")
parser.add_argument("--top-n",        type=int, default=10, metavar="N",
                    help="문서당 추출 키워드 수 (기본 10)")
parser.add_argument("--csv",          default="data/inputs/video_video.csv",   metavar="PATH")
parser.add_argument("--idf-out",      default="data/outputs/idf.json",         metavar="PATH")
parser.add_argument("--out",          default="data/outputs/daily_keywords.json", metavar="PATH")
args = parser.parse_args()

CSV_PATH = _DIR / args.csv
IDF_PATH = _DIR / args.idf_out
OUT_PATH = _DIR / args.out
TOP_N    = args.top_n
UPDATE_FROM: str | None = args.update_from

# ── 1  CSV 로딩 ───────────────────────────────────────────────────────────────
print("▶ 1  CSV 로딩...")
t_start = time.time()

df_all = pd.read_csv(CSV_PATH, low_memory=False,
                     usecols=["title", "description", "uploaded_at"])
df_all["title"]       = df_all["title"].fillna("").astype(str)
df_all["description"] = df_all["description"].fillna("").astype(str)
df_all["date"] = pd.to_datetime(df_all["uploaded_at"], errors="coerce").dt.date.astype(str)
df_all = df_all[df_all["date"] != "NaT"].reset_index(drop=True)

print(f"   전체 {len(df_all):,}개 문서  |  {df_all['date'].min()} ~ {df_all['date'].max()}")

STOPWORDS = preprocess.DEFAULT_STOPWORDS

# 증분 모드: 기준 날짜 이후 문서만 처리
if UPDATE_FROM:
    df_new = df_all[df_all["date"] > UPDATE_FROM].reset_index(drop=True)
    df_proc = df_new
    print(f"   증분 대상: {UPDATE_FROM} 이후  {len(df_proc):,}개 문서")
else:
    df_proc = df_all

texts = [preprocess.combined_text(t, d)
         for t, d in zip(df_proc["title"], df_proc["description"])]
dates = df_proc["date"].tolist()

# ── 2  IDF 학습 / 로드 / 증분 갱신 ──────────────────────────────────────────
if UPDATE_FROM and IDF_PATH.exists():
    print(f"▶ 2  IDF 증분 갱신 ({UPDATE_FROM} 이후 {len(df_proc):,}개 반영)...")
    t0 = time.time()
    new_token_seqs = [
        [tok for tok, _ in tokenize.tokenize(text, STOPWORDS)]
        for text in texts
    ]
    idf, n_docs = idf_mod.update(IDF_PATH, new_token_seqs)
    print(f"   완료: {time.time() - t0:.1f}s  |  어휘 {len(idf):,}개  |  총 문서 {n_docs:,}개")

elif not args.refit_idf and IDF_PATH.exists():
    print(f"▶ 2  IDF 로드  ({IDF_PATH.name})")
    idf, n_docs = idf_mod.load(IDF_PATH)
    print(f"   어휘 {len(idf):,}개  |  학습 문서 {n_docs:,}개")

else:
    print("▶ 2  IDF 전체 학습 중...")
    t0 = time.time()
    all_texts = [preprocess.combined_text(t, d)
                 for t, d in zip(df_all["title"], df_all["description"])]
    token_seqs = [
        [tok for tok, _ in tokenize.tokenize(text, STOPWORDS)]
        for text in all_texts
    ]
    idf, n_docs, df_counts = idf_mod.fit(token_seqs)
    idf_mod.save(idf, n_docs, IDF_PATH, df=df_counts)
    print(f"   완료: {time.time() - t0:.1f}s  |  어휘 {len(idf):,}개  →  {IDF_PATH.name}")

# ── 3  문서별 키워드 추출 → 일별 카운트 ─────────────────────────────────────
print(f"▶ 3  키워드 추출 (top_n={TOP_N})" +
      (" + 임베딩 재순위" if args.use_rerank else "") + "...")

reranker = None
if args.use_rerank:
    rerank_mod = _load_stage("05_rerank")
    reranker   = rerank_mod.Reranker()
    print("   재순위 모델 로드 완료 (ko-sroberta-multitask)")

# 증분 모드: 기존 결과 로드 후 새 날짜만 추가
daily_count: dict[str, Counter] = defaultdict(Counter)
if UPDATE_FROM and OUT_PATH.exists():
    with open(OUT_PATH, encoding="utf-8") as f:
        existing = json.load(f)
    for date, pairs in existing.items():
        daily_count[date].update({w: c for w, c in pairs})
    print(f"   기존 결과 로드: {len(existing)}일")

t0 = time.time()
for idx, (text, date) in enumerate(zip(texts, dates)):
    tagged = tokenize.tokenize(text, STOPWORDS)
    tf     = tokenize.weighted_tf(tagged)
    ranked = extract.score_keywords(tf, idf, n_docs)

    if reranker is not None:
        ranked = reranker.rerank(text, ranked)

    daily_count[date].update(w for w, _ in ranked[:TOP_N])

    if (idx + 1) % 20_000 == 0:
        elapsed   = time.time() - t0
        remaining = elapsed / (idx + 1) * (len(df_proc) - idx - 1)
        print(f"   {idx + 1:,}/{len(df_proc):,}  남은 시간 약 {remaining:.0f}초")

print(f"   완료: {time.time() - t0:.1f}s")

# ── 4  저장 ───────────────────────────────────────────────────────────────────
print("▶ 4  저장 중...")
result = {date: daily_count[date].most_common(50) for date in sorted(daily_count)}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)

total = time.time() - t_start
print(f"\n✓ 완료: {OUT_PATH.name}")
print(f"  일 범위: {min(result)} ~ {max(result)}  ({len(result)}일)")
print(f"  총 소요: {total:.1f}s  ({total / 60:.1f}분)")

print(f"\n샘플 (최근일 상위 10):")
last_date = max(result.keys())
for i, (w, c) in enumerate(result[last_date][:10], 1):
    print(f"  {i:2d}. {w:<14}  {c}개 문서")
