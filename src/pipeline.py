"""
총괄 파이프라인 실행 스크립트.

1_csv → 2_tokenize+modifier → 3_idf → 4_embed → 5_extract → 6_save

실행 모드:
    python run_pipeline.py                          # 전체 실행 (idf.json 있으면 로드)
    python run_pipeline.py --refit-idf              # IDF + 수식어 전체 재계산
    python run_pipeline.py --update-from 2026-08-10 # 새 문서만 증분 처리
    python run_pipeline.py --sim-threshold 0.25     # cosine similarity 하한 설정
    python run_pipeline.py --top-n 15               # 문서당 추출 키워드 수 변경
    python run_pipeline.py --visualize-only                   # 전체 날짜 워드클라우드 생성
    python run_pipeline.py --visualize-only 2026-08-09        # 특정 날짜만 생성
    python run_pipeline.py --visualize-only 2026-08-09 --viz-out my.html  # 출력 경로 지정
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── 경로 ─────────────────────────────────────────────────────────────────────
_SRC  = Path(__file__).parent          # src/ 디렉터리
_ROOT = _SRC.parent                    # word_extractor/ 루트


def _load_stage(stem: str):
    spec = importlib.util.spec_from_file_location(stem, _SRC / f"{stem}.py")
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
                    help="IDF + 수식어 비율 전체 재계산")
parser.add_argument("--update-from",  metavar="YYYY-MM-DD|auto",
                    help="이 날짜 이후 신규 문서만 증분 처리. 'auto'=daily_keywords.json 마지막 날짜 자동 감지")
parser.add_argument("--sim-threshold", type=float, default=0.20, metavar="F",
                    help="cosine similarity 하한 (기본 0.20)")
parser.add_argument("--max-df",        type=float, default=0.05, metavar="F",
                    help="전체 문서 비율 상한 (기본 0.05=5%% 초과 단어 제거)")
parser.add_argument("--modifier-threshold", type=float, default=0.80, metavar="F",
                    help="수식어 비율 상한 [0,1] NNG만 적용 (기본 0.80)")
parser.add_argument("--top-n",        type=int, default=5,  metavar="N",
                    help="영상당 최종 키워드 수 상한 (기본 5, 버스트 스코어 순 상위 N개)")
parser.add_argument("--min-df",        type=int,   default=3,   metavar="N",
                    help="일별 키워드 최소 문서 수 (기본 3, 미만이면 제외)")
parser.add_argument("--burst-smooth",  type=float, default=1.0, metavar="F",
                    help="버스트 감지 평활화 상수 (기본 1.0). "
                         "낮을수록 희귀어 버스트 강조, 높을수록 완화")
parser.add_argument("--emb-batch",     type=int, default=256, metavar="N",
                    help="임베딩 배치 크기 (GPU 메모리에 맞게 조정, 기본 256)")
parser.add_argument("--csv",            default="data/inputs/video_video.csv",      metavar="PATH")
parser.add_argument("--idf-out",        default="data/outputs/idf.json",             metavar="PATH")
parser.add_argument("--modifier-out",   default="data/outputs/word_modifier.json",   metavar="PATH")
parser.add_argument("--out",            default="data/outputs/daily_keywords.json",  metavar="PATH")
parser.add_argument("--visualize-only", nargs="?", const="__all__", metavar="YYYY-MM-DD",
                    help="워드클라우드만 생성. 날짜 미입력 시 전체 날짜 생성")
parser.add_argument("--viz-out",        default=None, metavar="PATH",
                    help="워드클라우드 출력 경로 (날짜 지정 시에만 유효)")
args = parser.parse_args()

# ── visualize-only 단축 경로 ─────────────────────────────────────────────────
if args.visualize_only is not None:
    visualize = _load_stage("06_visualize")
    if args.visualize_only == "__all__":
        visualize.wordcloud_all(out=args.viz_out)
    else:
        visualize.wordcloud_day(args.visualize_only, out=args.viz_out)
    sys.exit(0)

CSV_PATH      = _ROOT / args.csv
_HF_REPO      = "MindCastSogang/word_extractor"
if not CSV_PATH.exists():
    from huggingface_hub import hf_hub_download
    print(f"   CSV 없음 → HuggingFace {_HF_REPO} 에서 다운로드...")
    _dl = hf_hub_download(repo_id=_HF_REPO, filename="video_video.csv", repo_type="dataset",
                          local_dir=str(CSV_PATH.parent))
    print(f"   다운로드 완료: {_dl}")
IDF_PATH      = _ROOT / args.idf_out
MODIFIER_PATH = _ROOT / args.modifier_out
OUT_PATH      = _ROOT / args.out
VIDEO_OUT_PATH   = _ROOT / "data/outputs/video_keywords.json"
TOP_N        = args.top_n
MIN_DF       = args.min_df
if args.update_from == "auto":
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as _f:
            _ex = json.load(_f)
        UPDATE_FROM: str | None = max(_ex.keys()) if _ex else None
        print(f"   --update-from auto → 마지막 날짜: {UPDATE_FROM}")
    else:
        UPDATE_FROM = None
        print("   --update-from auto → 기존 결과 없음, 전체 처리")
else:
    UPDATE_FROM: str | None = args.update_from

# ── 1  CSV 로딩 ───────────────────────────────────────────────────────────────
print("▶ 1  CSV 로딩...")
t_start = time.time()

df_all = pd.read_csv(CSV_PATH, low_memory=False,
                     usecols=["video_id", "title", "description", "uploaded_at"])
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

texts     = [preprocess.combined_text(t, d)
             for t, d in zip(df_proc["title"], df_proc["description"])]
dates     = df_proc["date"].tolist()
video_ids = df_proc["video_id"].tolist()

# ── 2  형태소 분석 (Stage 3 IDF + Stage 6 modifier + Stage 7 추출, kiwi 1회) ──
need_modifier = (args.refit_idf or not MODIFIER_PATH.exists()) and not UPDATE_FROM
_kiwi_workers = min(32, max(4, os.cpu_count() // 2))
print(f"▶ 2  형태소 분석 (배치, num_workers={_kiwi_workers})...")
t0 = time.time()
from kiwipiepy import Kiwi as _KiwiBatch
_kiwi_batch = _KiwiBatch(num_workers=_kiwi_workers)
_KEEP = frozenset({"NNG", "NNP", "SL"})
_MOD_HEAD_PREFIXES = ("J", "X", "E", "SF", "SP")
all_tagged_seqs: list[list[tuple[str, str]]] = []
_modifier_count: defaultdict = defaultdict(int)
_head_count:     defaultdict = defaultdict(int)
for _token_list in _kiwi_batch.tokenize(texts):
    _tagged = []
    for _i, _tok in enumerate(_token_list):
        _tag = str(_tok.tag)
        _form = _tok.form.strip()
        if need_modifier and _tag in _KEEP and len(_form) >= 2 and _form not in STOPWORDS:
            if _i + 1 < len(_token_list):
                _next_tag = str(_token_list[_i + 1].tag)
                if _next_tag in _KEEP:
                    _modifier_count[_form] += 1
                elif any(_next_tag.startswith(p) for p in _MOD_HEAD_PREFIXES):
                    _head_count[_form] += 1
            else:
                _head_count[_form] += 1
        if _tok.tag not in _KEEP or len(_form) < 2 or _form in STOPWORDS:
            continue
        _tagged.append((_form, _tag))
    all_tagged_seqs.append(_tagged)
print(f"   완료: {time.time()-t0:.1f}s  ({len(all_tagged_seqs):,}개 문서)")

if need_modifier:
    _MIN_COUNT = 50
    _word_modifier_computed: dict[str, float] = {}
    for _w in set(_modifier_count) | set(_head_count):
        _mc, _hc = _modifier_count[_w], _head_count[_w]
        if _mc + _hc >= _MIN_COUNT:
            _word_modifier_computed[_w] = _mc / (_mc + _hc)
    with open(MODIFIER_PATH, "w", encoding="utf-8") as _f:
        json.dump(_word_modifier_computed, _f, ensure_ascii=False)
    print(f"   수식어 비율 저장: {len(_word_modifier_computed):,}개 → {MODIFIER_PATH.name}")

word_modifier: dict[str, float] = {}
if MODIFIER_PATH.exists():
    with open(MODIFIER_PATH, encoding="utf-8") as f:
        word_modifier = json.load(f)
    if not need_modifier:
        print(f"   수식어 비율 로드: {len(word_modifier):,}개")
USE_MODIFIER = bool(word_modifier) and args.modifier_threshold <= 1.0

# ── 3  IDF 학습 / 로드 / 증분 갱신 ──────────────────────────────────────────
all_token_seqs = None

if UPDATE_FROM and IDF_PATH.exists():
    print(f"▶ 3  IDF 증분 갱신 ({UPDATE_FROM} 이후 {len(df_proc):,}개 반영)...")
    t0 = time.time()
    new_token_seqs = [[tok for tok, _ in tagged] for tagged in all_tagged_seqs]
    idf, n_docs, df_counts = idf_mod.update(IDF_PATH, new_token_seqs)
    print(f"   완료: {time.time() - t0:.1f}s  |  어휘 {len(idf):,}개  |  총 문서 {n_docs:,}개")

elif not args.refit_idf and IDF_PATH.exists():
    print(f"▶ 3  IDF 로드  ({IDF_PATH.name})")
    idf, n_docs, df_counts = idf_mod.load(IDF_PATH)
    print(f"   어휘 {len(idf):,}개  |  학습 문서 {n_docs:,}개")

else:
    print("▶ 3  IDF 전체 학습 중...")
    t0 = time.time()
    all_token_seqs = [[tok for tok, _ in tagged] for tagged in all_tagged_seqs]
    idf, n_docs, df_counts = idf_mod.fit(all_token_seqs)
    idf_mod.save(idf, n_docs, IDF_PATH, df=df_counts)
    print(f"   완료: {time.time() - t0:.1f}s  |  어휘 {len(idf):,}개  →  {IDF_PATH.name}")

# ── 4  임베딩 ────────────────────────────────────────────────────────────────
print(f"▶ 4  임베딩 모델 로드 (ko-sroberta-multitask)...")
rerank_mod = _load_stage("05_rerank")
reranker   = rerank_mod.Reranker()
_mode = "증분" if UPDATE_FROM else "전체"
print(f"   {_mode} 임베딩 계산 중 ({len(texts):,}개, batch={args.emb_batch})...")
t0 = time.time()
doc_embs = reranker.encode_corpus(texts, batch_size=args.emb_batch)
print(f"   완료: {time.time()-t0:.1f}s")

# ── 5  문서별 키워드 추출 → 일별 카운트 ─────────────────────────────────────
MAX_DF_COUNT = args.max_df * n_docs
print(f"▶ 5  키워드 추출 (top_n={TOP_N}, sim_threshold={args.sim_threshold}, "
      f"max_df={args.max_df}, modifier={args.modifier_threshold})...")

# 증분 모드: UPDATE_FROM 이전 날짜만 로드 (이후 날짜는 새로 계산)
daily_count: dict[str, Counter] = defaultdict(Counter)
if UPDATE_FROM and OUT_PATH.exists():
    with open(OUT_PATH, encoding="utf-8") as f:
        existing = json.load(f)
    kept = {d: p for d, p in existing.items() if d <= UPDATE_FROM}
    for date, pairs in kept.items():
        daily_count[date].update({w: c for w, c in pairs})
    print(f"   기존 결과 로드: {len(kept)}일 (UPDATE_FROM 이전)")

video_kw: dict[str, list[str]] = {}
if UPDATE_FROM and VIDEO_OUT_PATH.exists():
    with open(VIDEO_OUT_PATH, encoding="utf-8") as f:
        video_kw = json.load(f)

t0 = time.time()
for idx, (text, date, vid) in enumerate(zip(texts, dates, video_ids)):
    tagged = all_tagged_seqs[idx]
    tf     = tokenize.weighted_tf(tagged)
    ranked = extract.score_keywords(tf, idf, n_docs)

    # max_df 필터
    ranked = [(w, s) for w, s in ranked if df_counts.get(w, 0) <= MAX_DF_COUNT]
    # 수식어 필터 (NNG만 적용, NNP/SL 보호)
    if USE_MODIFIER:
        word_tags = {w: tag for w, tag in tagged}
        ranked = [(w, s) for w, s in ranked
                  if word_tags.get(w, "NNG") != "NNG"
                  or word_modifier.get(w, 0.0) < args.modifier_threshold]
    ranked = reranker.rerank(doc_embs[idx], ranked, threshold=args.sim_threshold)
    # 버스트 스코어로 재정렬: tfidf_score / (df_rate + smooth)
    # df_rate = df / n_docs → 흔할수록 낮은 버스트
    ranked = sorted(
        ranked,
        key=lambda ws: ws[1] / (df_counts.get(ws[0], 0) / n_docs + args.burst_smooth / n_docs),
        reverse=True,
    )
    top_kws = [w for w, _ in ranked[:TOP_N]]
    daily_count[date].update(top_kws)
    video_kw[str(vid)] = top_kws

    if (idx + 1) % 20_000 == 0:
        elapsed   = time.time() - t0
        remaining = elapsed / (idx + 1) * (len(df_proc) - idx - 1)
        print(f"   {idx + 1:,}/{len(df_proc):,}  남은 시간 약 {remaining:.0f}초")

print(f"   완료: {time.time() - t0:.1f}s")

# ── 6  저장 ───────────────────────────────────────────────────────────────────
print("▶ 6  저장 중...")

# Stage 5에서 이미 burst 기준으로 top-N을 뽑았으므로
# daily_count는 burst 필터를 통과한 단어만 집계됨 → count 순 정렬로 충분
result = {
    date: [[w, c] for w, c in daily_count[date].most_common(30) if c >= MIN_DF]
    for date in sorted(daily_count)
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)

with open(VIDEO_OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(video_kw, f, ensure_ascii=False)

total = time.time() - t_start
print(f"\n✓ 완료: {OUT_PATH.name}  +  {VIDEO_OUT_PATH.name}  ({len(video_kw):,}개 영상)")
print(f"  일 범위: {min(result)} ~ {max(result)}  ({len(result)}일)")
print(f"  총 소요: {total:.1f}s  ({total / 60:.1f}분)")

print(f"\n샘플 (최근일 상위 10):")
last_date = max(result.keys())
for i, (w, c) in enumerate(result[last_date][:10], 1):
    print(f"  {i:2d}. {w:<14}  {c}개 문서")

