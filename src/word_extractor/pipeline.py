"""
총괄 파이프라인 실행 스크립트.

1 CSV 로딩 → 2 형태소 분석+수식어 → 3 IDF → 4 임베딩 → 5 키워드 추출 → 6 저장

실행 모드:
    python pipeline.py                          # 전체 실행 (idf.json 있으면 로드)
    python pipeline.py --refit-idf              # IDF + 수식어 전체 재계산
    python pipeline.py --update-from 2026-08-10 # 새 문서만 증분 처리
    python pipeline.py --sim-threshold 0.25     # cosine similarity 하한 설정
    python pipeline.py --top-n 15               # 문서당 추출 키워드 수 변경
    python pipeline.py --visualize-only                   # 전체 날짜 워드클라우드 생성
    python pipeline.py --visualize-only 2026-08-09        # 특정 날짜만 생성
    python pipeline.py --visualize-only 2026-08-09 --viz-out my.html  # 출력 경로 지정

run_pipeline()은 CSV → daily/video 키워드 딕셔너리를 in-memory로 계산만 하는
라이브러리 함수로, 파일 저장은 하지 않는다 (idf.json/word_modifier.json 캐시는 예외).
외부 서비스에서 JSON 결과 파일을 거치지 않고 직접 호출해 쓰기 위함.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import click
import pandas as pd
import word_extractor.stage01_preprocess as preprocess
import word_extractor.stage02_tokenize as tokenize
import word_extractor.stage03_idf as idf_mod
import word_extractor.stage04_extract as extract

# ── 경로 ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent  # word_extractor/ 루트

_HF_REPO = "MindCastSogang/word_extractor"


def _load_documents(
        csv_path: Path,
        update_from: str | None,
) -> tuple[list[str], list[str], list[str]]:
    """CSV → (texts, dates, video_ids). update_from 이후 문서만 남긴다."""
    df = pd.read_csv(csv_path, low_memory=False,
                     usecols=["video_id", "title", "description", "uploaded_at"])
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["date"] = pd.to_datetime(df["uploaded_at"], errors="coerce").dt.date.astype(str)
    df = df[df["date"] != "NaT"].reset_index(drop=True)
    print(f"   전체 {len(df):,}개 문서  |  {df['date'].min()} ~ {df['date'].max()}")

    if update_from:
        df = df[df["date"] > update_from].reset_index(drop=True)
        print(f"   증분 대상: {update_from} 이후  {len(df):,}개 문서")

    texts = [preprocess.combined_text(t, d)
             for t, d in zip(df["title"], df["description"])]
    return texts, df["date"].tolist(), df["video_id"].tolist()


def _resolve_idf(
        idf_path: Path,
        tagged_seqs: list[list[tuple[str, str]]],
        *,
        refit_idf: bool,
        update_from: str | None,
) -> tuple[dict[str, float], int, dict[str, int]]:
    """(idf, n_docs, df_counts) — 증분 갱신 / 기존 로드 / 전체 학습 중 하나."""
    token_seqs = [[tok for tok, _ in tagged] for tagged in tagged_seqs]

    if update_from and idf_path.exists():
        print(f"▶ 3  IDF 증분 갱신 ({update_from} 이후 {len(tagged_seqs):,}개 반영)...")
        t0 = time.time()
        idf, n_docs, df_counts = idf_mod.update(idf_path, token_seqs)
        print(f"   완료: {time.time() - t0:.1f}s  |  어휘 {len(idf):,}개  |  총 문서 {n_docs:,}개")

    elif not refit_idf and idf_path.exists():
        print(f"▶ 3  IDF 로드  ({idf_path.name})")
        idf, n_docs, df_counts = idf_mod.load(idf_path)
        print(f"   어휘 {len(idf):,}개  |  학습 문서 {n_docs:,}개")

    else:
        print("▶ 3  IDF 전체 학습 중...")
        t0 = time.time()
        idf, n_docs, df_counts = idf_mod.fit(token_seqs)
        idf_mod.save(idf, n_docs, idf_path, df=df_counts)
        print(f"   완료: {time.time() - t0:.1f}s  |  어휘 {len(idf):,}개  →  {idf_path.name}")

    return idf, n_docs, df_counts


def run_pipeline(
        csv_path: Path,
        idf_path: Path,
        modifier_path: Path,
        *,
        refit_idf: bool = False,
        update_from: str | None = None,
        sim_threshold: float = 0.20,
        max_df: float = 0.05,
        modifier_threshold: float = 0.80,
        top_n: int = 5,
        min_df: int = 3,
        burst_smooth: float = 1.0,
        emb_batch: int = 256,
        existing_daily: dict[str, list] | None = None,
        existing_video: dict[str, list[str]] | None = None,
) -> tuple[dict[str, list[list]], dict[str, list[str]]]:
    """CSV → (daily_keywords, video_keywords) 딕셔너리를 in-memory로 계산해 반환.

    idf.json / word_modifier.json 은 코퍼스 통계 캐시로 계속 읽고 쓰지만,
    결과물(daily/video 키워드)은 파일에 쓰지 않는다 — 호출자가 필요하면 직접 저장한다.
    """

    if not csv_path.exists():
        from huggingface_hub import hf_hub_download
        print(f"   CSV 없음 → HuggingFace {_HF_REPO} 에서 다운로드...")
        _dl = hf_hub_download(repo_id=_HF_REPO, filename="video_video.csv", repo_type="dataset",
                              local_dir=str(csv_path.parent))
        print(f"   다운로드 완료: {_dl}")

    # ── 1  CSV 로딩 ───────────────────────────────────────────────────────────
    print("▶ 1  CSV 로딩...")
    texts, dates, video_ids = _load_documents(csv_path, update_from)

    # ── 2  형태소 분석 (IDF + 수식어 비율 + 키워드 추출에서 재사용, kiwi 1회) ──
    need_modifier = (refit_idf or not modifier_path.exists()) and not update_from
    workers = tokenize.default_workers()
    print(f"▶ 2  형태소 분석 (배치, num_workers={workers})...")
    t0 = time.time()
    all_tagged_seqs, modifier_ratios = tokenize.tag_corpus(
        texts,
        preprocess.DEFAULT_STOPWORDS,
        count_modifiers=need_modifier,
        num_workers=workers,
    )
    print(f"   완료: {time.time() - t0:.1f}s  ({len(all_tagged_seqs):,}개 문서)")

    if need_modifier:
        word_modifier = modifier_ratios
        with open(modifier_path, "w", encoding="utf-8") as f:
            json.dump(word_modifier, f, ensure_ascii=False)
        print(f"   수식어 비율 저장: {len(word_modifier):,}개 → {modifier_path.name}")
    elif modifier_path.exists():
        with open(modifier_path, encoding="utf-8") as f:
            word_modifier = json.load(f)
        print(f"   수식어 비율 로드: {len(word_modifier):,}개")
    else:
        word_modifier = {}
    use_modifier = bool(word_modifier) and modifier_threshold <= 1.0

    # ── 3  IDF 학습 / 로드 / 증분 갱신 ──────────────────────────────────────────
    idf, n_docs, df_counts = _resolve_idf(
        idf_path, all_tagged_seqs, refit_idf=refit_idf, update_from=update_from,
    )

    # ── 4  임베딩 ────────────────────────────────────────────────────────────────
    print("▶ 4  임베딩 모델 로드 (ko-sroberta-multitask)...")
    from . import stage05_rerank as rerank_mod
    reranker = rerank_mod.Reranker()
    _mode = "증분" if update_from else "전체"
    print(f"   {_mode} 임베딩 계산 중 ({len(texts):,}개, batch={emb_batch})...")
    t0 = time.time()
    doc_embs = reranker.encode_corpus(texts, batch_size=emb_batch)
    print(f"   완료: {time.time() - t0:.1f}s")

    # ── 5  문서별 키워드 추출 → 일별 카운트 ─────────────────────────────────────
    max_df_count = max_df * n_docs
    print(f"▶ 5  키워드 추출 (top_n={top_n}, sim_threshold={sim_threshold}, "
          f"max_df={max_df}, modifier={modifier_threshold})...")

    daily_count: dict[str, Counter] = defaultdict(Counter)
    for date, pairs in (existing_daily or {}).items():
        daily_count[date].update({w: c for w, c in pairs})

    video_kw: dict[str, list[str]] = dict(existing_video or {})

    t0 = time.time()
    for idx, (date, vid) in enumerate(zip(dates, video_ids)):
        tagged = all_tagged_seqs[idx]
        tf = tokenize.weighted_tf(tagged)
        ranked = extract.score_keywords(tf, idf, n_docs)

        # max_df 필터
        ranked = [(w, s) for w, s in ranked if df_counts.get(w, 0) <= max_df_count]
        # 수식어 필터 (NNG만 적용, NNP/SL 보호)
        if use_modifier:
            word_tags = {w: tag for w, tag in tagged}
            ranked = [(w, s) for w, s in ranked
                      if word_tags.get(w, "NNG") != "NNG"
                      or word_modifier.get(w, 0.0) < modifier_threshold]
        ranked = reranker.rerank(doc_embs[idx], ranked, threshold=sim_threshold)
        # 버스트 스코어로 재정렬: tfidf_score / (df_rate + smooth)
        ranked = sorted(
            ranked,
            key=lambda ws: ws[1] / (df_counts.get(ws[0], 0) / n_docs + burst_smooth / n_docs),
            reverse=True,
        )
        top_kws = [w for w, _ in ranked[:top_n]]
        daily_count[date].update(top_kws)
        video_kw[str(vid)] = top_kws

        if (idx + 1) % 20_000 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (idx + 1) * (len(texts) - idx - 1)
            print(f"   {idx + 1:,}/{len(texts):,}  남은 시간 약 {remaining:.0f}초")

    print(f"   완료: {time.time() - t0:.1f}s")

    # Stage 5에서 이미 burst 기준으로 top-N을 뽑았으므로
    # daily_count는 burst 필터를 통과한 단어만 집계됨 → count 순 정렬로 충분
    result = {
        date: [[w, c] for w, c in daily_count[date].most_common(30) if c >= min_df]
        for date in sorted(daily_count)
    }
    return result, video_kw


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--refit-idf", is_flag=True,
              help="IDF + 수식어 비율 전체 재계산")
@click.option("--update-from", metavar="YYYY-MM-DD|auto", default=None,
              help="이 날짜 이후 신규 문서만 증분 처리. 'auto'=daily_keywords.json 마지막 날짜 자동 감지")
@click.option("--sim-threshold", type=float, default=0.20, metavar="F",
              help="cosine similarity 하한 (기본 0.20)")
@click.option("--max-df", type=float, default=0.05, metavar="F",
              help="전체 문서 비율 상한 (기본 0.05=5% 초과 단어 제거)")
@click.option("--modifier-threshold", type=float, default=0.80, metavar="F",
              help="수식어 비율 상한 [0,1] NNG만 적용 (기본 0.80)")
@click.option("--top-n", type=int, default=5, metavar="N",
              help="영상당 최종 키워드 수 상한 (기본 5, 버스트 스코어 순 상위 N개)")
@click.option("--min-df", type=int, default=3, metavar="N",
              help="일별 키워드 최소 문서 수 (기본 3, 미만이면 제외)")
@click.option("--burst-smooth", type=float, default=1.0, metavar="F",
              help="버스트 감지 평활화 상수 (기본 1.0). "
                   "낮을수록 희귀어 버스트 강조, 높을수록 완화")
@click.option("--emb-batch", type=int, default=256, metavar="N",
              help="임베딩 배치 크기 (GPU 메모리에 맞게 조정, 기본 256)")
@click.option("--csv", "csv_path", default="data/inputs/video_video.csv", metavar="PATH")
@click.option("--idf-out", default="data/outputs/idf.json", metavar="PATH")
@click.option("--modifier-out", default="data/outputs/word_modifier.json", metavar="PATH")
@click.option("--out", default="data/outputs/daily_keywords.json", metavar="PATH")
@click.option("--visualize-only", is_flag=False, flag_value="__all__", default=None,
              metavar="YYYY-MM-DD",
              help="워드클라우드만 생성. 날짜 미입력 시 전체 날짜 생성")
@click.option("--viz-out", default=None, metavar="PATH",
              help="워드클라우드 출력 경로 (날짜 지정 시에만 유효)")
def main(refit_idf, update_from, sim_threshold, max_df, modifier_threshold, top_n, min_df,
         burst_smooth, emb_batch, csv_path, idf_out, modifier_out, out,
         visualize_only, viz_out):
    """유튜브 영상 일별 키워드 추출 파이프라인"""

    # ── visualize-only 단축 경로 ─────────────────────────────────────────────
    if visualize_only is not None:
        from . import stage06_visualize as visualize
        if visualize_only == "__all__":
            visualize.wordcloud_all(out=viz_out)
        else:
            visualize.wordcloud_day(visualize_only, out=viz_out)
        sys.exit(0)

    CSV_PATH = _ROOT / csv_path
    IDF_PATH = _ROOT / idf_out
    MODIFIER_PATH = _ROOT / modifier_out
    OUT_PATH = _ROOT / out
    VIDEO_OUT_PATH = _ROOT / "data/outputs/video_keywords.json"

    UPDATE_FROM: str | None
    if update_from == "auto":
        if OUT_PATH.exists():
            with open(OUT_PATH, encoding="utf-8") as _f:
                _ex = json.load(_f)
            UPDATE_FROM = max(_ex.keys()) if _ex else None
            print(f"   --update-from auto → 마지막 날짜: {UPDATE_FROM}")
        else:
            UPDATE_FROM = None
            print("   --update-from auto → 기존 결과 없음, 전체 처리")
    else:
        UPDATE_FROM = update_from

    # 증분 모드에서만 기존 결과를 읽어 run_pipeline에 시드로 전달
    existing_daily: dict[str, list] = {}
    if UPDATE_FROM and OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        existing_daily = {d: p for d, p in existing.items() if d <= UPDATE_FROM}
        print(f"   기존 결과 로드: {len(existing_daily)}일 (UPDATE_FROM 이전)")

    existing_video: dict[str, list[str]] = {}
    if UPDATE_FROM and VIDEO_OUT_PATH.exists():
        with open(VIDEO_OUT_PATH, encoding="utf-8") as f:
            existing_video = json.load(f)

    t_start = time.time()
    result, video_kw = run_pipeline(
        CSV_PATH, IDF_PATH, MODIFIER_PATH,
        refit_idf=refit_idf, update_from=UPDATE_FROM,
        sim_threshold=sim_threshold, max_df=max_df, modifier_threshold=modifier_threshold,
        top_n=top_n, min_df=min_df, burst_smooth=burst_smooth, emb_batch=emb_batch,
        existing_daily=existing_daily, existing_video=existing_video,
    )

    # ── 6  저장 ───────────────────────────────────────────────────────────────────
    print("▶ 6  저장 중...")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    with open(VIDEO_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(video_kw, f, ensure_ascii=False)

    total = time.time() - t_start
    print(f"\n✓ 완료: {OUT_PATH.name}  +  {VIDEO_OUT_PATH.name}  ({len(video_kw):,}개 영상)")
    print(f"  일 범위: {min(result)} ~ {max(result)}  ({len(result)}일)")
    print(f"  총 소요: {total:.1f}s  ({total / 60:.1f}분)")

    print("\n샘플 (최근일 상위 10):")
    last_date = max(result.keys())
    for i, (w, c) in enumerate(result[last_date][:10], 1):
        print(f"  {i:2d}. {w:<14}  {c}개 문서")


if __name__ == "__main__":
    main()
