# word_extractor

유튜브 영상 title + description에서 일별 핵심 키워드를 추출하는 파이프라인.

## 파이프라인 구조

```
video_video.csv
      │
      ▼
01_preprocess  →  텍스트 정제 (URL 제거, 상투문구 제거, 불용어 목록)
      │
      ▼
02_tokenize    →  형태소 분석 (kiwipiepy) → NNG / NNP / SL 토큰
      │
      ▼
03_idf         →  코퍼스 전체 IDF 계산 → idf.json 저장/로드
      │
      ▼
04_extract     →  TF-IDF 스코어링 → 문서당 상위 N 키워드
      │
      ▼ (선택)
05_rerank      →  문서 임베딩 유사도 보정 (ko-sroberta-multitask)
      │
      ▼
daily_keywords.json
```

## 실행

```bash
# 기본 실행 (전체 파이프라인)
python run_pipeline.py

# IDF 재계산 (불용어 변경 후 등)
python run_pipeline.py --refit-idf

# 임베딩 재순위 적용 (느림, 품질 향상)
python run_pipeline.py --use-rerank

# 문서당 추출 키워드 수 변경
python run_pipeline.py --top-n 15

# 경로 커스텀
python run_pipeline.py --csv data.csv --idf-out idf.json --out daily_keywords.json
```

## 출력 형식

`daily_keywords.json`

```json
{
  "2026-08-09": [["폭염", 12], ["트럼프", 8], ["태풍", 7], ...],
  "2026-08-08": [["경선", 10], ["코스피", 6], ...],
  ...
}
```

- 키: `YYYY-MM-DD`
- 값: `[키워드, 해당_날짜_문서_수]` 리스트, 상위 50개

## 모듈별 설명

| 파일 | 역할 | 주요 함수 |
|------|------|-----------|
| `01_preprocess.py` | 텍스트 정제, 불용어 관리 | `clean_text`, `strip_boilerplate`, `combined_text`, `DEFAULT_STOPWORDS` |
| `02_tokenize.py` | 형태소 분석 | `tokenize(text, stopwords)`, `weighted_tf(tagged)` |
| `03_idf.py` | IDF 계산/저장/로드 | `fit(token_seqs)`, `save`, `load`, `unseen_idf` |
| `04_extract.py` | TF-IDF 스코어링 | `score_keywords(tf, idf, n_docs)`, `top_keywords` |
| `05_rerank.py` | 임베딩 유사도 재순위 | `Reranker().rerank(doc_text, ranked)` |
| `run_pipeline.py` | 전체 파이프라인 실행 | — |

## 불용어 관리

**`01_preprocess.py`의 `DEFAULT_STOPWORDS`** 가 정본.  
불용어 추가 후 `--refit-idf` 옵션으로 재실행하면 idf.json도 갱신됨.

```python
# 01_preprocess.py
DEFAULT_STOPWORDS: set[str] = {
    # 유튜브 UI / 포맷 용어
    "Shorts", "zip", "자막", ...
    # 방송사 채널명 / 뉴스 프로그램명
    "MBC뉴스", "YTN", "JTBC", ...
}
```

## 데이터

| 파일 | 크기 | 내용 |
|------|------|------|
| `video_video.csv` | 272 MB | 유튜브 영상 259,583개 (2014-04 ~ 2026-08) |
| `idf.json` | — | 어휘 74,805개 IDF 값 |
| `daily_keywords.json` | ~2.4 MB | 3,520일 × 상위 50 키워드 |

## 소요 시간 (259K 문서 기준)

| 단계 | 시간 |
|------|------|
| IDF 학습 | ~9분 |
| 키워드 추출 | ~9분 |
| **전체** | **~18분** |

※ `idf.json` 존재 시 IDF 학습 생략 → ~9분으로 단축
