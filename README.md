# word_extractor

유튜브 영상 title + description에서 일별·영상별 핵심 키워드를 추출하는 파이프라인.

## 실행

```bash
./run.sh full              # 전체 파이프라인 (백그라운드)
./run.sh update            # 증분 업데이트 — 마지막 날짜 자동 감지
./run.sh refit             # IDF + 수식어 전체 재계산
./run.sh wordcloud         # 오늘 날짜 워드클라우드
./run.sh wordcloud DATE    # 특정 날짜  (ex: 2026-08-09)
./run.sh log               # 백그라운드 로그 실시간 확인
```

## 파이프라인

```
 1  CSV 로딩        날짜 파싱, 증분 필터
 2  형태소 분석     kiwi 배치 — NNG/NNP/SL 추출 + 수식어 비율 계산  ①
 3  IDF             전체 학습 / 로드 / 증분 갱신 → idf.json
 4  임베딩          ko-sroberta-multitask
 5  키워드 추출     TF-IDF → max_df · modifier · sim 필터 → rerank
 6  저장            daily_keywords.json + video_keywords.json
```

① 수식어 비율: 파일 없거나 `refit`이면 계산 후 word_modifier.json 저장. 이후 실행에서는 로드만.

## 필터 (Stage 5)

| 필터 | 기준 | 제거 예시 |
|---|---|---|
| max_df | 전체 문서 5% 초과 | 전국, 서울 |
| modifier | NNG 수식어 비율 > 0.80 | 역대(0.982), 긴급(0.975) |
| sim | doc-keyword cosine < 0.20 | 문서와 무관한 단어 |

## 파일 구성

```
word_extractor/
├── run.sh
└── src/
    ├── pipeline.py       메인 파이프라인
    ├── 01_preprocess.py  텍스트 정제, 불용어
    ├── 02_tokenize.py    형태소 분석 (kiwipiepy)
    ├── 03_idf.py         IDF 계산 / 저장 / 로드
    ├── 04_extract.py     TF-IDF 스코어링
    ├── 05_rerank.py      임베딩 유사도 재순위
    └── 06_visualize.py   워드클라우드 HTML 생성

data/outputs/
├── idf.json              어휘 IDF 값
├── word_modifier.json    단어별 수식어 비율
├── daily_keywords.json   날짜별 상위 키워드
├── video_keywords.json   영상별 top-10 키워드
└── figures/              날짜별 워드클라우드 HTML
```

## 출력 형식

**`daily_keywords.json`**
```json
{
  "2026-08-09": [["폭염", 12], ["태풍", 8], ["경선", 7]],
  "2026-08-08": [["경선", 10], ["코스피", 6]]
}
```

**`video_keywords.json`**
```json
{
  "video_id": ["폭염", "태풍", "강원", "경보"]
}
```

## 불용어 관리

`src/01_preprocess.py`의 `DEFAULT_STOPWORDS`가 정본.  
수정 후 `./run.sh refit`으로 재실행하면 idf.json · word_modifier.json 모두 갱신됨.
