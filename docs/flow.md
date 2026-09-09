# word_extractor 흐름도

각 다이어그램은 노드 7개 이하로 제한해 단계별로 쪼갬.

## 1. 실행 진입점 (run.sh → pipeline.py)

```mermaid
flowchart LR
    A[run.sh] --> B{명령어}
    B -->|full| C["pipeline.py<br>--emb-batch 1024"]
    B -->|update| D["pipeline.py<br>--update-from auto"]
    B -->|refit| E["pipeline.py<br>--refit-idf"]
    B -->|wordcloud| F["pipeline.py<br>--visualize-only"]
    B -->|log| G[tail pipeline_run.log]
```

## 2. 파이프라인 전체 단계 (pipeline.py)

```mermaid
flowchart LR
    S1["1 CSV 로딩"] --> S2["2 형태소 분석 + 수식어"]
    S2 --> S3["3 IDF"]
    S3 --> S4["4 임베딩"]
    S4 --> S5["5 키워드 추출"]
    S5 --> S6["6 저장"]
```

## 3. Stage 1 · CSV 로딩 & 전처리 (01_preprocess.py)

```mermaid
flowchart LR
    A[CSV 읽기] --> B[clean_text: URL/# 제거]
    B --> C[strip_boilerplate: 저작권/타임코드 제거]
    C --> D[combined_text: title+description]
    D --> E{update_from 지정?}
    E -->|예| F[date > update_from 필터]
    E -->|아니오| G[전체 문서 사용]
```

## 4. Stage 2a · 형태소 태깅 (02_tokenize.py)

```mermaid
flowchart LR
    A[Kiwi 배치 tokenize] --> B{NNG/NNP/SL 인가?}
    B -->|아니오| C[토큰 폐기]
    B -->|예| D["(token, tag) 유지"]
    D --> E{다음 토큰도 NNG/NNP/SL?}
    E -->|예| F[modifier_count 증가]
    E -->|아니오, 조사/어미| G[head_count 증가]
```

## 4b. Stage 2b · 수식어 비율 산출

```mermaid
flowchart LR
    A["modifier_count, head_count"] --> B{"합계 >= 50?"}
    B -->|아니오| C[집계 제외]
    B -->|예| D["ratio = mod/(mod+head)"]
    D --> E[word_modifier.json 저장]
```

## 5. Stage 3 · IDF 계산/로드/증분 (03_idf.py)

```mermaid
flowchart TD
    A{idf.json 상태} -->|refit_idf 또는 미존재| B[fit: 전체 df 집계]
    A -->|update_from 지정 + 존재| C[update: 신규 df 병합]
    A -->|그 외| D[load: 그대로 읽기]
    B --> E["idf = log((N+1)/(df+1))+1"]
    C --> E
    E --> F[idf.json 저장]
    D --> G["idf, n_docs, df_counts 반환"]
    F --> G
```

## 6. Stage 5 · 문서별 키워드 필터 체인 (pipeline.py + 04_extract.py)

```mermaid
flowchart LR
    A[weighted_tf] --> B["score_keywords: TF x IDF"]
    B --> C[max_df 필터]
    C --> D["modifier 필터<br>(NNG만 적용)"]
    D --> E["Reranker.rerank<br>(sim_threshold)"]
    E --> F[burst 점수로 재정렬]
    F --> G[top-N 키워드]
```

burst 공식: `score / (df/n_docs + burst_smooth/n_docs)` — df가 낮을수록(희귀할수록) 상위.

## 6b. Stage 5 상세 · Reranker.rerank (05_rerank.py)

```mermaid
flowchart LR
    A["pool(top-N)/tail 분리"] --> B[문서·후보 단어 임베딩]
    B --> C[cosine similarity 계산]
    C --> D{"sim >= threshold?"}
    D -->|아니오| E[제거]
    D -->|예| F["정규화 후 alpha 보정<br>score * factor"]
    F --> G[재정렬 + tail 합치기]
```

## 7. Stage 6 · 결과 저장 (pipeline.py)

```mermaid
flowchart LR
    A[영상별 top_kws] --> B[daily_count 집계]
    A --> C[video_kw 딕셔너리]
    B --> D["most_common 30<br>count >= min_df"]
    D --> E[daily_keywords.json]
    C --> F[video_keywords.json]
```

## 8. 워드클라우드 생성 (06_visualize.py)

```mermaid
flowchart LR
    A["--visualize-only"] --> B{날짜 지정?}
    B -->|특정 날짜| C[wordcloud_day]
    B -->|미지정=all| D[wordcloud_all]
    C --> E[wordcloud_day.html.jinja 렌더]
    D --> F[wordcloud_all.html.jinja 렌더]
    E --> G[figures/*.html 저장]
    F --> G
```

## 9. 산출물 맵

```mermaid
flowchart LR
    A[video_video.csv] --> P[pipeline.py]
    P --> B[idf.json]
    P --> C[word_modifier.json]
    P --> D[daily_keywords.json]
    P --> E[video_keywords.json]
    D --> F[figures/*.html]
```
