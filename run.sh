#!/bin/bash
set -e

VENV=../venv/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE=src/pipeline.py
LOG=/tmp/pipeline_run.log

cd "$SCRIPT_DIR"

[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES=$GPU
GPU_INFO="${GPU:+GPU=$GPU}"

case "$1" in

  full)
    echo "▶ 전체 파이프라인 실행 ${GPU_INFO}"
    nohup $VENV -u $PIPELINE \
      --emb-batch 1024 \
      > "$LOG" 2>&1 &
    echo "PID: $!  |  로그: $LOG"
    ;;

  update)
    echo "▶ 증분 업데이트 실행 ${GPU_INFO}"
    nohup $VENV -u $PIPELINE \
      --update-from auto --emb-batch 1024 \
      > "$LOG" 2>&1 &
    echo "PID: $!  |  로그: $LOG"
    ;;

  refit)
    echo "▶ IDF + 수식어 전체 재계산 ${GPU_INFO}"
    nohup $VENV -u $PIPELINE \
      --refit-idf --emb-batch 1024 \
      > "$LOG" 2>&1 &
    echo "PID: $!  |  로그: $LOG"
    ;;

  wordcloud)
    DATE="${2:-$(date +%Y-%m-%d)}"
    echo "▶ 워드클라우드 생성: $DATE"
    $VENV $PIPELINE --visualize-only "$DATE"
    ;;

  log)
    tail -f "$LOG"
    ;;

  *)
    echo "사용법:"
    echo "  ./run.sh full              전체 파이프라인 (백그라운드)"
    echo "  ./run.sh update            새 데이터 증분 업데이트 (백그라운드)"
    echo "  ./run.sh refit             IDF + 수식어 전체 재계산 (백그라운드)"
    echo "  ./run.sh wordcloud         오늘 날짜 워드클라우드"
    echo "  ./run.sh wordcloud DATE    특정 날짜 워드클라우드  (ex: 2026-08-09)"
    echo "  ./run.sh log               실행 로그 실시간 확인"
    echo ""
    echo "  GPU=0 ./run.sh full        GPU 번호 지정 (미설정 시 CUDA 기본 동작)"
    ;;
esac
