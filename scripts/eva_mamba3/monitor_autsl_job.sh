#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG_DIR="${1:-logs/eva_autsl_timesformer_b6000}"
RUN_ROOT="${2:-checkpoints/$(basename "$LOG_DIR")}"
OUT="${LOG_DIR}/monitor.log"
INTERVAL="${MONITOR_INTERVAL:-180}"

mkdir -p "$LOG_DIR"

while true; do
  {
    echo "================================================================"
    echo "timestamp=$(date -Is)"

    if [[ -f "$LOG_DIR/job.pid" ]]; then
      pid="$(cat "$LOG_DIR/job.pid")"
      echo "--- job"
      ps -p "$pid" -o pid,stat,etime,cmd || true
    fi

    echo "--- gpu"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw,power.limit \
      --format=csv,noheader || true

    echo "--- driver"
    tail -20 "$LOG_DIR/driver.outer.log" 2>/dev/null || true

    echo "--- baseline metrics"
    grep -E '^\[[[:space:]]*[0-9]+/[0-9]+\]' "$LOG_DIR/timesformer_baseline.log" 2>/dev/null | tail -10 || true

    echo "--- eva metrics"
    grep -E '^\[[[:space:]]*[0-9]+/[0-9]+\]' "$LOG_DIR/eva_timesformer.log" 2>/dev/null | tail -10 || true

    echo "--- eval json"
    for f in "$LOG_DIR"/*_eval.json; do
      [[ -f "$f" ]] || continue
      echo "### $f"
      cat "$f"
      echo
    done

    echo "--- checkpoints"
    find "$RUN_ROOT" -maxdepth 4 -type f \( -name '*.pt' -o -name 'progress.json' \) \
      -printf '%TY-%Tm-%Td %TH:%TM %p %s bytes\n' 2>/dev/null | sort || true
    echo "--- progress"
    find "$RUN_ROOT" -maxdepth 4 -type f -name 'progress.json' -print -exec cat {} \; 2>/dev/null || true

    echo "--- finegym"
    ps -p "$(cat logs/dataset_downloads/finegym_download.pid 2>/dev/null)" \
      -o pid,stat,etime,cmd 2>/dev/null || true
    printf 'finegym_train_mp4='
    find data/FineGym_data/videos/train -type f -name '*.mp4' 2>/dev/null | wc -l
    printf 'finegym_val_mp4='
    find data/FineGym_data/videos/val -type f -name '*.mp4' 2>/dev/null | wc -l
  } >> "$OUT"

  sleep "$INTERVAL"
done
