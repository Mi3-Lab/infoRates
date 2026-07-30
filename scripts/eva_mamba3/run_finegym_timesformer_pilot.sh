#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHONPATH=src
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

RUN_ROOT="checkpoints/eva_timesformer_pilot"
LOG_ROOT="logs/eva_timesformer_pilot"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

COMMON_ARGS=(
  --dataset finegym
  --n_classes 99
  --epochs 5
  --batch_size 1
  --num_workers 2
  --T 32
  --train_split val
  --max_val_batches 20
  --output "$RUN_ROOT"
)

echo "[$(date)] Starting TimeSformer baseline pilot"
.venv/bin/python -m eva_mamba3.train \
  "${COMMON_ARGS[@]}" \
  --backbone timesformer_baseline \
  --train_strides 1 \
  > "$LOG_ROOT/timesformer_baseline.log" 2>&1

echo "[$(date)] Starting EVA-TimeSformer pilot"
.venv/bin/python -m eva_mamba3.train \
  "${COMMON_ARGS[@]}" \
  --backbone eva_timesformer \
  --init_timesformer_encoder "$RUN_ROOT/timesformer_baseline/finegym/best_acc.pt" \
  --target_events 2 \
  --train_strides 1 2 4 8 16 \
  --stride_warmup_epochs 2 \
  --warmup_strides 1 \
  --robust_tds_weight 0.5 \
  > "$LOG_ROOT/eva_timesformer.log" 2>&1

echo "[$(date)] Evaluating TimeSformer baseline"
.venv/bin/python -m eva_mamba3.evaluate \
  --dataset finegym \
  --n_classes 99 \
  --backbone timesformer_baseline \
  --checkpoint "$RUN_ROOT/timesformer_baseline/finegym/best_acc.pt" \
  --batch_size 1 \
  --num_workers 2 \
  --T 32 \
  --max_batches 50 \
  --no_latency \
  --output "$LOG_ROOT/timesformer_baseline_eval.json" \
  > "$LOG_ROOT/timesformer_baseline_eval.log" 2>&1

echo "[$(date)] Evaluating EVA-TimeSformer"
.venv/bin/python -m eva_mamba3.evaluate \
  --dataset finegym \
  --n_classes 99 \
  --backbone eva_timesformer \
  --checkpoint "$RUN_ROOT/eva_timesformer/finegym/best_robust.pt" \
  --batch_size 1 \
  --num_workers 2 \
  --T 32 \
  --max_batches 50 \
  --baseline_json "$LOG_ROOT/timesformer_baseline_eval.json" \
  --no_latency \
  --output "$LOG_ROOT/eva_timesformer_eval.json" \
  > "$LOG_ROOT/eva_timesformer_eval.log" 2>&1

echo "[$(date)] Done"
