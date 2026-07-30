#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHONPATH=src
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PYTHONUNBUFFERED=1

RUN_ROOT="${RUN_ROOT:-checkpoints/eva_autsl_timesformer_t64_protocol}"
LOG_ROOT="${LOG_ROOT:-logs/eva_autsl_timesformer_t64_protocol}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

COMMON_ARGS=(
  --dataset autsl
  --n_classes 226
  --epochs 10
  --batch_size 8
  --num_workers 8
  --T 64
  --max_val_batches 200
  --val_every 1
  --log_every_batches 50
  --output "$RUN_ROOT"
)

echo "[$(date)] Starting AUTSL TimeSformer baseline"
.venv/bin/python -u -m eva_mamba3.train \
  "${COMMON_ARGS[@]}" \
  --backbone timesformer_baseline \
  --lr 1e-4 \
  --train_strides 1 \
  > "$LOG_ROOT/timesformer_baseline.log" 2>&1

echo "[$(date)] Starting AUTSL EVA-TimeSformer"
.venv/bin/python -u -m eva_mamba3.train \
  "${COMMON_ARGS[@]}" \
  --backbone eva_timesformer \
  --init_timesformer_encoder "$RUN_ROOT/timesformer_baseline/autsl/best_acc.pt" \
  --lr 2e-4 \
  --target_events 2 \
  --train_strides 1 2 4 8 16 \
  --stride_warmup_epochs 2 \
  --warmup_strides 1 \
  --robust_tds_weight 0.5 \
  > "$LOG_ROOT/eva_timesformer.log" 2>&1

echo "[$(date)] Evaluating AUTSL TimeSformer baseline"
.venv/bin/python -u -m eva_mamba3.evaluate \
  --dataset autsl \
  --n_classes 226 \
  --backbone timesformer_baseline \
  --checkpoint "$RUN_ROOT/timesformer_baseline/autsl/best_acc.pt" \
  --batch_size 8 \
  --num_workers 8 \
  --T 64 \
  --max_batches 300 \
  --no_latency \
  --output "$LOG_ROOT/timesformer_baseline_eval.json" \
  > "$LOG_ROOT/timesformer_baseline_eval.log" 2>&1

echo "[$(date)] Evaluating AUTSL EVA-TimeSformer"
.venv/bin/python -u -m eva_mamba3.evaluate \
  --dataset autsl \
  --n_classes 226 \
  --backbone eva_timesformer \
  --checkpoint "$RUN_ROOT/eva_timesformer/autsl/best_robust.pt" \
  --batch_size 8 \
  --num_workers 8 \
  --T 64 \
  --max_batches 300 \
  --baseline_json "$LOG_ROOT/timesformer_baseline_eval.json" \
  --no_latency \
  --output "$LOG_ROOT/eva_timesformer_eval.json" \
  > "$LOG_ROOT/eva_timesformer_eval.log" 2>&1

echo "[$(date)] Done"
