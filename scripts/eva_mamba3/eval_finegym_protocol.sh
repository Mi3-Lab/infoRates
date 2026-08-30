#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHONPATH=src
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PYTHONUNBUFFERED=1

RUN_ROOT="${RUN_ROOT:-checkpoints/eva_finegym_protocol}"
LOG_ROOT="${LOG_ROOT:-logs/eva_finegym_protocol}"
MANIFEST="${MANIFEST:-evaluations/accv2026/manifests/finegym_val_20_per_class.csv}"
TIMESFORMER_MODEL="${TIMESFORMER_MODEL:-fine_tuned_models/accv2026_timesformer_finegym}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

COVERAGES=(${COVERAGES:-100})
STRIDES=(${STRIDES:-1 2 4 8 16})

echo "[$(date)] FineGym protocol baseline"
.venv/bin/python -u -m eva_mamba3.evaluate_protocol \
  --dataset finegym \
  --manifest "$MANIFEST" \
  --n_classes 99 \
  --backbone timesformer_baseline \
  --model_name "$TIMESFORMER_MODEL" \
  --mode baseline \
  --coverages "${COVERAGES[@]}" \
  --strides "${STRIDES[@]}" \
  --budget 8 \
  --batch_size 16 \
  --num_workers 8 \
  --output "$LOG_ROOT/timesformer_baseline_protocol.csv" \
  > "$LOG_ROOT/timesformer_baseline_protocol.log" 2>&1

EVA_CKPT="$RUN_ROOT/eva_timesformer/finegym/best_robust.pt"
if [[ ! -f "$EVA_CKPT" && -f "$RUN_ROOT/eva_timesformer/finegym/best_acc.pt" ]]; then
  EVA_CKPT="$RUN_ROOT/eva_timesformer/finegym/best_acc.pt"
fi

if [[ -f "$EVA_CKPT" ]]; then
  echo "[$(date)] FineGym protocol EVA-TimeSformer"
  .venv/bin/python -u -m eva_mamba3.evaluate_protocol \
    --dataset finegym \
    --manifest "$MANIFEST" \
    --n_classes 99 \
    --backbone eva_timesformer \
    --model_name "$TIMESFORMER_MODEL" \
    --checkpoint "$EVA_CKPT" \
    --mode eva_window \
    --coverages "${COVERAGES[@]}" \
    --strides "${STRIDES[@]}" \
    --budget 8 \
    --batch_size 4 \
    --num_workers 8 \
    --output "$LOG_ROOT/eva_timesformer_protocol.csv" \
    > "$LOG_ROOT/eva_timesformer_protocol.log" 2>&1
else
  echo "[$(date)] Skipping EVA eval: checkpoint not found at $RUN_ROOT/eva_timesformer/finegym/best_robust.pt or best_acc.pt"
fi
