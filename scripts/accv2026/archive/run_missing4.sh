#!/bin/bash
# SlowFast (int64 fix) + ViViT × UCFCrime + FineGym
set -euo pipefail
cd /mnt/datasets/infoRates
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

LOG_DIR="evaluations/accv2026/logs"

run() {
    local label="$1"; shift
    echo "[$(date)] ▶ $label"
    python "$@" > "$LOG_DIR/${label}.log" 2>&1 \
        && echo "[$(date)] ✓ $label" \
        || echo "[$(date)] ✗ $label FAILED — $(tail -3 $LOG_DIR/${label}.log | tr '\n' ' ')"
    echo ""
}

run slowfast_ufc_crime \
    scripts/accv2026/train_slowfast.py \
    --dataset ufc_crime --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

run slowfast_finegym \
    scripts/accv2026/train_slowfast.py \
    --dataset finegym --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion finegym

run transformer_vivit_ufc_crime \
    scripts/accv2026/train_transformers.py \
    --dataset ufc_crime --model vivit \
    --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

run transformer_vivit_finegym \
    scripts/accv2026/train_transformers.py \
    --dataset finegym --model vivit \
    --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion finegym

echo "[$(date)] DONE — 4 jobs completos"
