#!/bin/bash
# Pipeline final com batch 32 e pin_memory=False
set -euo pipefail
cd /mnt/datasets/infoRates
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

LOG_DIR="evaluations/accv2026/logs"
mkdir -p "$LOG_DIR"

ok() { echo "[$(date)] ✓ $1"; }
fail() { echo "[$(date)] ✗ $1 FAILED"; tail -20 "$LOG_DIR/$1.log"; }

run() {
    local label="$1"; shift
    echo "[$(date)] ▶ $label"
    python "$@" > "$LOG_DIR/${label}.log" 2>&1 && ok "$label" || fail "$label"
    echo ""
}

# SlowFast — batch 32, 8 workers (batch original, pin_memory fix)
run slowfast_ufc_crime \
    scripts/accv2026/train_slowfast.py \
    --dataset ufc_crime --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

run slowfast_finegym \
    scripts/accv2026/train_slowfast.py \
    --dataset finegym --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion finegym

# Transformers — batch 32, 16 workers
for MODEL in timesformer videomae vivit; do
    for DATASET in ufc_crime finegym; do
        run "transformer_${MODEL}_${DATASET}" \
            scripts/accv2026/train_transformers.py \
            --dataset "$DATASET" --model "$MODEL" \
            --epochs 10 --batch-size 32 --num-workers 16 \
            --wandb-tags accv2026 dataset_expansion "$DATASET"
    done
done

echo "[$(date)] ══════════════════════════════════════"
echo "[$(date)] PIPELINE COMPLETO!"
echo "[$(date)] ══════════════════════════════════════"
