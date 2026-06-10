#!/bin/bash
# Final clean training: SlowFast + Transformers on UCFCrime + FineGym only
# No FLAME, no parallelism overload
set -euo pipefail
cd /mnt/datasets/infoRates
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

LOG_DIR="evaluations/accv2026/logs"
mkdir -p "$LOG_DIR"

run_job() {
    local label="$1"; shift
    local logfile="$LOG_DIR/${label}.log"
    echo "[$(date)] ▶ $label"
    if python "$@" >"$logfile" 2>&1; then
        echo "[$(date)] ✓ $label"
    else
        echo "[$(date)] ✗ $label FAILED"
        tail -20 "$logfile"
    fi
    echo ""
}

# ─────────────────────────────────────────────────────────
# SlowFast: UCFCrime + FineGym (8 workers, safer)
# ─────────────────────────────────────────────────────────
for DATASET in ufc_crime finegym; do
    run_job "slowfast_${DATASET}" \
        scripts/accv2026/train_slowfast.py \
        --dataset "$DATASET" \
        --epochs 10 --batch-size 32 --num-workers 8 \
        --wandb-tags accv2026 dataset_expansion "$DATASET"
done

# ─────────────────────────────────────────────────────────
# Transformers: TimeSformer, VideoMAE, ViViT on UCFCrime + FineGym
# ─────────────────────────────────────────────────────────
for MODEL in timesformer videomae vivit; do
    for DATASET in ufc_crime finegym; do
        run_job "transformer_${MODEL}_${DATASET}" \
            scripts/accv2026/train_transformers.py \
            --dataset "$DATASET" --model "$MODEL" \
            --epochs 10 --batch-size 32 --num-workers 8 \
            --wandb-tags accv2026 dataset_expansion "$DATASET"
    done
done

echo "[$(date)] ══════════════════════════════════════════════"
echo "[$(date)] ALL TRAINING COMPLETE"
echo "[$(date)] ══════════════════════════════════════════════"
