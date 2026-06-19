#!/bin/bash
# Retry UCFCrime failures with --num-workers 4 to avoid RAM OOM (16 workers × HD videos = too much)
# Also retries VideoMamba FineGym (torch.compile now disabled with "if False and")
set -euo pipefail
cd /mnt/datasets/infoRates
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

LOG_DIR="evaluations/accv2026/logs"
mkdir -p "$LOG_DIR"

run() {
    local label="$1"; shift
    echo "[$(date '+%H:%M:%S')] ▶ $label"
    python "$@" > "$LOG_DIR/${label}.log" 2>&1 \
        && echo "[$(date '+%H:%M:%S')] ✓ $label" \
        || echo "[$(date '+%H:%M:%S')] ✗ $label FAILED — $(tail -2 $LOG_DIR/${label}.log | tr '\n' ' ')"
    echo ""
}

# R2+1D UCFCrime: resume from epoch 5, 4 workers
run torchvision_r2plus1d_18_ufc_crime \
    scripts/accv2026/train_torchvision.py \
    --model r2plus1d_18 --dataset ufc_crime --epochs 10 --batch-size 32 --num-workers 4 \
    --resume-from fine_tuned_models/accv2026_r2plus1d_18_ufc_crime_full_e10_a100 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

# SlowFast UCFCrime: 4 workers
run slowfast_ufc_crime \
    scripts/accv2026/train_slowfast.py \
    --dataset ufc_crime --epochs 10 --batch-size 32 --num-workers 4 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

# TimeSformer UCFCrime: 4 workers
run transformer_timesformer_ufc_crime \
    scripts/accv2026/train_transformers.py \
    --model timesformer --dataset ufc_crime --epochs 10 --batch-size 32 --num-workers 4 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

# VideoMamba UCFCrime: 4 workers (forkserver context in script)
run videomamba_ufc_crime \
    scripts/accv2026/train_videomamba.py \
    --dataset ufc_crime --epochs 10 --batch-size 32 --num-workers 4 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

# VideoMamba FineGym: torch.compile now disabled — normal 16 workers
run videomamba_finegym \
    scripts/accv2026/train_videomamba.py \
    --dataset finegym --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion finegym

echo ""
echo "[$(date '+%H:%M:%S')] ══════════════════════════════════════"
echo "[$(date '+%H:%M:%S')] retrain_ucfcrime.sh CONCLUÍDO"
echo "[$(date '+%H:%M:%S')] ══════════════════════════════════════"
