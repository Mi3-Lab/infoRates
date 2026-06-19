#!/bin/bash
# Re-run dos jobs que falharam no retrain_all.sh
# Fixes aplicados: decord em VideoMambaDataset, torch.compile desabilitado no VideoMamba
set -euo pipefail
cd /mnt/datasets/infoRates
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

LOG_DIR="evaluations/accv2026/logs"

run() {
    local label="$1"; shift
    echo "[$(date '+%H:%M:%S')] ▶ $label"
    python "$@" > "$LOG_DIR/${label}.log" 2>&1 \
        && echo "[$(date '+%H:%M:%S')] ✓ $label" \
        || echo "[$(date '+%H:%M:%S')] ✗ $label FAILED — $(tail -2 $LOG_DIR/${label}.log | tr '\n' ' ')"
    echo ""
}

# VideoMamba × ambos os datasets (fix: decord + sem torch.compile)
run videomamba_ufc_crime \
    scripts/accv2026/train_videomamba.py \
    --dataset ufc_crime --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

run videomamba_finegym \
    scripts/accv2026/train_videomamba.py \
    --dataset finegym --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion finegym

# UCFCrime pendentes (fix: decord em UCFDataset)
run torchvision_r2plus1d_18_ufc_crime \
    scripts/accv2026/train_torchvision.py \
    --model r2plus1d_18 --dataset ufc_crime --epochs 10 --batch-size 32 --num-workers 16 \
    --resume-from fine_tuned_models/accv2026_r2plus1d_18_ufc_crime_full_e10_a100 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

run slowfast_ufc_crime \
    scripts/accv2026/train_slowfast.py \
    --dataset ufc_crime --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

run transformer_timesformer_ufc_crime \
    scripts/accv2026/train_transformers.py \
    --model timesformer --dataset ufc_crime --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

echo ""
echo "[$(date '+%H:%M:%S')] retrain_failed.sh CONCLUÍDO"
