#!/bin/bash
# Complete retraining pipeline — all models × UCFCrime + FineGym for 10 epochs
# Skips: R3D/MC3 UCFCrime (done), VideoMAE FineGym (done, best at epoch 6)
# Fixes: SlowFast uses spawn context, ViViT uses batch_size=8
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

# ── TorchVision CNNs × FineGym (1 epoch each before → retrain for 10) ──────
run torchvision_r3d_18_finegym \
    scripts/accv2026/train_torchvision.py \
    --model r3d_18 --dataset finegym --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion finegym

run torchvision_mc3_18_finegym \
    scripts/accv2026/train_torchvision.py \
    --model mc3_18 --dataset finegym --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion finegym

run torchvision_r2plus1d_18_finegym \
    scripts/accv2026/train_torchvision.py \
    --model r2plus1d_18 --dataset finegym --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion finegym

# ── R2+1D UCFCrime — resume from epoch 5 ────────────────────────────────────
run torchvision_r2plus1d_18_ufc_crime \
    scripts/accv2026/train_torchvision.py \
    --model r2plus1d_18 --dataset ufc_crime --epochs 10 --batch-size 32 --num-workers 16 \
    --resume-from fine_tuned_models/accv2026_r2plus1d_18_ufc_crime_full_e10_a100 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

# ── SlowFast UCFCrime (spawn context fix) ───────────────────────────────────
run slowfast_ufc_crime \
    scripts/accv2026/train_slowfast.py \
    --dataset ufc_crime --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

# ── TimeSformer × both datasets ──────────────────────────────────────────────
run transformer_timesformer_ufc_crime \
    scripts/accv2026/train_transformers.py \
    --model timesformer --dataset ufc_crime --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

run transformer_timesformer_finegym \
    scripts/accv2026/train_transformers.py \
    --model timesformer --dataset finegym --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion finegym

# ── VideoMAE UCFCrime (FineGym already done at epoch 6 best) ─────────────────
run transformer_videomae_ufc_crime \
    scripts/accv2026/train_transformers.py \
    --model videomae --dataset ufc_crime --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

# ── VideoMamba × both datasets ───────────────────────────────────────────────
run videomamba_ufc_crime \
    scripts/accv2026/train_videomamba.py \
    --dataset ufc_crime --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

run videomamba_finegym \
    scripts/accv2026/train_videomamba.py \
    --dataset finegym --epochs 10 --batch-size 32 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion finegym

# ── ViViT × both datasets — batch 8 (attention matrix too large at 32) ───────
run transformer_vivit_ufc_crime \
    scripts/accv2026/train_transformers.py \
    --model vivit --dataset ufc_crime --epochs 10 --batch-size 8 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion ufc_crime

run transformer_vivit_finegym \
    scripts/accv2026/train_transformers.py \
    --model vivit --dataset finegym --epochs 10 --batch-size 8 --num-workers 16 \
    --wandb-tags accv2026 dataset_expansion finegym

echo ""
echo "[$(date '+%H:%M:%S')] ══════════════════════════════════════"
echo "[$(date '+%H:%M:%S')] PIPELINE COMPLETO!"
echo "[$(date '+%H:%M:%S')] ══════════════════════════════════════"
