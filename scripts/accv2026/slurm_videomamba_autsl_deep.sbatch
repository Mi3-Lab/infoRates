#!/usr/bin/env bash
#SBATCH --job-name=vmamba-autsl-deep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/vmamba-autsl-deep-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/vmamba-autsl-deep-%j.err

# VideoMamba / AUTSL — deep fine-tuning on A100 (40 GB)
#
# Fixes vs original 10-epoch run (stalled 0.5% for 7 epochs):
#   LR 5e-6→1e-5  (sqrt batch-scale: 32→64 eff. ×sqrt2≈×1.4, ~1e-5)
#   Linear warmup 3 ep + cosine decay — stabilises SSM gates early
#   30 epochs + early-stopping patience=7
#   batch=32/GPU (64 effective on 2×A100 40GB) — safe on A100
#   pin_memory=True, TF32, torch.set_float32_matmul_precision("high")
#   NO torch.compile — SSM CUDA Graphs pool ~49GB, OOM on A100 40GB

set -uo pipefail
echo "=== VideoMamba / AUTSL deep fine-tune — $(date) ==="
echo "Node: $(hostname) | GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

REPO="/data/wesleyferreiramaia/infoRates"
SCRATCH="/scratch/wesleyferreiramaia/infoRates"
DATA_ROOT="${SCRATCH}/data/AUTSL_data"
SAVE_PATH="${SCRATCH}/fine_tuned_models/accv2026_videomamba_autsl_deep_a100"

# CRITICAL: relative paths in AUTSL CSVs resolve from scratch
cd "${SCRATCH}"

source "${REPO}/.venv_mamba/bin/activate"

export WANDB_MODE=online
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1

torchrun \
    --nproc_per_node=2 \
    --master_port=29502 \
    "${REPO}/scripts/accv2026/train_videomamba.py" \
    --dataset autsl \
    --data-root "${DATA_ROOT}" \
    --save-path "${SAVE_PATH}" \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-5 \
    --warmup-epochs 3 \
    --lr-scheduler cosine \
    --early-stopping-patience 7 \
    --weight-decay 0.05 \
    --num-workers 8 \
    --input-size 224 \
    --ddp \
    --wandb-project inforates-accv2026 \
    --wandb-run-name "videomamba_autsl_a100_30ep_lr1e5_b32_cosine"

echo "=== DONE — $(date) ==="
