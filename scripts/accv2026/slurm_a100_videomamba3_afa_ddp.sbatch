#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=evaluations/accv2026/logs/afa-%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/afa-%x-%j.err
# ---------------------------------------------------------------------------
# VideoMamba3-AFA — 2x A100 (80GB each), DDP
#
# Usage:
#   DATASET=ssv2 VARIANT=tiny sbatch --job-name=afa-tiny-ssv2-k400 \
#     scripts/accv2026/slurm_a100_videomamba3_afa_ddp.sbatch
#
# With pretrained K400 init (recommended):
#   PRETRAINED=/data/wesleyferreiramaia/videomamba_ckpts/videomamba_t16_k400_f8_res224.pth \
#   DATASET=ssv2 VARIANT=tiny EPOCHS=40 LR=5e-5 WARMUP_EPOCHS=5 \
#     sbatch --job-name=afa-tiny-ssv2-k400init ...
# ---------------------------------------------------------------------------
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

source .venv_mamba/bin/activate
pip install decord --quiet 2>/dev/null || true

export PYTHONPATH=src:experiments/videomamba3
export WANDB_MODE=online
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export WANDB_DIR=/tmp/wandb_${SLURM_JOB_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p "${WANDB_DIR}"

python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba OK')" 2>/dev/null || echo "mamba rodando em modo PyTorch puro"

DATASET="${DATASET:-ssv2}"
VARIANT="${VARIANT:-tiny}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-4}"       # A100 80GB — 4×2GPUs=8 efetivo (2× vs L40S)
LR="${LR:-5e-5}"
T_MAX="${T_MAX:-16}"
LAMBDA_CONC="${LAMBDA_CONC:-0.5}"
LAMBDA_BUDGET="${LAMBDA_BUDGET:-0.1}"
SELECTOR_TEMP="${SELECTOR_TEMP:-1.0}"
DROP_PATH="${DROP_PATH:-0.1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
MAX_TRAIN="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL="${MAX_VAL_SAMPLES:-0}"
D_STATE="${D_STATE:-32}"            # A100 80GB — pode usar d_state=32 (vs 16 no L40S)
MAMBA3_VARIANT="${MAMBA3_VARIANT:-complex}"
PRETRAINED="${PRETRAINED:-/data/wesleyferreiramaia/videomamba_ckpts/videomamba_t16_k400_f8_res224.pth}"

SAVE_PATH="/data/wesleyferreiramaia/infoRates/fine_tuned_models/accv2026_videomamba3_afa_${VARIANT}_${DATASET}_e${EPOCHS}_a100"

echo "=== VideoMamba3-AFA-${VARIANT} | ${DATASET} | 2×A100 DDP | ${EPOCHS} epochs | job ${SLURM_JOB_ID} ==="
echo "=== Pretrained: ${PRETRAINED:-none} | LR=${LR} | Warmup=${WARMUP_EPOCHS} | batch=${BATCH_SIZE}/GPU ==="
nvidia-smi | head -8 || true

PRETRAINED_ARG=""
if [ -n "${PRETRAINED}" ] && [ -f "${PRETRAINED}" ]; then
    PRETRAINED_ARG="--pretrained ${PRETRAINED}"
    echo "=== Loading K400 pretrained weights from: ${PRETRAINED} ==="
else
    echo "=== Training from scratch (no pretrained weights) ==="
fi

torchrun \
  --nproc_per_node=2 \
  --master_port=29502 \
  scripts/accv2026/train_videomamba3_afa.py \
    --dataset        "${DATASET}" \
    --variant        "${VARIANT}" \
    --T-max          "${T_MAX}" \
    --epochs         "${EPOCHS}" \
    --batch-size     "${BATCH_SIZE}" \
    --lr             "${LR}" \
    --weight-decay   0.05 \
    --warmup-epochs  "${WARMUP_EPOCHS}" \
    --lambda-conc    "${LAMBDA_CONC}" \
    --lambda-budget  "${LAMBDA_BUDGET}" \
    --selector-temperature "${SELECTOR_TEMP}" \
    --drop-path-rate "${DROP_PATH}" \
    --d-state        "${D_STATE}" \
    --mamba3-variant "${MAMBA3_VARIANT}" \
    --num-workers    8 \
    --max-train-samples "${MAX_TRAIN}" \
    --max-val-samples   "${MAX_VAL}" \
    --save-path      "${SAVE_PATH}" \
    --save-every     1 \
    --ddp \
    --wandb-project  inforates-accv2026 \
    --wandb-run-name "afa-${VARIANT}-${DATASET}-a100-k400init-e${EPOCHS}-j${SLURM_JOB_ID}" \
    --wandb-tags     accv2026 a100 ddp "${DATASET}" videomamba3-afa "${VARIANT}" "${MAMBA3_VARIANT}" k400-pretrained "job-${SLURM_JOB_ID}" \
    ${PRETRAINED_ARG}

echo "=== AFA A100 DDP DONE | ${SAVE_PATH} ==="
