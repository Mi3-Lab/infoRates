#!/usr/bin/env bash
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=evaluations/accv2026/logs/afa-%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/afa-%x-%j.err
# ---------------------------------------------------------------------------
# VideoMamba3-AFA — 2x L40S (48GB each), DDP
#
# Usage:
#   DATASET=ssv2 VARIANT=tiny sbatch --job-name=afa-tiny-ssv2-pretrained \
#     scripts/accv2026/slurm_l40s_videomamba3_afa_ddp.sbatch
#
# With pretrained K400 init (recommended — reaches 50-65% vs 9% from scratch):
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
VARIANT="${VARIANT:-tiny}"          # tiny cabe em L40S (44GB); small/base requer H200
EPOCHS="${EPOCHS:-40}"              # 40 = standard SSv2 fine-tune (vs 15 from-scratch)
BATCH_SIZE="${BATCH_SIZE:-2}"       # L40S 48GB; 2×2GPUs=4 efetivo
LR="${LR:-5e-5}"                    # fine-tuning LR (vs 1e-4 from scratch)
T_MAX="${T_MAX:-16}"                # 16 frames input; AFA seleciona 8 (50%)
LAMBDA_CONC="${LAMBDA_CONC:-0.5}"
LAMBDA_BUDGET="${LAMBDA_BUDGET:-0.1}"
SELECTOR_TEMP="${SELECTOR_TEMP:-1.0}"
DROP_PATH="${DROP_PATH:-0.1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}" # 5 warmup (vs 2 from scratch)
MAX_TRAIN="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL="${MAX_VAL_SAMPLES:-0}"
D_STATE="${D_STATE:-32}"
MAMBA3_VARIANT="${MAMBA3_VARIANT:-complex}"
# K400 pretrained backbone — set to "" to train from scratch
PRETRAINED="${PRETRAINED:-/data/wesleyferreiramaia/videomamba_ckpts/videomamba_t16_k400_f8_res224.pth}"

SAVE_PATH="/data/wesleyferreiramaia/infoRates/fine_tuned_models/accv2026_videomamba3_afa_${VARIANT}_${DATASET}_e${EPOCHS}_l40s"

echo "=== VideoMamba3-AFA-${VARIANT} | ${DATASET} | 2×L40S DDP | ${EPOCHS} epochs | job ${SLURM_JOB_ID} ==="
echo "=== Pretrained: ${PRETRAINED:-none} | LR=${LR} | Warmup=${WARMUP_EPOCHS} ==="
nvidia-smi | head -8 || true

# Build torchrun args — conditionally add --pretrained
PRETRAINED_ARG=""
if [ -n "${PRETRAINED}" ] && [ -f "${PRETRAINED}" ]; then
    PRETRAINED_ARG="--pretrained ${PRETRAINED}"
    echo "=== Loading K400 pretrained weights from: ${PRETRAINED} ==="
else
    echo "=== Training from scratch (no pretrained weights) ==="
fi

torchrun \
  --nproc_per_node=2 \
  --master_port=29501 \
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
    --num-workers    6 \
    --max-train-samples "${MAX_TRAIN}" \
    --max-val-samples   "${MAX_VAL}" \
    --save-path      "${SAVE_PATH}" \
    --save-every     1 \
    --ddp \
    --wandb-project  inforates-accv2026 \
    --wandb-run-name "afa-${VARIANT}-${DATASET}-l40s-k400init-e${EPOCHS}-j${SLURM_JOB_ID}" \
    --wandb-tags     accv2026 l40s ddp "${DATASET}" videomamba3-afa "${VARIANT}" "${MAMBA3_VARIANT}" k400-pretrained "job-${SLURM_JOB_ID}" \
    ${PRETRAINED_ARG}

echo "=== AFA L40S DDP DONE | ${SAVE_PATH} ==="
