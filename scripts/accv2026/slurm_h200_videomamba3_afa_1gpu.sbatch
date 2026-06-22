#!/usr/bin/env bash
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=evaluations/accv2026/logs/afa-%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/afa-%x-%j.err
# ---------------------------------------------------------------------------
# VideoMamba3-AFA training — 1 GPU H200 (sem DDP, mais rápido de escalonar)
#
# Usage:
#   DATASET=ucf101 VARIANT=small sbatch --job-name=afa-small-ucf101 \
#     scripts/accv2026/slurm_h200_videomamba3_afa_1gpu.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

source .venv_mamba/bin/activate
pip install decord --quiet 2>/dev/null || true

export PYTHONPATH=src:experiments/videomamba3
export WANDB_MODE=online
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export WANDB_DIR=/scratch/wesleyferreiramaia/infoRates/wandb
mkdir -p "${WANDB_DIR}"

if [[ "${ENSURE_OFFICIAL_MAMBA3:-1}" == "1" ]]; then
  python - <<'PY' 2>/dev/null || MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall \
    git+https://github.com/state-spaces/mamba.git --no-build-isolation
from mamba_ssm.modules.mamba3 import Mamba3
print("Official Mamba3 available:", Mamba3)
PY
fi

DATASET="${DATASET:-ucf101}"
VARIANT="${VARIANT:-small}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
T_MAX="${T_MAX:-48}"
LAMBDA_CONC="${LAMBDA_CONC:-0.5}"
LAMBDA_BUDGET="${LAMBDA_BUDGET:-0.1}"
SELECTOR_TEMP="${SELECTOR_TEMP:-1.0}"
DROP_PATH="${DROP_PATH:-0.1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
MAX_TRAIN="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL="${MAX_VAL_SAMPLES:-0}"

SAVE_PATH="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models/accv2026_videomamba3_afa_${VARIANT}_${DATASET}_e${EPOCHS}_h200"

echo "=== VideoMamba3-AFA-${VARIANT} | ${DATASET} | 1 GPU | ${EPOCHS} epochs | job ${SLURM_JOB_ID} ==="
nvidia-smi | head -6 || true

python scripts/accv2026/train_videomamba3_afa.py \
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
    --num-workers    6 \
    --max-train-samples "${MAX_TRAIN}" \
    --max-val-samples   "${MAX_VAL}" \
    --save-path      "${SAVE_PATH}" \
    --save-every     1 \
    --wandb-project  inforates-accv2026 \
    --wandb-run-name "afa-${VARIANT}-${DATASET}-1gpu-e${EPOCHS}-j${SLURM_JOB_ID}" \
    --wandb-tags     accv2026 h200 "${DATASET}" videomamba3-afa "${VARIANT}" 1gpu "job-${SLURM_JOB_ID}"

echo "=== AFA training DONE | ${SAVE_PATH} ==="
