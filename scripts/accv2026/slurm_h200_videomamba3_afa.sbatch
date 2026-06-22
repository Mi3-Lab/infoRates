#!/usr/bin/env bash
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=evaluations/accv2026/logs/afa-%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/afa-%x-%j.err
# ---------------------------------------------------------------------------
# VideoMamba3-AFA training on H200 (2 GPUs, DDP)
#
# Usage:
#   # small variant, ucf101, 15 epochs
#   DATASET=ucf101 VARIANT=small sbatch --job-name=afa-small-ucf101 scripts/accv2026/slurm_h200_videomamba3_afa.sbatch
#
#   # tiny variant, hmdb51, quick ablation
#   DATASET=hmdb51 VARIANT=tiny EPOCHS=10 sbatch --job-name=afa-tiny-hmdb51 scripts/accv2026/slurm_h200_videomamba3_afa.sbatch
#
#   # base variant, ssv2, full SOTA run
#   DATASET=ssv2 VARIANT=base EPOCHS=20 sbatch --job-name=afa-base-ssv2 scripts/accv2026/slurm_h200_videomamba3_afa.sbatch
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

# verify or build official Mamba3
if [[ "${ENSURE_OFFICIAL_MAMBA3:-1}" == "1" ]]; then
  python - <<'PY' 2>/dev/null || MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall \
    git+https://github.com/state-spaces/mamba.git --no-build-isolation
from mamba_ssm.modules.mamba3 import Mamba3
print("Official Mamba3 available:", Mamba3)
PY
fi

# ── Job parameters ─────────────────────────────────────────────────────────
DATASET="${DATASET:-ucf101}"
VARIANT="${VARIANT:-small}"         # tiny | small | base
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-4}"       # per-GPU batch; effective = BATCH_SIZE × 2 GPUs
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

echo "=== VideoMamba3-AFA-${VARIANT} | ${DATASET} | ${EPOCHS} epochs | job ${SLURM_JOB_ID} ==="
echo "    T_max=${T_MAX}  λ_conc=${LAMBDA_CONC}  λ_budget=${LAMBDA_BUDGET}  τ=${SELECTOR_TEMP}"
nvidia-smi | head -8 || true

torchrun \
  --nproc_per_node=2 \
  --master_port=29500 \
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
    --num-workers    6 \
    --max-train-samples "${MAX_TRAIN}" \
    --max-val-samples   "${MAX_VAL}" \
    --save-path      "${SAVE_PATH}" \
    --save-every     1 \
    --ddp \
    --wandb-project  inforates-accv2026 \
    --wandb-run-name "afa-${VARIANT}-${DATASET}-e${EPOCHS}-j${SLURM_JOB_ID}" \
    --wandb-tags     accv2026 h200 "${DATASET}" videomamba3-afa "${VARIANT}" "job-${SLURM_JOB_ID}"

echo "=== VideoMamba3-AFA training DONE | saved to ${SAVE_PATH} ==="
