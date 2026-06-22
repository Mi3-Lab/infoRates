#!/usr/bin/env bash
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=evaluations/accv2026/logs/vmamba3-ucf101-%j.out
#SBATCH --error=evaluations/accv2026/logs/vmamba3-ucf101-%j.err
set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

source .venv_mamba/bin/activate
pip install decord --quiet 2>/dev/null || true
export PYTHONPATH=src
export WANDB_MODE=online
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified

if [[ "${ENSURE_OFFICIAL_MAMBA3:-1}" == "1" ]]; then
  python - <<'PY' || MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall \
    git+https://github.com/state-spaces/mamba.git --no-build-isolation
from mamba_ssm.modules.mamba3 import Mamba3
print("Official Mamba3 available:", Mamba3)
PY
fi

MODEL_SIZE="${MODEL_SIZE:-middle}"
VARIANT="${VARIANT:-complex}"
MAMBA3_IMPL="${MAMBA3_IMPL:-official}"
EPOCHS="${EPOCHS:-10}"
NUM_FRAMES="${NUM_FRAMES:-8}"
INPUT_SIZE="${INPUT_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SSM_D_STATE="${SSM_D_STATE:-64}"
SSM_EXPAND="${SSM_EXPAND:-2}"
SSM_HEADDIM="${SSM_HEADDIM:-64}"
SSM_MIMO_RANK="${SSM_MIMO_RANK:-4}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
SAVE_PATH="fine_tuned_models/accv2026_videomamba3_${MODEL_SIZE}_${VARIANT}_${MAMBA3_IMPL}_ucf101_${INPUT_SIZE}r_f${NUM_FRAMES}_e${EPOCHS}_h200"

echo "=== VideoMamba3-${MODEL_SIZE}[${VARIANT}] UCF-101 training — job ${SLURM_JOB_ID} ==="
nvidia-smi | head -6 || true

python scripts/accv2026/train_videomamba3.py \
    --dataset ucf101 \
    --model-size "${MODEL_SIZE}" \
    --mamba3-variant "${VARIANT}" \
    --mamba3-impl "${MAMBA3_IMPL}" \
    --num-frames "${NUM_FRAMES}" \
    --input-size "${INPUT_SIZE}" \
    --ssm-d-state "${SSM_D_STATE}" \
    --ssm-expand "${SSM_EXPAND}" \
    --ssm-headdim "${SSM_HEADDIM}" \
    --ssm-mimo-rank "${SSM_MIMO_RANK}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr 2e-5 \
    --weight-decay 0.05 \
    --num-workers 6 \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-val-samples "${MAX_VAL_SAMPLES}" \
    --save-path "${SAVE_PATH}" \
    --wandb-project inforates-accv2026 \
    --wandb-run-name "videomamba3-${MODEL_SIZE}-${VARIANT}-${MAMBA3_IMPL}-ucf101-${INPUT_SIZE}r-f${NUM_FRAMES}-e${EPOCHS}-j${SLURM_JOB_ID}" \
    --wandb-tags accv2026 h200 ucf101 videomamba3 "${VARIANT}" "${MAMBA3_IMPL}" "${MODEL_SIZE}" full-sota "job-${SLURM_JOB_ID}"

echo "=== VideoMamba3 UCF-101 training DONE ==="
