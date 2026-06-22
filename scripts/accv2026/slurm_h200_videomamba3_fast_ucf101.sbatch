#!/usr/bin/env bash
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=vmamba3-fast-ucf101
#SBATCH --output=evaluations/accv2026/logs/vmamba3-fast-ucf101-%j.out
#SBATCH --error=evaluations/accv2026/logs/vmamba3-fast-ucf101-%j.err
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs
source .venv_mamba/bin/activate
export PYTHONPATH=src
export WANDB_MODE="${WANDB_MODE:-online}"
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified

DATASET="${DATASET:-ucf101}"
MODEL_SIZE="${MODEL_SIZE:-middle}"
VARIANT="${VARIANT:-complex}"
MAMBA3_IMPL="${MAMBA3_IMPL:-auto}"
DEPTH="${DEPTH:-4}"
EPOCHS="${EPOCHS:-3}"
NUM_FRAMES="${NUM_FRAMES:-4}"
INPUT_SIZE="${INPUT_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1024}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-505}"
SAVE_PATH="fine_tuned_models/accv2026_videomamba3_${MODEL_SIZE}_${VARIANT}_${MAMBA3_IMPL}_${DATASET}_${INPUT_SIZE}r_f${NUM_FRAMES}_d${DEPTH}_e${EPOCHS}_fast_h200"
EXTRA_ARGS=()
if [[ "${NO_PRETRAINED:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-pretrained)
fi

nvidia-smi || true

if [[ "${RUN_SYSTEM_VALIDATION:-1}" == "1" ]]; then
  python -u scripts/accv2026/validate_videomamba3_system.py \
    --variants "${VARIANT}" trapezoidal \
    --mamba3-impl "${MAMBA3_IMPL}" \
    --model-size "${MODEL_SIZE}" \
    --depths ${VALIDATION_DEPTHS:-"${DEPTH}"} \
    --modes inference train \
    --num-frames "${NUM_FRAMES}" \
    --input-size "${INPUT_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --ssm-d-state "${SSM_D_STATE:-16}" \
    --ssm-expand "${SSM_EXPAND:-1}" \
    --ssm-headdim "${SSM_HEADDIM:-32}" \
    --ssm-mimo-rank "${SSM_MIMO_RANK:-2}" \
    --warmup "${VALIDATION_WARMUP:-1}" \
    --iters "${VALIDATION_ITERS:-3}" \
    --output "evaluations/accv2026/videomamba3/system_validation_fast_ucf101_${SLURM_JOB_ID}.csv"
fi

python -u scripts/accv2026/train_videomamba3.py \
  --dataset "${DATASET}" \
  --model-size "${MODEL_SIZE}" \
  --depth "${DEPTH}" \
  --mamba3-variant "${VARIANT}" \
  --mamba3-impl "${MAMBA3_IMPL}" \
  --num-frames "${NUM_FRAMES}" \
  --input-size "${INPUT_SIZE}" \
  --ssm-d-state "${SSM_D_STATE:-16}" \
  --ssm-expand "${SSM_EXPAND:-1}" \
  --ssm-headdim "${SSM_HEADDIM:-32}" \
  --ssm-mimo-rank "${SSM_MIMO_RANK:-2}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR:-3e-5}" \
  --weight-decay "${WEIGHT_DECAY:-0.05}" \
  --num-workers "${NUM_WORKERS:-6}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --max-val-samples "${MAX_VAL_SAMPLES}" \
  "${EXTRA_ARGS[@]}" \
  --save-path "${SAVE_PATH}" \
  --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
  --wandb-run-name "videomamba3-${MODEL_SIZE}-${VARIANT}-${MAMBA3_IMPL}-${DATASET}-${INPUT_SIZE}r-f${NUM_FRAMES}-d${DEPTH}-e${EPOCHS}-fast-j${SLURM_JOB_ID}" \
  --wandb-tags accv2026 h200 "${DATASET}" videomamba3 "${VARIANT}" "${MAMBA3_IMPL}" "${MODEL_SIZE}" fast-validation "depth-${DEPTH}"

python scripts/accv2026/18_compile_videomamba3_results.py
