#!/usr/bin/env bash
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export ACCV_JOB_ID="${ACCV_JOB_ID:-${SLURM_JOB_ID:-manual}}"

MODEL_NAME="${MODEL_NAME:-r3d_18}"
MODEL_SLUG="${MODEL_SLUG:-${MODEL_NAME//_/-}}"
TRAIN_TAG="${TRAIN_TAG:-${MODEL_SLUG}-ssv2-5k-e1}"
CHECKPOINT="${CHECKPOINT:-fine_tuned_models/accv2026_${MODEL_NAME}_ssv2_5k_e1}"
OUT_DIR="${OUT_DIR:-evaluations/accv2026/fixed_budget/${MODEL_NAME}_ssv2_5k_e1}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
SUMMARY="${OUT_DIR}/somethingv2_validation_${CHECKPOINT_NAME}_fixed_budget_summary.csv"

mkdir -p evaluations/accv2026/fixed_budget fine_tuned_models

if [[ "${WANDB_MODE}" == "online" ]]; then
  python scripts/accv2026/check_wandb_login.py
fi

echo "[a100-torchvision-pilot] GPU status"
nvidia-smi

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "[a100-torchvision-pilot] Training ${MODEL_NAME}"
  if [[ "${TRAIN_DDP:-0}" == "1" ]]; then
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" scripts/accv2026/train_torchvision.py \
      --ddp \
      --data-root data/Something_data \
      --model "${MODEL_NAME}" \
      --epochs "${EPOCHS:-1}" \
      --batch-size "${BATCH_SIZE:-16}" \
      --lr "${LR:-1e-4}" \
      --num-workers "${NUM_WORKERS:-6}" \
      --max-train-samples "${MAX_TRAIN_SAMPLES:-5000}" \
      --max-val-samples "${MAX_VAL_SAMPLES:-1000}" \
      --num-frames "${MODEL_FRAMES:-16}" \
      --input-size "${INPUT_SIZE:-112}" \
      --save-path "${CHECKPOINT}" \
      --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
      --wandb-run-name "${WANDB_RUN_NAME:-train-a100ddp-${TRAIN_TAG}-job${ACCV_JOB_ID}}" \
      --wandb-tags accv2026 a100 ssv2 "${MODEL_NAME}" 3d-cnn pilot ddp train "job-${ACCV_JOB_ID}"
  else
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU:-0}" python scripts/accv2026/train_torchvision.py \
      --data-root data/Something_data \
      --model "${MODEL_NAME}" \
      --epochs "${EPOCHS:-1}" \
      --batch-size "${BATCH_SIZE:-16}" \
      --lr "${LR:-1e-4}" \
      --num-workers "${NUM_WORKERS:-6}" \
      --max-train-samples "${MAX_TRAIN_SAMPLES:-5000}" \
      --max-val-samples "${MAX_VAL_SAMPLES:-1000}" \
      --num-frames "${MODEL_FRAMES:-16}" \
      --input-size "${INPUT_SIZE:-112}" \
      --save-path "${CHECKPOINT}" \
      --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
      --wandb-run-name "${WANDB_RUN_NAME:-train-a100-${TRAIN_TAG}-job${ACCV_JOB_ID}}" \
      --wandb-tags accv2026 a100 ssv2 "${MODEL_NAME}" 3d-cnn pilot train "job-${ACCV_JOB_ID}"
  fi
else
  echo "[a100-torchvision-pilot] Checkpoint already exists: ${CHECKPOINT}"
fi

if [[ ! -f "${SUMMARY}" ]]; then
  echo "[a100-torchvision-pilot] Evaluating fixed temporal budgets"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python scripts/accv2026/02_run_fixed_budget_eval.py \
    --manifest evaluations/accv2026/manifests/somethingv2_val_5_per_class.csv \
    --dataset-name somethingv2 \
    --split validation \
    --checkpoint "${CHECKPOINT}" \
    --budgets ${BUDGETS:-4 8 16} \
    --model-frames "${MODEL_FRAMES:-16}" \
    --batch-size "${EVAL_BATCH_SIZE:-16}" \
    --resize "${INPUT_SIZE:-112}" \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_EVAL_RUN_NAME:-eval-a100-${TRAIN_TAG}-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 a100 ssv2 "${MODEL_NAME}" 3d-cnn evaluation eval "job-${ACCV_JOB_ID}"
else
  echo "[a100-torchvision-pilot] Summary already exists: ${SUMMARY}"
fi

echo "[a100-torchvision-pilot] Computing temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
  --summary "${SUMMARY}" \
  --output "${OUT_DIR}/temporal_metrics.csv"

echo "[a100-torchvision-pilot] Done"
