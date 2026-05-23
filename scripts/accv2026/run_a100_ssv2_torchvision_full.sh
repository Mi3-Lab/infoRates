#!/usr/bin/env bash
# Paper-quality TorchVision 3D CNN run: full SSV2, 5 epochs, eval on val_20_per_class.
# Override with env vars before calling; model-specific wrappers set defaults.
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export ACCV_JOB_ID="${ACCV_JOB_ID:-${SLURM_JOB_ID:-manual}}"

MODEL_NAME="${MODEL_NAME:-r2plus1d_18}"
MODEL_SLUG="${MODEL_SLUG:-r2plus1d-18}"
EPOCHS="${EPOCHS:-5}"
TRAIN_TAG="${TRAIN_TAG:-${MODEL_SLUG}-ssv2-full-e${EPOCHS}}"
CHECKPOINT="${CHECKPOINT:-fine_tuned_models/accv2026_${MODEL_NAME}_ssv2_full_e${EPOCHS}_a100}"
OUT_DIR="${OUT_DIR:-evaluations/accv2026/fixed_budget/${MODEL_NAME}_ssv2_full_e${EPOCHS}_a100}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
SUMMARY="${OUT_DIR}/somethingv2_validation_${CHECKPOINT_NAME}_fixed_budget_summary.csv"
EVAL_MANIFEST="${EVAL_MANIFEST:-evaluations/accv2026/manifests/somethingv2_val_20_per_class.csv}"

mkdir -p evaluations/accv2026/fixed_budget fine_tuned_models evaluations/accv2026/logs

echo "[a100-torchvision-full] GPU status"
nvidia-smi

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "[a100-torchvision-full] Training ${MODEL_NAME} — full SSV2, ${EPOCHS} epochs, 2xA100 DDP"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" scripts/accv2026/train_torchvision.py \
    --ddp \
    --data-root data/Something_data \
    --model "${MODEL_NAME}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE:-24}" \
    --lr "${LR:-5e-5}" \
    --weight-decay "${WEIGHT_DECAY:-0.05}" \
    --num-workers "${NUM_WORKERS:-8}" \
    --num-frames "${MODEL_FRAMES:-16}" \
    --input-size "${INPUT_SIZE:-112}" \
    --save-path "${CHECKPOINT}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_RUN_NAME:-train-a100ddp-${TRAIN_TAG}-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 a100 ssv2 "${MODEL_NAME}" 3d-cnn full ddp train "job-${ACCV_JOB_ID}"
else
  echo "[a100-torchvision-full] Checkpoint already exists: ${CHECKPOINT}"
fi

if [[ ! -f "${SUMMARY}" ]]; then
  echo "[a100-torchvision-full] Fixed-budget evaluation on val_20_per_class"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python scripts/accv2026/02_run_fixed_budget_eval.py \
    --manifest "${EVAL_MANIFEST}" \
    --dataset-name somethingv2 \
    --split validation \
    --checkpoint "${CHECKPOINT}" \
    --budgets ${BUDGETS:-4 8 16 32} \
    --model-frames "${MODEL_FRAMES:-16}" \
    --batch-size "${EVAL_BATCH_SIZE:-24}" \
    --resize "${INPUT_SIZE:-112}" \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_EVAL_RUN_NAME:-eval-a100-${TRAIN_TAG}-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 a100 ssv2 "${MODEL_NAME}" 3d-cnn full evaluation eval "job-${ACCV_JOB_ID}"
else
  echo "[a100-torchvision-full] Summary already exists: ${SUMMARY}"
fi

echo "[a100-torchvision-full] Computing temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
  --summary "${SUMMARY}" \
  --output "${OUT_DIR}/temporal_metrics.csv"

echo "[a100-torchvision-full] Done — ${MODEL_NAME} full run complete"
