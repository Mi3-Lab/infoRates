#!/usr/bin/env bash
# SlowFast R50 pilot: 5k SSV2 samples, 1 epoch, 2xA100 DDP.
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

EPOCHS="${EPOCHS:-1}"
CHECKPOINT="${CHECKPOINT:-fine_tuned_models/accv2026_slowfast_r50_ssv2_5k_e1_a100}"
OUT_DIR="${OUT_DIR:-evaluations/accv2026/fixed_budget/slowfast_r50_ssv2_5k_e1_a100}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
SUMMARY="${OUT_DIR}/somethingv2_validation_${CHECKPOINT_NAME}_fixed_budget_summary.csv"

mkdir -p evaluations/accv2026/fixed_budget fine_tuned_models evaluations/accv2026/logs

echo "[a100-slowfast-pilot] GPU status"
nvidia-smi

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "[a100-slowfast-pilot] Training SlowFast R50 SSV2 5k pilot with 2-GPU DDP"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" scripts/accv2026/04_train_slowfast.py \
    --ddp \
    --data-root data/Something_data \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --lr "${LR:-1e-4}" \
    --num-workers "${NUM_WORKERS:-6}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES:-5000}" \
    --max-val-samples "${MAX_VAL_SAMPLES:-1000}" \
    --input-size 224 \
    --save-path "${CHECKPOINT}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_RUN_NAME:-train-a100ddp-slowfast-r50-ssv2-5k-e1-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 a100 ssv2 slowfast pilot ddp train "job-${ACCV_JOB_ID}"
else
  echo "[a100-slowfast-pilot] Checkpoint already exists: ${CHECKPOINT}"
fi

if [[ ! -f "${SUMMARY}" ]]; then
  echo "[a100-slowfast-pilot] Fixed-budget evaluation"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python scripts/accv2026/02_run_fixed_budget_eval.py \
    --manifest evaluations/accv2026/manifests/somethingv2_val_5_per_class.csv \
    --dataset-name somethingv2 \
    --split validation \
    --checkpoint "${CHECKPOINT}" \
    --budgets ${BUDGETS:-4 8 16 32} \
    --model-frames 32 \
    --batch-size "${EVAL_BATCH_SIZE:-4}" \
    --resize 224 \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_EVAL_RUN_NAME:-eval-a100-slowfast-r50-ssv2-5k-e1-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 a100 ssv2 slowfast evaluation eval "job-${ACCV_JOB_ID}"
else
  echo "[a100-slowfast-pilot] Summary already exists: ${SUMMARY}"
fi

echo "[a100-slowfast-pilot] Computing temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
  --summary "${SUMMARY}" \
  --output "${OUT_DIR}/temporal_metrics.csv"

echo "[a100-slowfast-pilot] Done"
