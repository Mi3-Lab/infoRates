#!/usr/bin/env bash
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH=src
export HF_HOME=/scratch/wesleyferreiramaia/infoRates/hf_cache
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export ACCV_JOB_ID="${ACCV_JOB_ID:-${SLURM_JOB_ID:-manual}}"

mkdir -p "${HF_HOME}" evaluations/accv2026/fixed_budget fine_tuned_models

if [[ "${WANDB_MODE}" == "online" ]]; then
  python scripts/accv2026/check_wandb_login.py
fi

CHECKPOINT="fine_tuned_models/accv2026_timesformer_ssv2_full_e3"
OUT_DIR="evaluations/accv2026/fixed_budget/timesformer_ssv2_full_e3"
SUMMARY="${OUT_DIR}/somethingv2_validation_accv2026_timesformer_ssv2_full_e3_fixed_budget_summary.csv"

echo "[h200-timesformer-full] GPU status"
nvidia-smi

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "[h200-timesformer-full] Training full SSV2 TimeSformer checkpoint"
  python scripts/accv2026/train_something.py \
    --data-root data/Something_data \
    --model timesformer \
    --epochs "${EPOCHS:-3}" \
    --batch-size "${BATCH_SIZE:-48}" \
    --lr "${LR:-2e-5}" \
    --num-workers "${NUM_WORKERS:-12}" \
    --save-path "${CHECKPOINT}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_RUN_NAME:-train-h200-timesformer-ssv2-full-e3-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 h200 ssv2 timesformer full train "job-${ACCV_JOB_ID}"
else
  echo "[h200-timesformer-full] Checkpoint already exists: ${CHECKPOINT}"
fi

if [[ ! -f "${SUMMARY}" ]]; then
  echo "[h200-timesformer-full] Evaluating fixed temporal budgets"
  python scripts/accv2026/02_run_fixed_budget_eval.py \
    --manifest evaluations/accv2026/manifests/somethingv2_val_20_per_class.csv \
    --dataset-name somethingv2 \
    --split validation \
    --checkpoint "${CHECKPOINT}" \
    --budgets 2 4 8 \
    --model-frames 8 \
    --batch-size "${EVAL_BATCH_SIZE:-32}" \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_EVAL_RUN_NAME:-eval-h200-timesformer-ssv2-full-e3-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 h200 ssv2 timesformer evaluation eval "job-${ACCV_JOB_ID}"
else
  echo "[h200-timesformer-full] Summary already exists: ${SUMMARY}"
fi

echo "[h200-timesformer-full] Computing temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
  --summary "${SUMMARY}" \
  --output "${OUT_DIR}/temporal_metrics.csv"

echo "[h200-timesformer-full] Done"
