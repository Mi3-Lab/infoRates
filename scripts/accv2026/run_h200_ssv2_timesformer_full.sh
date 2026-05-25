#!/usr/bin/env bash
# TimeSformer full SSV2 on H200: 10 epochs, batch 64, eval on val_20_per_class.
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=src
export HF_HOME=/scratch/wesleyferreiramaia/infoRates/hf_cache
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export ACCV_JOB_ID="${ACCV_JOB_ID:-${SLURM_JOB_ID:-manual}}"

EPOCHS="${EPOCHS:-10}"
CHECKPOINT="${CHECKPOINT:-fine_tuned_models/accv2026_timesformer_ssv2_full_e${EPOCHS}_h200}"
OUT_DIR="${OUT_DIR:-evaluations/accv2026/fixed_budget/timesformer_ssv2_full_e${EPOCHS}_h200}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
SUMMARY="${OUT_DIR}/somethingv2_validation_${CHECKPOINT_NAME}_fixed_budget_summary.csv"
EVAL_MANIFEST="${EVAL_MANIFEST:-evaluations/accv2026/manifests/somethingv2_val_20_per_class.csv}"

mkdir -p "${HF_HOME}" evaluations/accv2026/fixed_budget fine_tuned_models evaluations/accv2026/logs

echo "[h200-timesformer-full] GPU status"
nvidia-smi

if [[ ! -f "${CHECKPOINT}/accv_meta.json" ]]; then
  echo "[h200-timesformer-full] Training full SSV2 TimeSformer — ${EPOCHS} epochs, batch ${BATCH_SIZE:-64}"
  python scripts/accv2026/train_transformers.py \
    --data-root data/Something_data \
    --model timesformer \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --lr "${LR:-2e-5}" \
    --weight-decay "${WEIGHT_DECAY:-0.05}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --save-path "${CHECKPOINT}" \
    ${RESUME_FROM:+--resume-from "${RESUME_FROM}"} \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_RUN_NAME:-train-h200-timesformer-ssv2-full-e${EPOCHS}-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 h200 ssv2 timesformer full train "job-${ACCV_JOB_ID}"
else
  echo "[h200-timesformer-full] Checkpoint already exists: ${CHECKPOINT}"
fi

if [[ ! -f "${SUMMARY}" ]]; then
  echo "[h200-timesformer-full] Fixed-budget evaluation on val_20_per_class"
  python scripts/accv2026/02_run_fixed_budget_eval.py \
    --manifest "${EVAL_MANIFEST}" \
    --dataset-name somethingv2 \
    --split validation \
    --checkpoint "${CHECKPOINT}" \
    --budgets ${BUDGETS:-4 8 16 32} \
    --model-frames "${MODEL_FRAMES:-8}" \
    --batch-size "${EVAL_BATCH_SIZE:-32}" \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_EVAL_RUN_NAME:-eval-h200-timesformer-ssv2-full-e${EPOCHS}-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 h200 ssv2 timesformer full evaluation eval "job-${ACCV_JOB_ID}"
else
  echo "[h200-timesformer-full] Summary already exists: ${SUMMARY}"
fi

echo "[h200-timesformer-full] Computing temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
  --summary "${SUMMARY}" \
  --output "${OUT_DIR}/temporal_metrics.csv"

echo "[h200-timesformer-full] Done"
