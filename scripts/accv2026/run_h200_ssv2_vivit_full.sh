#!/usr/bin/env bash
# Full ViViT SSV2 run on H200: full dataset, 5 epochs, eval on val_20_per_class.
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=src
export HF_HOME=/scratch/wesleyferreiramaia/infoRates/hf_cache
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export ACCV_JOB_ID="${ACCV_JOB_ID:-${SLURM_JOB_ID:-manual}}"

EPOCHS="${EPOCHS:-5}"
CHECKPOINT="${CHECKPOINT:-fine_tuned_models/accv2026_vivit_ssv2_full_e${EPOCHS}_h200}"
OUT_DIR="${OUT_DIR:-evaluations/accv2026/fixed_budget/vivit_ssv2_full_e${EPOCHS}_h200}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
SUMMARY="${OUT_DIR}/somethingv2_validation_${CHECKPOINT_NAME}_fixed_budget_summary.csv"

mkdir -p "${HF_HOME}" evaluations/accv2026/fixed_budget fine_tuned_models evaluations/accv2026/logs

echo "[h200-vivit-full] GPU status"
nvidia-smi

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "[h200-vivit-full] Training full SSV2 ViViT — ${EPOCHS} epochs"
  python scripts/accv2026/train_something.py \
    --data-root data/Something_data \
    --model vivit \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --lr "${LR:-1e-5}" \
    --num-workers "${NUM_WORKERS:-12}" \
    --save-path "${CHECKPOINT}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_RUN_NAME:-train-h200-vivit-ssv2-full-e${EPOCHS}-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 h200 ssv2 vivit full train "job-${ACCV_JOB_ID}"
else
  echo "[h200-vivit-full] Checkpoint already exists: ${CHECKPOINT}"
fi

if [[ ! -f "${SUMMARY}" ]]; then
  echo "[h200-vivit-full] Fixed-budget evaluation on val_20_per_class"
  python scripts/accv2026/02_run_fixed_budget_eval.py \
    --manifest evaluations/accv2026/manifests/somethingv2_val_20_per_class.csv \
    --dataset-name somethingv2 \
    --split validation \
    --checkpoint "${CHECKPOINT}" \
    --budgets ${BUDGETS:-4 8 16 32} \
    --model-frames "${MODEL_FRAMES:-32}" \
    --batch-size "${EVAL_BATCH_SIZE:-8}" \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_EVAL_RUN_NAME:-eval-h200-vivit-ssv2-full-e${EPOCHS}-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 h200 ssv2 vivit full evaluation eval "job-${ACCV_JOB_ID}"
else
  echo "[h200-vivit-full] Summary already exists: ${SUMMARY}"
fi

echo "[h200-vivit-full] Computing temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
  --summary "${SUMMARY}" \
  --output "${OUT_DIR}/temporal_metrics.csv"

echo "[h200-vivit-full] Done"
