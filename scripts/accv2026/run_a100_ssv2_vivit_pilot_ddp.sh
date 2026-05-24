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
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export ACCV_JOB_ID="${ACCV_JOB_ID:-${SLURM_JOB_ID:-manual}}"

mkdir -p "${HF_HOME}" evaluations/accv2026/fixed_budget fine_tuned_models

if [[ "${WANDB_MODE}" == "online" ]]; then
  python scripts/accv2026/check_wandb_login.py
fi

CHECKPOINT="fine_tuned_models/accv2026_vivit_ssv2_5k_e1_a100ddp"
OUT_DIR="evaluations/accv2026/fixed_budget/vivit_ssv2_5k_e1_a100ddp"
SUMMARY="${OUT_DIR}/somethingv2_validation_accv2026_vivit_ssv2_5k_e1_a100ddp_fixed_budget_summary.csv"

echo "[a100-vivit-pilot] GPU status"
nvidia-smi

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "[a100-vivit-pilot] Training ViViT SSV2 pilot with 2-GPU DDP"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" scripts/accv2026/train_something.py \
    --ddp \
    --data-root data/Something_data \
    --model vivit \
    --epochs "${EPOCHS:-1}" \
    --batch-size "${BATCH_SIZE:-2}" \
    --lr "${LR:-2e-5}" \
    --num-workers "${NUM_WORKERS:-6}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES:-5000}" \
    --max-val-samples "${MAX_VAL_SAMPLES:-1000}" \
    --save-path "${CHECKPOINT}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_RUN_NAME:-train-a100ddp-vivit-ssv2-5k-e1-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 a100 ssv2 vivit pilot ddp train "job-${ACCV_JOB_ID}"
else
  echo "[a100-vivit-pilot] Checkpoint already exists: ${CHECKPOINT}"
fi

if [[ ! -f "${SUMMARY}" ]]; then
  echo "[a100-vivit-pilot] Evaluating fixed temporal budgets on one A100"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python scripts/accv2026/02_run_fixed_budget_eval.py \
    --manifest evaluations/accv2026/manifests/somethingv2_val_5_per_class.csv \
    --dataset-name somethingv2 \
    --split validation \
    --checkpoint "${CHECKPOINT}" \
    --budgets 4 8 16 32 \
    --model-frames 32 \
    --batch-size "${EVAL_BATCH_SIZE:-4}" \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_EVAL_RUN_NAME:-eval-a100-vivit-ssv2-5k-e1-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 a100 ssv2 vivit evaluation eval "job-${ACCV_JOB_ID}"
else
  echo "[a100-vivit-pilot] Summary already exists: ${SUMMARY}"
fi

echo "[a100-vivit-pilot] Computing temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
  --summary "${SUMMARY}" \
  --output "${OUT_DIR}/temporal_metrics.csv"

echo "[a100-vivit-pilot] Done"
