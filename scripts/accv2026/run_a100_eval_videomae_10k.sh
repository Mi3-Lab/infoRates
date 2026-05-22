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

mkdir -p "${HF_HOME}" evaluations/accv2026/fixed_budget

if [[ "${WANDB_MODE}" == "online" ]]; then
  python scripts/accv2026/check_wandb_login.py
fi

CHECKPOINT="fine_tuned_models/accv2026_videomae_ssv2_10k_e1_a100ddp"
OUT_DIR="evaluations/accv2026/fixed_budget/videomae_ssv2_10k_e1_a100ddp"
SUMMARY="${OUT_DIR}/somethingv2_validation_accv2026_videomae_ssv2_10k_e1_a100ddp_fixed_budget_summary.csv"

echo "[a100-videomae-eval] GPU status"
nvidia-smi

echo "[a100-videomae-eval] Evaluating fixed temporal budgets"
CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python scripts/accv2026/02_run_fixed_budget_eval.py \
  --manifest evaluations/accv2026/manifests/somethingv2_val_5_per_class.csv \
  --dataset-name somethingv2 \
  --split validation \
  --checkpoint "${CHECKPOINT}" \
  --budgets 4 8 16 \
  --model-frames 16 \
  --batch-size "${EVAL_BATCH_SIZE:-12}" \
  --output-dir "${OUT_DIR}" \
  --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
  --wandb-run-name "${WANDB_EVAL_RUN_NAME:-eval-a100-videomae-ssv2-10k-e1-job${ACCV_JOB_ID}}" \
  --wandb-tags accv2026 a100 ssv2 videomae evaluation recovered eval "job-${ACCV_JOB_ID}"

echo "[a100-videomae-eval] Computing temporal metrics"
python scripts/accv2026/04_compute_temporal_metrics.py \
  --summary "${SUMMARY}" \
  --output "${OUT_DIR}/temporal_metrics.csv"

echo "[a100-videomae-eval] Done"
