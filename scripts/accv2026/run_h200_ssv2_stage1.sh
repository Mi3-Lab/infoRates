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

echo "[stage1] GPU status"
nvidia-smi

echo "[stage1] Compile smoke check"
python -m compileall -q \
  src/info_rates/evaluation/benchmark.py \
  src/info_rates/metrics/temporal_robustness.py \
  scripts/accv2026/02_run_fixed_budget_eval.py \
  scripts/accv2026/05_compute_temporal_metrics.py

echo "[stage1] Evaluator smoke test. Accuracy is not a paper result."
python scripts/accv2026/02_run_fixed_budget_eval.py \
  --manifest evaluations/accv2026/manifests/somethingv2_val_5_per_class.csv \
  --dataset-name somethingv2 \
  --split validation \
  --model timesformer \
  --num-labels 174 \
  --budgets 4 8 \
  --batch-size 16 \
  --max-samples 32 \
  --output-dir evaluations/accv2026/fixed_budget/smoke_timesformer_ssv2

echo "[stage1] Train first useful SSV2 checkpoint"
python scripts/accv2026/train_something.py \
  --data-root data/Something_data \
  --model timesformer \
  --epochs 1 \
  --batch-size "${BATCH_SIZE:-32}" \
  --lr 2e-5 \
  --num-workers "${NUM_WORKERS:-8}" \
  --max-train-samples 10000 \
  --max-val-samples 2000 \
  --save-path fine_tuned_models/accv2026_timesformer_ssv2_10k_e1 \
  --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
  --wandb-run-name "${WANDB_RUN_NAME:-train-h200-timesformer-ssv2-10k-e1-job${ACCV_JOB_ID}}" \
  --wandb-tags accv2026 h200 ssv2 timesformer pilot train "job-${ACCV_JOB_ID}"

echo "[stage1] Evaluate first useful checkpoint"
python scripts/accv2026/02_run_fixed_budget_eval.py \
  --manifest evaluations/accv2026/manifests/somethingv2_val_5_per_class.csv \
  --dataset-name somethingv2 \
  --split validation \
  --checkpoint fine_tuned_models/accv2026_timesformer_ssv2_10k_e1 \
  --budgets 2 4 8 \
  --model-frames 8 \
  --batch-size "${EVAL_BATCH_SIZE:-32}" \
  --output-dir evaluations/accv2026/fixed_budget/timesformer_ssv2_10k_e1 \
  --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
  --wandb-run-name "${WANDB_EVAL_RUN_NAME:-eval-h200-timesformer-ssv2-10k-e1-job${ACCV_JOB_ID}}" \
  --wandb-tags accv2026 h200 ssv2 timesformer evaluation eval "job-${ACCV_JOB_ID}"

echo "[stage1] Compute temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
  --summary evaluations/accv2026/fixed_budget/timesformer_ssv2_10k_e1/somethingv2_validation_accv2026_timesformer_ssv2_10k_e1_fixed_budget_summary.csv \
  --output evaluations/accv2026/fixed_budget/timesformer_ssv2_10k_e1/temporal_metrics.csv

echo "[stage1] Done"
