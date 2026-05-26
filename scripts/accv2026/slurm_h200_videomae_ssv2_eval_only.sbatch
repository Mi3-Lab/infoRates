#!/usr/bin/env bash
#SBATCH --job-name=accv26-videomae-ssv2-eval
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONPATH=src
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export ACCV_JOB_ID="${SLURM_JOB_ID:-manual}"

CHECKPOINT=fine_tuned_models/accv2026_videomae_ssv2_full_e5_h200
OUT_DIR=evaluations/accv2026/fixed_budget/videomae_ssv2_full_e5_h200
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
MANIFEST=evaluations/accv2026/manifests/somethingv2_val_20_per_class.csv
SUMMARY="${OUT_DIR}/somethingv2_validation_${CHECKPOINT_NAME}_fixed_budget_summary.csv"

mkdir -p "${OUT_DIR}"

echo "[eval-only] Fixed-budget evaluation VideoMAE SSV2 e5"
nvidia-smi

if [[ ! -f "${SUMMARY}" ]]; then
  srun --ntasks=1 --cpus-per-task=8 --gpus=1 \
    python scripts/accv2026/02_run_fixed_budget_eval.py \
      --manifest "${MANIFEST}" \
      --dataset-name somethingv2 \
      --split validation \
      --checkpoint "${CHECKPOINT}" \
      --budgets 4 8 16 32 \
      --model-frames 16 \
      --batch-size 32 \
      --output-dir "${OUT_DIR}" \
      --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
      --wandb-run-name "eval-videomae-ssv2-e5-job${ACCV_JOB_ID}" \
      --wandb-tags accv2026 h200 ssv2 videomae evaluation
fi

echo "[eval-only] Computing temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
  --summary "${SUMMARY}" \
  --output "${OUT_DIR}/temporal_metrics.csv"

echo "[eval-only] Done"
cat "${OUT_DIR}/temporal_metrics.csv"
