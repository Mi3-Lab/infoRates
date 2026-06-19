#!/usr/bin/env bash
#SBATCH --job-name=epic-r2plus1d-reeval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-02:00:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err
set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
source .venv/bin/activate
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="inforates-accv2026"

# Fix: previous eval ran with --model-frames 8 (wrong for R2Plus1D).
# Training checkpoint is correct (val_acc=0.6054 on clean split). Only eval is wrong.
CKPT="fine_tuned_models/accv2026_r2plus1d_18_epic_kitchens_full_e10_a100"
OUT_DIR="evaluations/accv2026/fixed_budget/r2plus1d_18_epic_kitchens_full_e10_a100"
MANIFEST="evaluations/accv2026/manifests/epic_kitchens_val_20_per_class.csv"
SUMMARY="${OUT_DIR}/epic_kitchens_val_accv2026_r2plus1d_18_epic_kitchens_full_e10_a100_fixed_budget_summary.csv"

echo "=== R2Plus1D-18 EPIC reeval (model-frames=16) — job ${SLURM_JOB_ID} ==="
nvidia-smi | head -6 || true

# Remove wrong eval results
rm -f "${SUMMARY}" "${OUT_DIR}/"*"_fixed_budget_samples.csv" "${OUT_DIR}/temporal_metrics.csv"

python scripts/accv2026/eval_fixed_budget.py \
    --manifest "${MANIFEST}" \
    --dataset-name epic_kitchens \
    --split val \
    --checkpoint "${CKPT}" \
    --budgets 4 8 16 32 \
    --model-frames 16 \
    --batch-size 24 \
    --resize 112 \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "eval-r2plus1d_18-epic_kitchens-reeval-job${SLURM_JOB_ID}" \
    --wandb-tags accv2026 a100 epic_kitchens r2plus1d_18 reeval "job-${SLURM_JOB_ID}"

python scripts/accv2026/05_compute_temporal_metrics.py \
    --summary "${SUMMARY}" \
    --output "${OUT_DIR}/temporal_metrics.csv"

echo "=== R2Plus1D-18 EPIC reeval COMPLETE ==="
cat "${SUMMARY}"
