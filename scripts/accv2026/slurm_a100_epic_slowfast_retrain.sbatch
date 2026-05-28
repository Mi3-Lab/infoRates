#!/usr/bin/env bash
#SBATCH --job-name=epic-slowfast-retrain
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1-12:00:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
source .venv/bin/activate
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="inforates-accv2026"

CKPT="fine_tuned_models/accv2026_slowfast_r50_epic_kitchens_full_e10_a100"
CKPT_CONTAMINATED="${CKPT}_CONTAMINATED"
OUT_DIR="evaluations/accv2026/fixed_budget/slowfast_r50_epic_kitchens_full_e10_a100"
MANIFEST="evaluations/accv2026/manifests/epic_kitchens_val_20_per_class.csv"
SUMMARY="${OUT_DIR}/epic_kitchens_val_accv2026_slowfast_r50_epic_kitchens_full_e10_a100_fixed_budget_summary.csv"

echo "=== SlowFast-R50 EPIC clean retrain — job ${SLURM_JOB_ID} ==="
nvidia-smi | head -6 || true

# Retire contaminated checkpoint so training runs fresh
if [[ -d "${CKPT}" && ! -d "${CKPT_CONTAMINATED}" ]]; then
    echo "[retrain] Renaming contaminated checkpoint to ${CKPT_CONTAMINATED}"
    mv "${CKPT}" "${CKPT_CONTAMINATED}"
fi

# Remove contaminated eval results so eval re-runs after training
rm -f "${SUMMARY}" "${OUT_DIR}/"*"_fixed_budget_samples.csv" "${OUT_DIR}/temporal_metrics.csv" 2>/dev/null || true

echo "[retrain] Training SlowFast-R50 on epic_kitchens (clean split)"
python scripts/accv2026/train_slowfast.py \
    --dataset epic_kitchens \
    --epochs 10 \
    --batch-size 8 \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --num-workers 4 \
    --input-size 224 \
    --save-path "${CKPT}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "train-slowfast-r50-epic_kitchens-clean-job${SLURM_JOB_ID}" \
    --wandb-tags accv2026 a100 epic_kitchens slowfast retrain clean "job-${SLURM_JOB_ID}"

echo "[retrain] Fixed-budget eval (model-frames=32)"
python scripts/accv2026/eval_fixed_budget.py \
    --manifest "${MANIFEST}" \
    --dataset-name epic_kitchens \
    --split val \
    --checkpoint "${CKPT}" \
    --budgets 4 8 16 32 \
    --model-frames 32 \
    --batch-size 8 \
    --resize 224 \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "eval-slowfast-r50-epic_kitchens-clean-job${SLURM_JOB_ID}" \
    --wandb-tags accv2026 a100 epic_kitchens slowfast retrain clean evaluation "job-${SLURM_JOB_ID}"

python scripts/accv2026/05_compute_temporal_metrics.py \
    --summary "${SUMMARY}" \
    --output "${OUT_DIR}/temporal_metrics.csv"

echo "=== SlowFast-R50 EPIC retrain COMPLETE ==="
cat "${SUMMARY}"
