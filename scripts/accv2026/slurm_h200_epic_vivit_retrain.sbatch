#!/usr/bin/env bash
#SBATCH --job-name=epic-vivit-retrain
#SBATCH --partition=cenvalarc.gpu
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
export HF_HOME="${HF_HOME:-/scratch/wesleyferreiramaia/hf_unified}"
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="inforates-accv2026"

CKPT="fine_tuned_models/accv2026_vivit_epic_kitchens_full_e10_h200"
CKPT_CONTAMINATED="${CKPT}_CONTAMINATED"
OUT_DIR="evaluations/accv2026/fixed_budget/vivit_epic_kitchens_full_e10_h200"
MANIFEST="evaluations/accv2026/manifests/epic_kitchens_val_20_per_class.csv"
SUMMARY="${OUT_DIR}/epic_kitchens_val_accv2026_vivit_epic_kitchens_full_e10_h200_fixed_budget_summary.csv"

echo "=== ViViT EPIC clean retrain — job ${SLURM_JOB_ID} ==="
nvidia-smi | head -6 || true

# Retire contaminated checkpoint so training runs fresh
if [[ -d "${CKPT}" && ! -d "${CKPT_CONTAMINATED}" ]]; then
    echo "[retrain] Renaming contaminated checkpoint to ${CKPT_CONTAMINATED}"
    mv "${CKPT}" "${CKPT_CONTAMINATED}"
fi

# Remove contaminated eval results so eval re-runs after training
rm -f "${SUMMARY}" "${OUT_DIR}/"*"_fixed_budget_samples.csv" "${OUT_DIR}/temporal_metrics.csv" 2>/dev/null || true

echo "[retrain] Training ViViT on epic_kitchens (clean split)"
python scripts/accv2026/train_transformers.py \
    --dataset epic_kitchens \
    --model vivit \
    --epochs 10 \
    --batch-size 8 \
    --lr 2e-5 \
    --weight-decay 0.05 \
    --num-workers 4 \
    --save-path "${CKPT}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "train-vivit-epic_kitchens-clean-job${SLURM_JOB_ID}" \
    --wandb-tags accv2026 h200 epic_kitchens vivit retrain clean "job-${SLURM_JOB_ID}"

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
    --wandb-run-name "eval-vivit-epic_kitchens-clean-job${SLURM_JOB_ID}" \
    --wandb-tags accv2026 h200 epic_kitchens vivit retrain clean evaluation "job-${SLURM_JOB_ID}"

python scripts/accv2026/05_compute_temporal_metrics.py \
    --summary "${SUMMARY}" \
    --output "${OUT_DIR}/temporal_metrics.csv"

echo "=== ViViT EPIC retrain COMPLETE ==="
cat "${SUMMARY}"
