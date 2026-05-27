#!/usr/bin/env bash
#SBATCH --job-name=accv26-videomae-epic-reeval
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err
set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

echo "=== VideoMAE EPIC-Kitchens re-eval (clean frozen split) — job ${SLURM_JOB_ID} ==="
echo "Node: $(hostname) | $(date)"
nvidia-smi | head -6 || true

source .venv/bin/activate
export PYTHONPATH=src
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export ACCV_JOB_ID="${SLURM_JOB_ID}"
export HF_HOME="${HF_HOME:-/scratch/wesleyferreiramaia/hf_unified}"

CHECKPOINT="fine_tuned_models/accv2026_videomae_epic_kitchens_full_e10_h200"
MANIFEST="evaluations/accv2026/manifests/epic_kitchens_val_20_per_class.csv"
OUT_DIR="evaluations/accv2026/fixed_budget/videomae_epic_kitchens_full_e10_h200"
SUMMARY="${OUT_DIR}/epic_kitchens_val_accv2026_videomae_epic_kitchens_full_e10_h200_fixed_budget_summary.csv"

echo "Manifest rows: $(wc -l < ${MANIFEST})"
echo "Checkpoint: ${CHECKPOINT}"

# Remove old contaminated results so eval re-runs
rm -f "${SUMMARY}" "${OUT_DIR}/epic_kitchens_val_*_fixed_budget_samples.csv" "${OUT_DIR}/fde_cache.csv" "${OUT_DIR}/temporal_metrics.csv" 2>/dev/null || true

echo "[re-eval] Running fixed-budget evaluation..."
python scripts/accv2026/eval_fixed_budget.py \
    --manifest "${MANIFEST}" \
    --dataset-name epic_kitchens \
    --split val \
    --checkpoint "${CHECKPOINT}" \
    --budgets 4 8 16 32 \
    --model-frames 16 \
    --batch-size 24 \
    --resize 224 \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "eval-videomae-epic_kitchens-e10-job${SLURM_JOB_ID}-reeval" \
    --wandb-tags accv2026 h200 epic_kitchens videomae transformer full evaluation reeval "job-${SLURM_JOB_ID}"

echo "[re-eval] Computing temporal metrics..."
python scripts/accv2026/05_compute_temporal_metrics.py \
    --summary "${SUMMARY}" \
    --output "${OUT_DIR}/temporal_metrics.csv"

echo "=== VideoMAE EPIC re-eval COMPLETE ==="
cat "${SUMMARY}"
