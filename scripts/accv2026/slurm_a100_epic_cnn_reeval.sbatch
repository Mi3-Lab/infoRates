#!/usr/bin/env bash
#SBATCH --job-name=epic-cnn-reeval
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
export WANDB_MODE=online
export WANDB_PROJECT=inforates-accv2026
export ACCV_JOB_ID="${SLURM_JOB_ID}"

MANIFEST="evaluations/accv2026/manifests/epic_kitchens_val_20_per_class.csv"
nvidia-smi | head -6 || true

for MODEL in r3d_18 mc3_18; do
    CKPT="fine_tuned_models/accv2026_${MODEL}_epic_kitchens_full_e10_a100"
    OUT_DIR="evaluations/accv2026/fixed_budget/${MODEL}_epic_kitchens_full_e10_a100"
    SUMMARY="${OUT_DIR}/epic_kitchens_val_accv2026_${MODEL}_epic_kitchens_full_e10_a100_fixed_budget_summary.csv"

    echo "=== Reeval ${MODEL} with model-frames=16 resize=112 ==="
    # Remove stale results so eval runs fresh
    rm -f "${SUMMARY}" "${OUT_DIR}/temporal_metrics.csv"

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
        --wandb-run-name "reeval-${MODEL}-epic_kitchens-job${SLURM_JOB_ID}" \
        --wandb-tags accv2026 a100 epic_kitchens "${MODEL}" reeval

    python scripts/accv2026/05_compute_temporal_metrics.py \
        --summary "${SUMMARY}" \
        --output "${OUT_DIR}/temporal_metrics.csv"

    echo "=== ${MODEL} done ==="
    cat "${SUMMARY}"
done
