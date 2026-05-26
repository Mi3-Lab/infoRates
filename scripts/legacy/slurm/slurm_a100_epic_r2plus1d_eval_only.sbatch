#!/usr/bin/env bash
#SBATCH --job-name=accv26-a100-epic-r2-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

# Run fixed-budget evaluation for EPIC-Kitchens R2+1D using best available checkpoint.
set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONPATH=src
export WANDB_MODE="${WANDB_MODE:-online}"
export ACCV_JOB_ID="${SLURM_JOB_ID:-manual}"

DATASET="epic_kitchens"
EPOCHS=10
MANIFEST="evaluations/accv2026/manifests/epic_kitchens_val_20_per_class.csv"
OUT_DIR="evaluations/accv2026/fixed_budget/r2plus1d_18_epic_kitchens_full_e${EPOCHS}_a100"
SUMMARY="${OUT_DIR}/epic_kitchens_val_accv2026_r2plus1d_18_epic_kitchens_full_e${EPOCHS}_a100_fixed_budget_summary.csv"

# Pick best checkpoint by val_acc from config.json
BEST_CKPT=""
BEST_ACC=-1
for epoch in $(seq 1 10); do
  ckpt="fine_tuned_models/accv2026_r2plus1d_18_epic_kitchens_full_e${EPOCHS}_a100_epoch${epoch}"
  cfg="${ckpt}/config.json"
  if [[ -f "$cfg" ]]; then
    acc=$(python3 -c "import json; print(json.load(open('${cfg}')).get('val_acc', 0))" 2>/dev/null || echo 0)
    if python3 -c "exit(0 if float('${acc}') > float('${BEST_ACC}') else 1)" 2>/dev/null; then
      BEST_ACC="$acc"
      BEST_CKPT="$ckpt"
    fi
  fi
done

if [[ -z "$BEST_CKPT" ]]; then
  echo "[epic-r2-eval] No checkpoint found. Exiting."
  exit 1
fi
echo "[epic-r2-eval] Using checkpoint: $BEST_CKPT (val_acc=${BEST_ACC})"

mkdir -p "$OUT_DIR"

if [[ ! -f "$SUMMARY" ]]; then
  echo "[epic-r2-eval] Running fixed-budget evaluation"
  python scripts/accv2026/02_run_fixed_budget_eval.py \
    --manifest "$MANIFEST" \
    --dataset-name "$DATASET" \
    --split "val" \
    --checkpoint "$BEST_CKPT" \
    --budgets 4 8 16 32 \
    --model-frames 16 \
    --batch-size 32 \
    --resize 112 \
    --output-dir "$OUT_DIR" \
    --wandb-project "inforates-accv2026" \
    --wandb-run-name "eval-r2plus1d-epic-best-epoch-job${ACCV_JOB_ID}" \
    --wandb-tags accv2026 a100 epic_kitchens r2plus1d torchvision full evaluation
else
  echo "[epic-r2-eval] Summary already exists: $SUMMARY"
fi

echo "[epic-r2-eval] Computing temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
  --summary "$SUMMARY" \
  --output "${OUT_DIR}/temporal_metrics.csv" 2>/dev/null || true

echo "[epic-r2-eval] Done."
