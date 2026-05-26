#!/usr/bin/env bash
#SBATCH --job-name=accv26-k400-videomae-eval
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

# Fixed-budget eval for Kinetics-400 using pre-trained VideoMAE (no fine-tuning).
# Model: MCG-NJU/videomae-base-finetuned-kinetics (~81% top1 on K400).
set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONPATH=src
export WANDB_MODE="${WANDB_MODE:-online}"
export HF_HOME="/data/wesleyferreiramaia/hf_cache"
export ACCV_JOB_ID="${SLURM_JOB_ID:-manual}"

DATASET="kinetics400"
MANIFEST="evaluations/accv2026/manifests/kinetics400_val_20_per_class.csv"
CHECKPOINT="MCG-NJU/videomae-base-finetuned-kinetics"
OUT_DIR="evaluations/accv2026/fixed_budget/videomae_kinetics400_pretrained_h200"
SUMMARY="${OUT_DIR}/kinetics400_val_accv2026_videomae_kinetics400_pretrained_h200_fixed_budget_summary.csv"

mkdir -p "$OUT_DIR"

if [[ ! -f "$SUMMARY" ]]; then
  echo "[k400-eval] Running fixed-budget evaluation (VideoMAE pre-trained, no fine-tuning)"
  echo "[k400-eval] Checkpoint: $CHECKPOINT"
  echo "[k400-eval] Manifest:   $MANIFEST (8000 videos, 400 classes)"
  python scripts/accv2026/02_run_fixed_budget_eval.py \
    --manifest      "$MANIFEST" \
    --dataset-name  "$DATASET" \
    --split         "val" \
    --checkpoint    "$CHECKPOINT" \
    --budgets       4 8 16 32 \
    --model-frames  16 \
    --batch-size    16 \
    --resize        224 \
    --output-dir    "$OUT_DIR" \
    --wandb-project "inforates-accv2026" \
    --wandb-run-name "eval-videomae-k400-pretrained-job${ACCV_JOB_ID}" \
    --wandb-tags    accv2026 h200 kinetics400 videomae pretrained evaluation
else
  echo "[k400-eval] Summary already exists: $SUMMARY"
fi

echo "[k400-eval] Computing temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
  --summary "$SUMMARY" \
  --output  "${OUT_DIR}/temporal_metrics.csv" 2>/dev/null || true

echo "[k400-eval] Done."
