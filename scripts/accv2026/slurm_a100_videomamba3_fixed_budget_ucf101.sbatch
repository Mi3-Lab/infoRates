#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=vmamba3-fb-ucf101
#SBATCH --output=evaluations/accv2026/logs/vmamba3-fb-ucf101-%j.out
#SBATCH --error=evaluations/accv2026/logs/vmamba3-fb-ucf101-%j.err
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs
source .venv_mamba/bin/activate
export PYTHONPATH=src
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified

CHECKPOINT="${CHECKPOINT:-fine_tuned_models/accv2026_videomamba3_middle_complex_official_ucf101_224r_f8_e10_h200}"
MANIFEST="${MANIFEST:-evaluations/accv2026/manifests/ucf101_val_20_per_class.csv}"
OUT_DIR="${OUT_DIR:-evaluations/accv2026/fixed_budget/videomamba3_middle_complex_official_ucf101_224r_f8_e10}"

python -u scripts/accv2026/eval_videomamba3_fixed_budget.py \
  --manifest "${MANIFEST}" \
  --dataset-name ucf101 \
  --split val \
  --checkpoint "${CHECKPOINT}" \
  --budgets ${BUDGETS:-4 8 16 32} \
  --coverage 100 \
  --stride 1 \
  --batch-size "${BATCH_SIZE:-8}" \
  --model-frames "${MODEL_FRAMES:-8}" \
  --resize "${RESIZE:-224}" \
  --output-dir "${OUT_DIR}" \
  --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
  --wandb-run-name "fixed-budget-videomamba3-ucf101-j${SLURM_JOB_ID}" \
  --wandb-tags accv2026 ucf101 videomamba3 fixed-budget a100

python scripts/accv2026/19_compare_videomamba3_fixed_budget.py
