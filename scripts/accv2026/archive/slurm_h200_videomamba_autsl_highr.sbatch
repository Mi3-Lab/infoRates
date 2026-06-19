#!/usr/bin/env bash
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=evaluations/accv2026/logs/vmamba-autsl-highr-%j.out
#SBATCH --error=evaluations/accv2026/logs/vmamba-autsl-highr-%j.err
set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

source .venv_mamba/bin/activate
pip install decord --quiet 2>/dev/null || true
export PYTHONPATH=src
export WANDB_MODE=online
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified

# High LR (5e-4) to break out of random-chance basin on AUTSL sign language.
# Previous run at LR=1e-4 was stuck at loss=ln(226)≈5.42 through 6+ epochs.
SAVE_PATH="fine_tuned_models/accv2026_videomamba_autsl_highr_e10_h200"

echo "=== VideoMamba AUTSL high-LR retrain (LR=5e-4) — job ${SLURM_JOB_ID} ==="
nvidia-smi | head -6 || true

python scripts/accv2026/train_videomamba.py \
    --dataset autsl \
    --epochs 10 \
    --batch-size 16 \
    --lr 5e-4 \
    --weight-decay 0.01 \
    --num-workers 4 \
    --save-path "${SAVE_PATH}" \
    --wandb-project inforates-accv2026 \
    --wandb-run-name "videomamba-autsl-highr-e10-j${SLURM_JOB_ID}" \
    --wandb-tags accv2026 h200 autsl videomamba ssm highr "job-${SLURM_JOB_ID}"

echo "=== VideoMamba AUTSL high-LR retrain DONE ==="
