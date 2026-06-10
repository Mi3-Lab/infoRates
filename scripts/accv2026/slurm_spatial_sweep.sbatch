#!/usr/bin/env bash
#SBATCH --job-name=spatial-sweep
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

# Exp 1 (Spatial Aliasing): eval-only sweep — loads native checkpoint once, tests 5 resolutions
# DATASET must be set via --export=DATASET=...
set -uo pipefail

# cd /scratch so relative video paths in manifests resolve correctly
cd /scratch/wesleyferreiramaia/infoRates
export PYTHONPATH=/data/wesleyferreiramaia/infoRates/src
export TORCH_HOME=/scratch/wesleyferreiramaia/infoRates/torch_cache
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export WANDB_MODE=online
export WANDB_PROJECT=inforates-accv2026

DATASET="${DATASET:?DATASET must be set}"
SCRIPT=/data/wesleyferreiramaia/infoRates/scripts/accv2026/sweep_spatial_resolution.py

echo "=== Spatial Sweep — dataset=${DATASET} — job ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true

# CNN + Transformer models (use .venv)
source /data/wesleyferreiramaia/infoRates/.venv/bin/activate

for MODEL in r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae; do
    echo ""
    echo "--- ${MODEL}/${DATASET} ---"
    python "${SCRIPT}" \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --batch-size 32 || echo "[WARN] ${MODEL}/${DATASET} failed — continuing"
done

deactivate

# VideoMamba uses separate venv
source /data/wesleyferreiramaia/infoRates/.venv_mamba/bin/activate

echo ""
echo "--- videomamba/${DATASET} ---"
python "${SCRIPT}" \
    --model videomamba \
    --dataset "${DATASET}" \
    --batch-size 16 || echo "[WARN] videomamba/${DATASET} failed — continuing"

deactivate

echo ""
echo "=== DONE — $(date) ==="
