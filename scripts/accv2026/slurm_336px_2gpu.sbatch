#!/usr/bin/env bash
#SBATCH --job-name=retrain-336px
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE=online
export WANDB_PROJECT=inforates-accv2026

MODEL="${MODEL:?MODEL must be set}"
DATASET="${DATASET:?DATASET must be set}"
INPUT_SIZE=336
EPOCHS=10

source .venv/bin/activate

CHECKPOINT="fine_tuned_models/accv2026_${MODEL}_${DATASET}_336px_e${EPOCHS}_h200"

echo "=== 336px 2-GPU Retrain: ${MODEL} / ${DATASET} — job ${SLURM_JOB_ID} ==="
nvidia-smi | head -10 || true

if [[ -f "${CHECKPOINT}/config.json" ]]; then
    echo "[SKIP] Checkpoint exists: ${CHECKPOINT}"
    exit 0
fi

if [[ "$MODEL" == "r3d_18" || "$MODEL" == "mc3_18" || "$MODEL" == "r2plus1d_18" ]]; then
    # 2× L40s (48GB each) — batch=64/GPU = 128 total, DDP
    export TORCH_COMPILE_DISABLE=1
    torchrun --nproc_per_node=2 scripts/accv2026/train_torchvision.py \
        --dataset   "${DATASET}" \
        --model     "${MODEL}" \
        --epochs    "${EPOCHS}" \
        --batch-size 64 \
        --num-workers 8 \
        --input-size "${INPUT_SIZE}" \
        --save-path  "${CHECKPOINT}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "retrain-${MODEL}-${DATASET}-336px-2gpu-j${SLURM_JOB_ID}" \
        --wandb-tags accv2026 cenvalarc "${DATASET}" "${MODEL}" 336px 2gpu spatial-aliasing \
        --lr 1e-4 --weight-decay 0.05 --num-frames 16 \
        --ddp

elif [[ "$MODEL" == "slowfast_r50" ]]; then
    torchrun --nproc_per_node=2 scripts/accv2026/train_slowfast.py \
        --dataset "${DATASET}" \
        --epochs "${EPOCHS}" \
        --batch-size 32 \
        --num-workers 8 \
        --input-size "${INPUT_SIZE}" \
        --save-path "${CHECKPOINT}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "retrain-slowfast-${DATASET}-336px-2gpu-j${SLURM_JOB_ID}" \
        --wandb-tags accv2026 cenvalarc "${DATASET}" slowfast 336px 2gpu spatial-aliasing \
        --ddp
fi

echo "=== DONE — $(date) ==="
