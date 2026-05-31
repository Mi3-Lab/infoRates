#!/usr/bin/env bash
#SBATCH --job-name=cnn-224px
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
source .venv/bin/activate
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE=online
export WANDB_PROJECT=inforates-accv2026

MODEL="${MODEL:?MODEL must be set}"   # r3d_18 | mc3_18 | r2plus1d_18
DATASET="${DATASET:?DATASET must be set}"

# H200 has 141GB VRAM — use large batch for speed
BATCH_SIZE=128
NUM_WORKERS=16
EPOCHS=10

CHECKPOINT="fine_tuned_models/accv2026_${MODEL}_${DATASET}_224px_e${EPOCHS}_h200"
EVAL_MANIFEST="evaluations/accv2026/manifests/${DATASET}_val_20_per_class.csv"

echo "=== CNN@224px Retraining — ${MODEL} / ${DATASET} — job ${SLURM_JOB_ID} ==="
nvidia-smi | head -8

# Skip if checkpoint already exists
if [[ -f "${CHECKPOINT}/config.json" ]]; then
    echo "[SKIP] Checkpoint exists: ${CHECKPOINT}"
else
    echo "Training ${MODEL}@224px on ${DATASET} (batch=${BATCH_SIZE}, epochs=${EPOCHS})"
    torchrun --standalone --nproc_per_node=1 scripts/accv2026/train_torchvision.py \
        --dataset   "${DATASET}" \
        --model     "${MODEL}" \
        --epochs    "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr        1e-4 \
        --weight-decay 0.05 \
        --num-workers "${NUM_WORKERS}" \
        --num-frames 16 \
        --input-size 224 \
        --save-path  "${CHECKPOINT}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "train-${MODEL}-${DATASET}-224px-e${EPOCHS}-h200-j${SLURM_JOB_ID}" \
        --wandb-tags accv2026 h200 "${DATASET}" "${MODEL}" 224px retraining spatial-ablation
fi

echo "=== DONE — $(date) ==="
