#!/usr/bin/env bash
#SBATCH --job-name=retrain-h200
#SBATCH --partition=cenvalarc.gpu
#SBATCH --constraint=H200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

# 1x H200 (141GB) com batch grande >> 2x L40s (48GB) com batch=16
# SSv2: batch=128 → 168k/128 = 1313 steps/epoch → ~1.5h vs ~9.7h (6x mais rápido)
set -uo pipefail
cd /scratch/wesleyferreiramaia/infoRates
export PYTHONPATH=/data/wesleyferreiramaia/infoRates/src
export TORCH_HOME=/scratch/wesleyferreiramaia/infoRates/torch_cache
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export WANDB_MODE=online
export WANDB_PROJECT=inforates-accv2026

MODEL="${MODEL:?MODEL must be set}"
DATASET="${DATASET:?DATASET must be set}"
INPUT_SIZE="${INPUT_SIZE:?INPUT_SIZE must be set}"
EPOCHS=10

SCRATCH_DATA="/scratch/wesleyferreiramaia/infoRates/data"
declare -A DATA_ROOTS=(
    [ucf101]="${SCRATCH_DATA}/UCF101_data"
    [ssv2]="${SCRATCH_DATA}/Something_data"
    [hmdb51]="${SCRATCH_DATA}/HMDB51_data"
    [diving48]="${SCRATCH_DATA}/Diving48_data"
    [autsl]="${SCRATCH_DATA}/AUTSL_data"
    [driveact]="${SCRATCH_DATA}/DriveAct_data"
    [epic_kitchens]="${SCRATCH_DATA}/EPIC_data"
)
DATA_ROOT="${DATA_ROOTS[$DATASET]:-${SCRATCH_DATA}}"
CKPT_BASE="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models"
CHECKPOINT="${CKPT_BASE}/accv2026_${MODEL}_${DATASET}_${INPUT_SIZE}px_e${EPOCHS}_h200"

echo "=== H200 Single-GPU Retrain: ${MODEL} / ${DATASET} @ ${INPUT_SIZE}px — job ${SLURM_JOB_ID} ==="
GPU_MEM=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
echo "GPU: ${GPU_MEM}"

if [[ -f "${CHECKPOINT}/config.json" ]]; then
    VAL=$(python3 -c "import json; d=json.load(open('${CHECKPOINT}/config.json')); print(d.get('val_acc',0))" 2>/dev/null)
    EP=$(python3  -c "import json; d=json.load(open('${CHECKPOINT}/config.json')); print(d.get('epoch',0))"    2>/dev/null)
    if python3 -c "import sys; sys.exit(0 if float('${VAL}')>0 and int('${EP}')>=9 else 1)" 2>/dev/null; then
        echo "[SKIP] Checkpoint completo (ep=${EP}, val=${VAL})"
        exit 0
    else
        echo "[CONTINUE] Checkpoint incompleto (ep=${EP}) — continuando treino"
    fi
fi

# H200 141GB — batch máximo sem DDP (single GPU mais eficiente que DDP neste caso)
# Comparação de velocidade (SSv2, 168k amostras):
#   2×L40s batch=16: 5278 steps/epoch → ~9.7h
#   1×H200 batch=128: 1313 steps/epoch → ~1.5h  (6x faster)
if [[ "$MODEL" == "r3d_18" || "$MODEL" == "mc3_18" || "$MODEL" == "r2plus1d_18" ]]; then
    source /data/wesleyferreiramaia/infoRates/.venv/bin/activate
    export TORCH_COMPILE_DISABLE=1
    [[ "$INPUT_SIZE" -le 160 ]] && BATCH=128 || ([[ "$INPUT_SIZE" -le 224 ]] && BATCH=96 || BATCH=64)

    echo "Batch: ${BATCH} (single H200, ${INPUT_SIZE}px)"
    python /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_torchvision.py \
        --dataset "${DATASET}" --data-root "${DATA_ROOT}" --model "${MODEL}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH}" --num-workers 16 \
        --input-size "${INPUT_SIZE}" --save-path "${CHECKPOINT}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "retrain-${MODEL}-${DATASET}-${INPUT_SIZE}px-h200s-j${SLURM_JOB_ID}" \
        --wandb-tags accv2026 h200 "${DATASET}" "${MODEL}" "${INPUT_SIZE}px" spatial-aliasing \
        --lr 1e-4 --weight-decay 0.05 --num-frames 16

elif [[ "$MODEL" == "slowfast_r50" ]]; then
    source /data/wesleyferreiramaia/infoRates/.venv/bin/activate
    [[ "$INPUT_SIZE" -le 160 ]] && BATCH=96 || ([[ "$INPUT_SIZE" -le 224 ]] && BATCH=64 || BATCH=48)

    echo "Batch: ${BATCH} (single H200, ${INPUT_SIZE}px)"
    python /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_slowfast.py \
        --dataset "${DATASET}" --data-root "${DATA_ROOT}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH}" --num-workers 16 \
        --input-size "${INPUT_SIZE}" --save-path "${CHECKPOINT}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "retrain-slowfast-${DATASET}-${INPUT_SIZE}px-h200s-j${SLURM_JOB_ID}" \
        --wandb-tags accv2026 h200 "${DATASET}" slowfast "${INPUT_SIZE}px" spatial-aliasing

elif [[ "$MODEL" == "videomamba" ]]; then
    source /data/wesleyferreiramaia/infoRates/.venv_mamba/bin/activate
    [[ "$INPUT_SIZE" -le 224 ]] && BATCH=64 || BATCH=48

    echo "Batch: ${BATCH} (single H200, ${INPUT_SIZE}px)"
    python /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_videomamba.py \
        --dataset "${DATASET}" --data-root "${DATA_ROOT}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH}" --num-workers 16 \
        --input-size "${INPUT_SIZE}" --save-path "${CHECKPOINT}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "retrain-videomamba-${DATASET}-${INPUT_SIZE}px-h200s-j${SLURM_JOB_ID}" \
        --wandb-tags accv2026 h200 "${DATASET}" videomamba "${INPUT_SIZE}px" spatial-aliasing \
        --lr 2e-5 --weight-decay 0.05

else
    # timesformer, vivit, videomae
    source /data/wesleyferreiramaia/infoRates/.venv/bin/activate
    [[ "$INPUT_SIZE" -le 224 ]] && BATCH=48 || BATCH=32

    echo "Batch: ${BATCH} (single H200, ${INPUT_SIZE}px)"
    python /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_transformers.py \
        --dataset "${DATASET}" --data-root "${DATA_ROOT}" --model "${MODEL}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH}" --num-workers 16 \
        --input-size "${INPUT_SIZE}" --save-path "${CHECKPOINT}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "retrain-${MODEL}-${DATASET}-${INPUT_SIZE}px-h200s-j${SLURM_JOB_ID}" \
        --wandb-tags accv2026 h200 "${DATASET}" "${MODEL}" "${INPUT_SIZE}px" spatial-aliasing \
        --lr 2e-5 --weight-decay 0.05
fi

echo "=== DONE — $(date) ==="
