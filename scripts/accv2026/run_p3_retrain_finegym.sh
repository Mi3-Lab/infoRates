#!/usr/bin/env bash
# P3 Resolution Retraining — FineGym, local GPU (RTX PRO 6000 Blackwell 97GB)
#
# Retrains all 8 models at 4 resolutions [96, 112, 160, 224]px.
# Skips if checkpoint already exists.
# Naming convention: accv2026_{model}_finegym_{res}px_e10_h200
# (matches cluster daemon naming so is_done() picks it up on pull)
#
# Usage:
#   cd /mnt/datasets/infoRates
#   nohup bash scripts/accv2026/run_p3_retrain_finegym.sh \
#         > evaluations/accv2026/logs/p3_retrain_finegym.log 2>&1 &
#   tail -f evaluations/accv2026/logs/p3_retrain_finegym.log

set -uo pipefail
cd /mnt/datasets/infoRates
export PYTHONPATH=src
# Use local cache defaults; cluster overrides if the scratch path exists
if [[ -d "/scratch/wesleyferreiramaia" ]]; then
    export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
    export HF_HOME="${HF_HOME:-/scratch/wesleyferreiramaia/hf_unified}"
fi
# else: fall back to ~/.cache/torch and ~/.cache/huggingface (PyTorch/HF defaults)
export WANDB_PROJECT=inforates-accv2026
export WANDB_MODE=online

VENV=".venv"
CKPT_BASE="fine_tuned_models"
DATASET="finegym"
EPOCHS=10

# Resolutions: 4-point grid (96, 112, 160, 224) — 336px paused (batch_size study pending)
RESOLUTIONS=(96 112 160 224)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] P3-FineGym | $*"; }

is_done() {
    local model=$1 res=$2
    [[ -f "${CKPT_BASE}/accv2026_${model}_${DATASET}_${res}px_e${EPOCHS}_h200/config.json" ]]
}

run_cnn() {
    local model=$1 res=$2
    local ckpt="${CKPT_BASE}/accv2026_${model}_${DATASET}_${res}px_e${EPOCHS}_h200"
    is_done "$model" "$res" && { log "SKIP ${model}@${res}px (done)"; return 0; }

    # Batch size: RTX PRO 6000 has 97GB — use large batches for fast training
    local bs=256
    [[ $res -ge 224 ]] && bs=128

    log "START ${model} @ ${res}px  batch=${bs}"
    source "${VENV}/bin/activate"
    python scripts/accv2026/train_torchvision.py \
        --dataset       "${DATASET}" \
        --model         "${model}" \
        --epochs        "${EPOCHS}" \
        --batch-size    "${bs}" \
        --num-workers   12 \
        --input-size    "${res}" \
        --save-path     "${ckpt}" \
        --num-frames    16 \
        --lr            1e-4 \
        --weight-decay  0.05 \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "p3-${model}-${DATASET}-${res}px-e${EPOCHS}-local" \
        --wandb-tags    accv2026 "${DATASET}" "${model}" "${res}px" resolution-ablation spatial-aliasing local
    log "DONE  ${model} @ ${res}px"
}

run_slowfast() {
    local res=$1
    local model="slowfast_r50"
    local ckpt="${CKPT_BASE}/accv2026_${model}_${DATASET}_${res}px_e${EPOCHS}_h200"
    is_done "$model" "$res" && { log "SKIP ${model}@${res}px (done)"; return 0; }

    local bs=64
    [[ $res -ge 224 ]] && bs=32

    log "START ${model} @ ${res}px  batch=${bs}"
    source "${VENV}/bin/activate"
    python scripts/accv2026/train_slowfast.py \
        --dataset       "${DATASET}" \
        --epochs        "${EPOCHS}" \
        --batch-size    "${bs}" \
        --num-workers   12 \
        --input-size    "${res}" \
        --save-path     "${ckpt}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "p3-slowfast-${DATASET}-${res}px-e${EPOCHS}-local" \
        --wandb-tags    accv2026 "${DATASET}" slowfast "${res}px" resolution-ablation spatial-aliasing local
    log "DONE  ${model} @ ${res}px"
}

run_transformer() {
    local model=$1 res=$2
    local ckpt="${CKPT_BASE}/accv2026_${model}_${DATASET}_${res}px_e${EPOCHS}_h200"
    is_done "$model" "$res" && { log "SKIP ${model}@${res}px (done)"; return 0; }

    local bs=32
    [[ $res -le 112 ]] && bs=64

    log "START ${model} @ ${res}px  batch=${bs}"
    source "${VENV}/bin/activate"
    python scripts/accv2026/train_transformers.py \
        --dataset       "${DATASET}" \
        --model         "${model}" \
        --epochs        "${EPOCHS}" \
        --batch-size    "${bs}" \
        --num-workers   12 \
        --input-size    "${res}" \
        --save-path     "${ckpt}" \
        --lr            2e-5 \
        --weight-decay  0.05 \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "p3-${model}-${DATASET}-${res}px-e${EPOCHS}-local" \
        --wandb-tags    accv2026 "${DATASET}" "${model}" "${res}px" resolution-ablation spatial-aliasing local
    log "DONE  ${model} @ ${res}px"
}

run_videomamba() {
    local res=$1
    local model="videomamba"
    local ckpt="${CKPT_BASE}/accv2026_${model}_${DATASET}_${res}px_e${EPOCHS}_h200"
    is_done "$model" "$res" && { log "SKIP ${model}@${res}px (done)"; return 0; }

    local bs=48
    [[ $res -le 112 ]] && bs=64

    log "START ${model} @ ${res}px  batch=${bs}"
    # VideoMamba works under .venv on this machine
    source "${VENV}/bin/activate"
    python scripts/accv2026/train_videomamba.py \
        --dataset       "${DATASET}" \
        --epochs        "${EPOCHS}" \
        --batch-size    "${bs}" \
        --num-workers   12 \
        --input-size    "${res}" \
        --save-path     "${ckpt}" \
        --lr            2e-5 \
        --weight-decay  0.05 \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "p3-videomamba-${DATASET}-${res}px-e${EPOCHS}-local" \
        --wandb-tags    accv2026 "${DATASET}" videomamba "${res}px" resolution-ablation spatial-aliasing local
    log "DONE  ${model} @ ${res}px"
}

# ── Count total ────────────────────────────────────────────────────────────────
TOTAL=$(( 8 * ${#RESOLUTIONS[@]} ))
log "=== P3 FineGym Resolution Retraining ==="
log "8 models × ${#RESOLUTIONS[@]} resolutions = ${TOTAL} runs"
log "Resolutions: ${RESOLUTIONS[*]}"
log "Checkpoint naming: accv2026_{model}_finegym_{res}px_e${EPOCHS}_h200"
log ""

done_count=0
for model in r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba; do
    for res in "${RESOLUTIONS[@]}"; do
        is_done "$model" "$res" && ((done_count++)) || true
    done
done
log "Already done: ${done_count}/${TOTAL}"
log ""

# ── CNNs (native=112px) ────────────────────────────────────────────────────────
log "--- CNNs (native=112px) ---"
for res in "${RESOLUTIONS[@]}"; do
    for model in r3d_18 mc3_18 r2plus1d_18; do
        run_cnn "$model" "$res"
    done
done

# ── SlowFast (native=224px) ────────────────────────────────────────────────────
log "--- SlowFast-R50 (native=224px) ---"
for res in "${RESOLUTIONS[@]}"; do
    run_slowfast "$res"
done

# ── Transformers (native=224px) ────────────────────────────────────────────────
log "--- Transformers: TimeSformer, ViViT, VideoMAE (native=224px) ---"
for res in "${RESOLUTIONS[@]}"; do
    for model in timesformer vivit videomae; do
        run_transformer "$model" "$res"
    done
done

# ── VideoMamba (native=224px) ──────────────────────────────────────────────────
log "--- VideoMamba (native=224px) ---"
for res in "${RESOLUTIONS[@]}"; do
    run_videomamba "$res"
done

log ""
log "=== ALL DONE — $(date) ==="
log "Results in: ${CKPT_BASE}/accv2026_*_finegym_*px_e${EPOCHS}_h200/"
log "Run eval_p3_retrained.py to evaluate all retrained checkpoints"
