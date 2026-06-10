#!/usr/bin/env bash
# P3 Resolution Retraining — FineGym, local GPU (RTX PRO 6000 Blackwell 97GB)
#
# Retrains all 8 models at 5 resolutions [96, 112, 160, 224, 336]px.
# Skips if checkpoint already exists.
# Naming convention: accv2026_{model}_finegym_{res}px_e10_h200
# (matches cluster daemon naming so is_done() picks it up on pull)
#
# ── 336px batch size strategy ─────────────────────────────────────────────────
# The cluster A100 (40GB) collapsed at 336px with batch=16-24 because BatchNorm
# statistics become too noisy at small batch sizes. With 97GB here we can afford:
#   CNNs           bs=128  → prevents BatchNorm collapse (need ≥64)
#   SlowFast       bs=48   → dual-path but 97GB handles it
#   Transformers   bs=24   → LayerNorm (robust), but 336px = 441 patches → memory
#   VideoMamba     bs=32   → SSM linear, no O(n²) attention
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

# Full 5-point grid; 336px uses large batches to prevent BatchNorm collapse
RESOLUTIONS=(96 112 160 224 336)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] P3-FineGym | $*"; }

is_done() {
    local model=$1 res=$2
    [[ -f "${CKPT_BASE}/accv2026_${model}_${DATASET}_${res}px_e${EPOCHS}_h200/config.json" ]]
}

run_cnn() {
    local model=$1 res=$2
    local ckpt="${CKPT_BASE}/accv2026_${model}_${DATASET}_${res}px_e${EPOCHS}_h200"
    is_done "$model" "$res" && { log "SKIP ${model}@${res}px (done)"; return 0; }

    # Batch size for CNNs (BatchNorm — never drop below 64 to prevent collapse):
    #   96/112/160px → 256  (small input, fits easily)
    #   224px        → 128
    #   336px        → 128  (keep large! cluster A100 collapsed at bs=16-24)
    local bs=256
    [[ $res -ge 224 ]] && bs=128
    # 336px: explicitly enforce bs=128 (same tier as 224px — do NOT reduce further)
    # This is the fix for the cluster collapse: small bs → noisy BN stats → degenerate model

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

    # SlowFast: dual-path architecture is memory-heavy but 97GB handles well
    # 336px: bs=48 keeps enough samples for stable training
    local bs=64
    [[ $res -ge 224 ]] && bs=48
    [[ $res -ge 336 ]] && bs=48  # explicit floor at 336px

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

    # Transformers (TimeSformer/ViViT/VideoMAE): LayerNorm — less sensitive to batch size,
    # but 336px → 441 patches per frame (vs 196 at 224px) — memory scales linearly
    # 336px: bs=24 keeps ~25GB/GPU well under 97GB ceiling
    local bs=32
    [[ $res -le 112 ]] && bs=64
    [[ $res -ge 336 ]] && bs=24

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

    # VideoMamba: SSM is linear in sequence length (no O(n²) attention)
    # 336px is safe at bs=32 — plenty of room in 97GB
    local bs=48
    [[ $res -le 112 ]] && bs=64
    [[ $res -ge 336 ]] && bs=32

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
log "336px batch sizes: CNNs=128, SlowFast=48, Transformers=24, VideoMamba=32"
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
