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

# Wait for CUDA memory to be fully released between processes.
# CUDAGraph private pools can hold 10-20GB after process exit; without a sleep
# the next process hits OOM immediately (seen: R2+1D@160px crash, 7GB alloc, 2GB free).
cuda_drain() {
    sleep 30
}

run_cnn() {
    local model=$1 res=$2
    local ckpt="${CKPT_BASE}/accv2026_${model}_${DATASET}_${res}px_e${EPOCHS}_h200"
    is_done "$model" "$res" && { log "SKIP ${model}@${res}px (done)"; return 0; }

    # Batch size for CNNs (BatchNorm — never drop below 64 to prevent collapse):
    #   R3D/MC3   96-160px → 256 | 224px → 128 | 336px → 64
    #   R2+1D     96-112px → 256 | 160px → 128 | ≥224px → 64
    # R2+1D split convolutions create 2× intermediate tensors vs R3D:
    #   - 160px bs=128 → 89GB ✓
    #   - 224px bs=128 → ~174GB ✗ (crashes) → bs=64 → ~87GB ✓
    # 336px for all: spatial area 2.25× vs 224px, so bs=128 → OOM; bs=64 stays ~74GB
    local bs=256
    [[ $res -ge 224 ]] && bs=128
    [[ $res -ge 336 ]] && bs=64
    [[ "$model" == "r2plus1d_18" && $res -ge 160 ]] && bs=128
    [[ "$model" == "r2plus1d_18" && $res -ge 224 ]] && bs=64

    log "START ${model} @ ${res}px  batch=${bs}"
    source "${VENV}/bin/activate"
    cuda_drain
    if ! python scripts/accv2026/train_torchvision.py \
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
        --wandb-tags    accv2026 "${DATASET}" "${model}" "${res}px" resolution-ablation spatial-aliasing local; then
        log "ERROR ${model} @ ${res}px — training failed (OOM or crash), will retry at end"
        return 1
    fi
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
    cuda_drain
    if ! python scripts/accv2026/train_slowfast.py \
        --dataset       "${DATASET}" \
        --epochs        "${EPOCHS}" \
        --batch-size    "${bs}" \
        --num-workers   12 \
        --input-size    "${res}" \
        --save-path     "${ckpt}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "p3-slowfast-${DATASET}-${res}px-e${EPOCHS}-local" \
        --wandb-tags    accv2026 "${DATASET}" slowfast "${res}px" resolution-ablation spatial-aliasing local; then
        log "ERROR ${model} @ ${res}px — training failed (OOM or crash), will retry at end"
        return 1
    fi
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
    cuda_drain
    if ! python scripts/accv2026/train_transformers.py \
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
        --wandb-tags    accv2026 "${DATASET}" "${model}" "${res}px" resolution-ablation spatial-aliasing local; then
        log "ERROR ${model} @ ${res}px — training failed (OOM or crash), will retry at end"
        return 1
    fi
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
    cuda_drain
    if ! python scripts/accv2026/train_videomamba.py \
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
        --wandb-tags    accv2026 "${DATASET}" videomamba "${res}px" resolution-ablation spatial-aliasing local; then
        log "ERROR ${model} @ ${res}px — training failed (OOM or crash), will retry at end"
        return 1
    fi
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

# ── Retry any failed checkpoints (catches OOM crashes from runs above) ────────
log ""
log "--- Retry pass: checking for any missing checkpoints ---"
log "Waiting 60s for CUDA private pools to drain before retry pass..."
sleep 60

# Enable expandable segments to reduce fragmentation in tight-memory cases.
# This is the fix for R3D/MC3/R2+1D at 336px: bs=64 hits ~94GB (just over limit);
# expandable_segments lets the allocator reuse freed blocks more aggressively.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RETRY_FAILED=0

retry_cnn() {
    local model=$1 res=$2
    is_done "$model" "$res" && return 0
    # 336px: bs=32 (pure 3D convs at 336px fill ~94GB at bs=64; bs=32 → ~47GB)
    # ≥224px R2+1D: bs=32 (split convs are 2× heavier)
    # ≤224px other CNNs: bs=64 (safe for BN, fits in 97GB)
    local bs=64
    [[ $res -ge 336 ]] && bs=32
    [[ "$model" == "r2plus1d_18" && $res -ge 224 ]] && bs=32
    local ckpt="${CKPT_BASE}/accv2026_${model}_${DATASET}_${res}px_e${EPOCHS}_h200"
    log "RETRY ${model} @ ${res}px  bs=${bs}"
    sleep 60
    source "${VENV}/bin/activate"
    if ! python scripts/accv2026/train_torchvision.py \
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
        --wandb-run-name "p3-${model}-${DATASET}-${res}px-e${EPOCHS}-local-retry" \
        --wandb-tags    accv2026 "${DATASET}" "${model}" "${res}px" resolution-ablation spatial-aliasing local; then
        log "ERROR ${model} @ ${res}px retry failed"
        return 1
    fi
    log "DONE  ${model} @ ${res}px (retry)"
}

for res in "${RESOLUTIONS[@]}"; do
    for model in r3d_18 mc3_18 r2plus1d_18; do
        retry_cnn "$model" "$res" || RETRY_FAILED=$((RETRY_FAILED+1))
    done
    if ! is_done "slowfast_r50" "$res"; then
        # 336px bs=48 OOMs (~74GB activations + 18GB CUDAGraph pools ≈ 92GB, tight);
        # bs=32 cuts activation memory by ~33% → safely under 94.97GB
        local sf_bs=48
        [[ $res -ge 336 ]] && sf_bs=32
        [[ $res -le 160 ]] && sf_bs=64
        local sf_ckpt="${CKPT_BASE}/accv2026_slowfast_r50_${DATASET}_${res}px_e${EPOCHS}_h200"
        log "RETRY slowfast_r50 @ ${res}px  bs=${sf_bs}"
        sleep 60
        source "${VENV}/bin/activate"
        if ! python scripts/accv2026/train_slowfast.py \
            --dataset       "${DATASET}" \
            --epochs        "${EPOCHS}" \
            --batch-size    "${sf_bs}" \
            --num-workers   12 \
            --input-size    "${res}" \
            --save-path     "${sf_ckpt}" \
            --wandb-project "${WANDB_PROJECT}" \
            --wandb-run-name "p3-slowfast-${DATASET}-${res}px-e${EPOCHS}-local-retry" \
            --wandb-tags    accv2026 "${DATASET}" slowfast "${res}px" resolution-ablation spatial-aliasing local; then
            log "ERROR slowfast_r50 @ ${res}px retry failed"
            RETRY_FAILED=$((RETRY_FAILED+1))
        else
            log "DONE  slowfast_r50 @ ${res}px (retry)"
        fi
    fi
    for model in timesformer vivit videomae; do
        if ! is_done "$model" "$res"; then
            log "RETRY ${model} @ ${res}px"
            sleep 60
            run_transformer "$model" "$res" || RETRY_FAILED=$((RETRY_FAILED+1))
        fi
    done
    if ! is_done "videomamba" "$res"; then
        log "RETRY videomamba @ ${res}px"
        sleep 60
        run_videomamba "$res" || RETRY_FAILED=$((RETRY_FAILED+1))
    fi
done

log ""
log "=== ALL DONE — $(date) ==="
if [[ $RETRY_FAILED -gt 0 ]]; then
    log "WARNING: ${RETRY_FAILED} checkpoint(s) failed even after retry — check logs above"
fi
log "Results in: ${CKPT_BASE}/accv2026_*_finegym_*px_e${EPOCHS}_h200/"
log "Run eval_p3_retrained.py to evaluate all retrained checkpoints"
