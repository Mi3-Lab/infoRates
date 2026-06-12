#!/usr/bin/env bash
# P3 FineGym — Retrain r3d_18 and mc3_18 at 336px with clean GPU.
#
# Root cause of bad results (r3d=48.5%, mc3=28.4%):
#   - bs=128 at 336px OOMed (3 attempts, various causes: GPU pool, shm full)
#   - Final retry ran at bs=64 but GPU was fragmented from previous long runs
#     (no expandable_segments, insufficient drain between runs)
#   - Existing checkpoints may be partially-trained or uninitialized
#
# This script removes the buggy checkpoints and retrains cleanly:
#   - expandable_segments:True
#   - bs=64 (above BatchNorm minimum, same as original intent)
#   - 120s drain + shm cleanup between runs
#   - GPU must be idle before launching (check with nvidia-smi first)
#
# Usage:
#   cd /mnt/datasets/infoRates
#   nohup bash scripts/accv2026/run_p3_retrain_finegym_cnn336.sh \
#         > evaluations/accv2026/logs/p3_retrain_finegym_cnn336.log 2>&1 &
#   tail -f evaluations/accv2026/logs/p3_retrain_finegym_cnn336.log

set -uo pipefail
cd /mnt/datasets/infoRates
export PYTHONPATH=src

if [[ -d "/scratch/wesleyferreiramaia" ]]; then
    export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
    export HF_HOME="${HF_HOME:-/scratch/wesleyferreiramaia/hf_unified}"
fi
export WANDB_PROJECT=inforates-accv2026
export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV=".venv"
CKPT_BASE="fine_tuned_models"
DATASET="finegym"
EPOCHS=10
RES=336

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] P3-CNN336 | $*"; }

cuda_drain() {
    sleep 120
    rm -f /dev/shm/torch_* /dev/shm/sem.mp-* 2>/dev/null || true
}

run_cnn() {
    local model=$1
    local ckpt="${CKPT_BASE}/accv2026_${model}_${DATASET}_${RES}px_e${EPOCHS}_h200"

    # Remove buggy checkpoint so training starts from scratch
    if [[ -d "${ckpt}" ]]; then
        log "Removing buggy checkpoint: ${ckpt}"
        rm -rf "${ckpt}"
    fi

    log "START ${model} @ ${RES}px  batch=64  [expandable_segments, clean GPU]"
    source "${VENV}/bin/activate"
    cuda_drain

    if ! python scripts/accv2026/train_torchvision.py \
        --dataset       "${DATASET}" \
        --model         "${model}" \
        --epochs        "${EPOCHS}" \
        --batch-size    64 \
        --num-workers   12 \
        --input-size    "${RES}" \
        --save-path     "${ckpt}" \
        --num-frames    16 \
        --lr            1e-4 \
        --weight-decay  0.05 \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "p3-${model}-${DATASET}-${RES}px-e${EPOCHS}-clean" \
        --wandb-tags    accv2026 "${DATASET}" "${model}" "${RES}px" resolution-ablation spatial-aliasing cnn336-clean; then
        log "ERROR ${model} @ ${RES}px — training failed"
        return 1
    fi
    log "DONE  ${model} @ ${RES}px"
}

log "=== P3 FineGym CNN@336px clean retrain: r3d_18, mc3_18 ==="
log "bs=64 + expandable_segments + 120s drain — GPU must be idle at start"
log ""

failed=()
run_cnn r3d_18  || failed+=(r3d_18@336)
run_cnn mc3_18  || failed+=(mc3_18@336)

log ""
if [[ ${#failed[@]} -eq 0 ]]; then
    log "=== ALL 2 runs complete ==="
else
    log "=== Done with ${#failed[@]} failure(s): ${failed[*]} ==="
fi
log "Next: python scripts/accv2026/eval_p3_retrained.py --dataset finegym"
