#!/usr/bin/env bash
# Moves all HuggingFace/model caches from /data to /scratch and creates symlinks.
# Keeps everything accessible (other projects won't break).
# Also moves infoRates-specific models (timesformer, vivit, videomae) to
# /scratch/wesleyferreiramaia/infoRates/hf_cache/ (used by HF_HOME in training jobs).
set -uo pipefail

DATA_BASE="/data/wesleyferreiramaia"
SCRATCH_BASE="/scratch/wesleyferreiramaia"
INFORATES_HF="${SCRATCH_BASE}/infoRates/hf_cache"
LOG="${DATA_BASE}/infoRates/evaluations/accv2026/logs/cache_cleanup_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")" "${INFORATES_HF}/hub"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

move_and_link() {
    local src="$1"
    local dst="$2"
    local name
    name=$(basename "$src")

    if [[ -L "$src" ]]; then
        log "  SKIP $src — already a symlink"
        return
    fi
    if [[ ! -d "$src" && ! -f "$src" ]]; then
        log "  SKIP $src — does not exist"
        return
    fi

    log "  Moving $src → $dst"
    mkdir -p "$(dirname "$dst")"
    rsync -a --remove-source-files "$src/" "$dst/" 2>&1 | tail -1
    find "$src" -depth -type d -empty -delete 2>/dev/null || true
    if [[ -e "$src" ]]; then
        log "  WARNING: $src still has files — check manually"
    else
        ln -s "$dst" "$src"
        log "  Symlink: $src → $dst"
    fi
}

log "===== HF Cache Cleanup Started ====="
log ""

# Step 1: infoRates models to /scratch/infoRates/hf_cache/hub/
# (so HF_HOME=/scratch/.../hf_cache finds them without re-downloading)
log "--- Step 1: infoRates models to scratch/infoRates/hf_cache ---"
HF_SRC="${DATA_BASE}/huggingface_cache/hub"
for model_dir in \
    "models--facebook--timesformer-base-finetuned-k400" \
    "models--google--vivit-b-16x2" \
    "models--MCG-NJU--videomae-base-finetuned-kinetics"; do
    SRC="${HF_SRC}/${model_dir}"
    DST="${INFORATES_HF}/hub/${model_dir}"
    if [[ -d "$SRC" && ! -L "$SRC" ]]; then
        log "  $model_dir → infoRates hf_cache"
        rsync -a --remove-source-files "$SRC/" "$DST/" 2>/dev/null || true
        find "$SRC" -depth -type d -empty -delete 2>/dev/null || true
        log "  Done: $(du -sh "${DST}" 2>/dev/null | cut -f1)"
    elif [[ -d "${INFORATES_HF}/hub/${model_dir}" ]]; then
        log "  $model_dir already in infoRates hf_cache"
    else
        log "  $model_dir not found in huggingface_cache — will download on first run"
    fi
done
log ""

# Also check hf_cache (12GB) for videomae duplicate
for model_dir in "models--MCG-NJU--videomae-base-finetuned-kinetics"; do
    SRC="${DATA_BASE}/hf_cache/hub/${model_dir}"
    DST="${INFORATES_HF}/hub/${model_dir}"
    if [[ -d "$SRC" && ! -d "$DST" ]]; then
        log "  Moving ${model_dir} from hf_cache → infoRates hf_cache"
        rsync -a --remove-source-files "$SRC/" "$DST/" 2>/dev/null || true
        find "$SRC" -depth -type d -empty -delete 2>/dev/null || true
    fi
done
log ""

# Step 2: ~/.cache/huggingface/ (154GB) — biggest
log "--- Step 2: ~/.cache/huggingface (154GB) → scratch ---"
move_and_link \
    "${DATA_BASE}/.cache/huggingface" \
    "${SCRATCH_BASE}/hf_cache_main"
log ""

# Step 3: .cache_hf/ (52GB, Cosmos models)
log "--- Step 3: .cache_hf (52GB) → scratch ---"
move_and_link \
    "${DATA_BASE}/.cache_hf" \
    "${SCRATCH_BASE}/hf_cache_hf"
log ""

# Step 4: huggingface_cache/ (49GB, remaining after step 1)
log "--- Step 4: huggingface_cache (49GB) → scratch ---"
move_and_link \
    "${DATA_BASE}/huggingface_cache" \
    "${SCRATCH_BASE}/huggingface_cache"
log ""

# Step 5: hf_cache/ (12GB, remaining after videomae moved)
log "--- Step 5: hf_cache (12GB) → scratch ---"
move_and_link \
    "${DATA_BASE}/hf_cache" \
    "${SCRATCH_BASE}/hf_cache_old"
log ""

log "===== Done ====="
log ""
log "=== Final /data quota ==="
quota -u wesleyferreiramaia 2>&1 | tee -a "$LOG"
log ""
log "=== /data usage summary ==="
du -sh "${DATA_BASE}"/.[^.]* "${DATA_BASE}"/* 2>/dev/null | sort -rh | head -15 | tee -a "$LOG"
