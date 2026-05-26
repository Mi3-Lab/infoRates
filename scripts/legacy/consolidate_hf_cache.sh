#!/usr/bin/env bash
# Consolidates ALL HuggingFace caches into one folder on /scratch.
# Deletes Cosmos models, deduplicates, creates symlinks.
# Waits for the in-progress rsync (hf_cache_main) to finish first.
set -uo pipefail

SCRATCH="/scratch/wesleyferreiramaia"
DATA="/data/wesleyferreiramaia"
UNIFIED="${SCRATCH}/hf_unified"        # single destination
UNIFIED_HUB="${UNIFIED}/hub"
LOG="${DATA}/infoRates/evaluations/accv2026/logs/consolidate_hf_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${UNIFIED_HUB}" "$(dirname "$LOG")"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ── wait for in-progress rsync to finish ─────────────────────────────────────
log "Waiting for rsync (hf_cache_main) to finish..."
while pgrep -f "rsync.*hf_cache_main" > /dev/null 2>&1; do
    sleep 15
done
log "rsync done."

# ── helper: merge hub dir into UNIFIED_HUB, skip duplicates ──────────────────
merge_hub() {
    local src_hub="$1"
    local label="$2"
    [[ -d "$src_hub" ]] || { log "  SKIP $label — not found"; return; }
    log "  Merging $label/hub → hf_unified/hub"
    for model_dir in "${src_hub}"/*/; do
        [[ -d "$model_dir" ]] || continue
        local name
        name=$(basename "$model_dir")
        local dst="${UNIFIED_HUB}/${name}"
        if [[ -d "$dst" ]]; then
            log "    SKIP (already in unified): ${name}"
        else
            mv "$model_dir" "${UNIFIED_HUB}/"
            log "    MOVED: ${name}"
        fi
    done
}

# ── Step 1: delete Cosmos (big, not used by any active project) ───────────────
log ""
log "=== Step 1: Delete Cosmos models ==="
for cosmos_path in \
    "${SCRATCH}/hf_cache_main/hub/models--nvidia--Cosmos-1.0-Guardrail" \
    "${DATA}/.cache_hf/hub/models--nvidia--Cosmos-1.0-Guardrail" \
    "${DATA}/.cache_hf/models--nvidia--Cosmos-1.0-Diffusion-14B-Text2World" \
    "${DATA}/huggingface_cache/hub/models--nvidia--Cosmos-1.0-Guardrail"; do
    if [[ -d "$cosmos_path" ]]; then
        log "  Deleting: $cosmos_path ($(du -sh "$cosmos_path" 2>/dev/null | cut -f1))"
        rm -rf "$cosmos_path"
    fi
done

# ── Step 2: delete duplicate dinov2-large from hf_cache (already in huggingface_cache) ──
log ""
log "=== Step 2: Delete duplicates from /data ==="
for dup in \
    "${DATA}/hf_cache/hub/models--facebook--dinov2-large" \
    "${DATA}/hf_cache/hub/models--MCG-NJU--videomae-base-finetuned-kinetics"; do
    if [[ -d "$dup" ]]; then
        log "  Deleting duplicate: $(basename "$dup")"
        rm -rf "$dup"
    fi
done

# ── Step 3: merge all sources into UNIFIED_HUB ───────────────────────────────
log ""
log "=== Step 3: Merge into ${UNIFIED_HUB} ==="

# 3a. hf_cache_main (came from ~/.cache/huggingface — already on scratch, fast mv)
merge_hub "${SCRATCH}/hf_cache_main/hub" "hf_cache_main"
# copy xet dir too
if [[ -d "${SCRATCH}/hf_cache_main/xet" && ! -d "${UNIFIED}/xet" ]]; then
    mv "${SCRATCH}/hf_cache_main/xet" "${UNIFIED}/"
fi
rm -rf "${SCRATCH}/hf_cache_main"

# 3b. .cache_hf (Cosmos deleted, remainder is small)
merge_hub "${DATA}/.cache_hf/hub" ".cache_hf"

# 3c. huggingface_cache (Alpamayo, SD-XL, LLaVA, etc.)
merge_hub "${DATA}/huggingface_cache/hub" "huggingface_cache"

# 3d. hf_cache (duplicates removed, only nllb-200 remains)
merge_hub "${DATA}/hf_cache/hub" "hf_cache"
# nllb at root level
if [[ -d "${DATA}/hf_cache/models--facebook--nllb-200-1.3B" ]]; then
    mv "${DATA}/hf_cache/models--facebook--nllb-200-1.3B" "${UNIFIED_HUB}/"
    log "  MOVED: nllb-200-1.3B (from hf_cache root)"
fi

# 3e. infoRates models already on scratch — merge those too
merge_hub "${SCRATCH}/infoRates/hf_cache/hub" "infoRates/hf_cache"
rm -rf "${SCRATCH}/infoRates/hf_cache"
log "  infoRates/hf_cache merged and removed"

# ── Step 4: create symlinks from all old locations → unified ─────────────────
log ""
log "=== Step 4: Symlinks ==="

# ~/.cache/huggingface → unified (default HF location)
rm -rf "${DATA}/.cache/huggingface"
ln -s "${UNIFIED}" "${DATA}/.cache/huggingface"
log "  ~/.cache/huggingface → ${UNIFIED}"

# .cache_hf → unified
rm -rf "${DATA}/.cache_hf"
ln -s "${UNIFIED}" "${DATA}/.cache_hf"
log "  .cache_hf → ${UNIFIED}"

# huggingface_cache → unified
rm -rf "${DATA}/huggingface_cache"
ln -s "${UNIFIED}" "${DATA}/huggingface_cache"
log "  huggingface_cache → ${UNIFIED}"

# hf_cache → unified
rm -rf "${DATA}/hf_cache"
ln -s "${UNIFIED}" "${DATA}/hf_cache"
log "  hf_cache → ${UNIFIED}"

# ── Step 5: update HF_HOME in infoRates submit script ────────────────────────
log ""
log "=== Step 5: Update HF_HOME in submit_missing_jobs.sh ==="
sed -i "s|HF_HOME=/scratch/wesleyferreiramaia/infoRates/hf_cache|HF_HOME=${UNIFIED}|g" \
    "${DATA}/infoRates/scripts/accv2026/submit_missing_jobs.sh"
log "  Updated submit_missing_jobs.sh → HF_HOME=${UNIFIED}"

# ── Final report ─────────────────────────────────────────────────────────────
log ""
log "=== Final state ==="
log "Unified cache contents:"
du -sh "${UNIFIED_HUB}"/*/  2>/dev/null | sort -rh | tee -a "$LOG"
log ""
log "Total unified: $(du -sh "${UNIFIED}" 2>/dev/null | cut -f1)"
log ""
log "Quota /data:"
quota -u wesleyferreiramaia 2>&1 | grep "/data" | tee -a "$LOG"
log ""
log "Symlinks:"
ls -la "${DATA}/.cache/huggingface" "${DATA}/.cache_hf" "${DATA}/huggingface_cache" "${DATA}/hf_cache" 2>/dev/null | tee -a "$LOG"
log ""
log "=== DONE ==="
