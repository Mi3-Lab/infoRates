#!/usr/bin/env bash
# Waits for the background `mv data /scratch/...` to finish,
# creates the data/ symlink, merges stray scratch dirs, then starts the feeder.
# Usage: nohup bash scripts/accv2026/post_mv_setup.sh > evaluations/accv2026/logs/post_mv_setup.log 2>&1 &
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

SCRATCH_BASE="/scratch/wesleyferreiramaia/infoRates"
MV_PID=589012
LOG="evaluations/accv2026/logs/post_mv_setup_$(date +%Y%m%d_%H%M%S).log"
mkdir -p evaluations/accv2026/logs

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "post_mv_setup started — waiting for mv PID ${MV_PID}"

# Wait for mv to finish
while kill -0 "${MV_PID}" 2>/dev/null; do
    log "  mv still running (PID ${MV_PID})..."
    sleep 30
done

log "mv finished — checking state"

# If data/ still exists as a real dir on /data, mv may have failed partway.
# Use rsync to complete any missing pieces, then remove source.
if [[ -d "data" && ! -L "data" ]]; then
    log "data/ still a real dir — checking if fully copied to scratch"
    SRC_COUNT=$(find data/ -type f | wc -l)
    DST_COUNT=$(find "${SCRATCH_BASE}/data/" -type f 2>/dev/null | wc -l)
    log "  /data files: ${SRC_COUNT}  /scratch files: ${DST_COUNT}"

    if [[ "${SRC_COUNT}" -gt 0 && "${DST_COUNT}" -lt "${SRC_COUNT}" ]]; then
        log "  Incomplete — running rsync to finish the copy"
        rsync -a --remove-source-files --progress \
            data/ "${SCRATCH_BASE}/data/" 2>&1 | tee -a "$LOG"
        # Remove now-empty source dirs
        find data/ -depth -type d -empty -delete 2>/dev/null || true
    fi

    # Remove the now-empty source dir and create symlink
    if [[ -d "data" && ! -L "data" ]]; then
        remaining=$(find data/ -type f 2>/dev/null | wc -l)
        if [[ "${remaining}" -eq 0 ]]; then
            rm -rf data/
            log "  Removed empty data/ from /data"
        else
            log "  WARNING: ${remaining} files still in data/ — NOT removing"
        fi
    fi
fi

# Create symlink if it doesn't exist
if [[ ! -L "data" ]]; then
    ln -s "${SCRATCH_BASE}/data" data
    log "Created symlink: data -> ${SCRATCH_BASE}/data"
else
    log "data/ symlink already exists"
fi

# Merge stray Diving48_data and Something_data from /scratch root into data/
for ds in Diving48_data Something_data WLASL_data; do
    STRAY="${SCRATCH_BASE}/${ds}"
    TARGET="${SCRATCH_BASE}/data/${ds}"
    if [[ -d "${STRAY}" && ! -L "${STRAY}" ]]; then
        if [[ -d "${TARGET}" ]]; then
            STRAY_COUNT=$(find "${STRAY}" -type f | wc -l)
            log "  Merging stray ${ds} (${STRAY_COUNT} files) into data/"
            rsync -a --remove-source-files "${STRAY}/" "${TARGET}/" 2>&1 | tee -a "$LOG"
            find "${STRAY}" -depth -type d -empty -delete 2>/dev/null || true
        else
            log "  Moving stray ${ds} into data/"
            mv "${STRAY}" "${TARGET}"
        fi
        log "  ${ds} merged"
    fi
done

# Verify
log ""
log "=== Final state ==="
log "  data symlink: $(readlink data 2>/dev/null || echo NOT_A_SYMLINK)"
log "  data/ contents:"
ls "${SCRATCH_BASE}/data/" 2>/dev/null | while read d; do
    log "    ${d}: $(du -sh "${SCRATCH_BASE}/data/${d}" 2>/dev/null | cut -f1)"
done

# Check quota
log ""
log "=== Quota ==="
quota -u wesleyferreiramaia 2>&1 | tee -a "$LOG"

log ""
log "=== Starting feeder ==="
nohup bash scripts/accv2026/feeder_submit_jobs.sh > /dev/null 2>&1 &
FEEDER_PID=$!
log "Feeder started — PID ${FEEDER_PID}"
log "post_mv_setup COMPLETE"
