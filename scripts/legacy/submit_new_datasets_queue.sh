#!/usr/bin/env bash
# Submit WLASL + EPIC-Kitchens jobs + SlowFast Diving48 resume.
# Polls every 60s and submits respecting the QOS 6-job limit.
# Run AFTER EPIC-Kitchens download is complete (~37K clips).
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

# Verify EPIC-Kitchens clips are downloaded before submitting those jobs
EPIC_CLIPS=$(find data/EPIC_data/clips -name "*.mp4" 2>/dev/null | wc -l)
echo "[submit_queue] EPIC-Kitchens clips available: ${EPIC_CLIPS}"
if [[ $EPIC_CLIPS -lt 30000 ]]; then
  echo "[submit_queue] WARNING: EPIC-Kitchens clips may not be fully downloaded (${EPIC_CLIPS}/~37455)"
  echo "  Consider waiting for download to complete before running this script."
fi

PENDING=(
  scripts/accv2026/slurm_a100_diving48_slowfast_resume.sbatch
  scripts/accv2026/slurm_h200_wlasl_videomae.sbatch
  scripts/accv2026/slurm_a100_wlasl_r2plus1d.sbatch
  scripts/accv2026/slurm_h200_epic_kitchens_videomae.sbatch
  scripts/accv2026/slurm_a100_epic_kitchens_r2plus1d.sbatch
)

MAX_JOBS=6
idx=0

echo "[submit_queue] ${#PENDING[@]} jobs to submit, max $MAX_JOBS simultaneous"

while [[ $idx -lt ${#PENDING[@]} ]]; do
  current=$(squeue -u wesleyferreiramaia -h | wc -l)
  free=$(( MAX_JOBS - current ))
  echo "[submit_queue] $(date '+%H:%M:%S') — $current running/queued, $free slots free"

  while [[ $free -gt 0 && $idx -lt ${#PENDING[@]} ]]; do
    sbatch "${PENDING[$idx]}"
    echo "  Submitted: ${PENDING[$idx]}"
    (( idx++ )) || true
    (( free-- )) || true
  done

  if [[ $idx -lt ${#PENDING[@]} ]]; then
    echo "  All slots full, waiting 60s..."
    sleep 60
  fi
done

echo "[submit_queue] All ${#PENDING[@]} jobs submitted."
