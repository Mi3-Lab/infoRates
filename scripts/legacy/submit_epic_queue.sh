#!/usr/bin/env bash
# Submit EPIC-Kitchens jobs once download is complete (~37455 clips).
# Respects QOS limit of 6 simultaneous jobs.
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

EPIC_CLIPS=$(find data/EPIC_data/clips -name "*.mp4" 2>/dev/null | wc -l)
echo "[epic_queue] EPIC-Kitchens clips available: ${EPIC_CLIPS}/~37455"
if [[ $EPIC_CLIPS -lt 35000 ]]; then
  echo "[epic_queue] ERROR: download not yet complete (${EPIC_CLIPS} < 35000). Aborting."
  exit 1
fi

PENDING=(
  scripts/accv2026/slurm_h200_epic_kitchens_videomae.sbatch
  scripts/accv2026/slurm_a100_epic_kitchens_r2plus1d.sbatch
)

MAX_JOBS=6

echo "[epic_queue] Submitting ${#PENDING[@]} EPIC-Kitchens jobs"

for sbatch_file in "${PENDING[@]}"; do
  while true; do
    current=$(squeue -u wesleyferreiramaia -h 2>/dev/null | wc -l)
    if [[ $current -lt $MAX_JOBS ]]; then
      sbatch "$sbatch_file"
      echo "  Submitted: $sbatch_file (${current} jobs were running)"
      sleep 3
      break
    fi
    echo "  [$(date '+%H:%M:%S')] $current/$MAX_JOBS jobs running, waiting 60s..."
    sleep 60
  done
done

echo "[epic_queue] Done. Check: squeue -u wesleyferreiramaia"
