#!/usr/bin/env bash
# Submit remaining multi-dataset jobs respecting the QOS 6-job limit.
# Polls every 60s and submits when slots are free.
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

PENDING=(
  scripts/accv2026/slurm_a100_ucf101_slowfast.sbatch
  scripts/accv2026/slurm_a100_hmdb51_slowfast.sbatch
  scripts/accv2026/slurm_a100_diving48_slowfast.sbatch
  scripts/accv2026/slurm_h200_ucf101_videomae.sbatch
  scripts/accv2026/slurm_h200_hmdb51_videomae.sbatch
  scripts/accv2026/slurm_h200_diving48_videomae.sbatch
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
