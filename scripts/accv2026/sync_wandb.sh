#!/usr/bin/env bash
# Run this from the LOGIN NODE after compute jobs finish.
# Compute nodes don't have internet, so wandb saves runs locally.
# This script syncs all offline runs to wandb.ai.
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

PROJECT="${WANDB_PROJECT:-inforates-accv2026}"

echo "[sync_wandb] Scanning wandb/ for offline runs..."
SYNCED=0
SKIPPED=0

for run_dir in wandb/run-*/; do
  [[ -d "$run_dir" ]] || continue

  # Check if this run has a .wandb file (valid run)
  wandb_file=$(ls "${run_dir}"*.wandb 2>/dev/null | head -1)
  if [[ -z "$wandb_file" ]]; then
    continue
  fi

  # Check if already synced (synced flag written by wandb)
  if [[ -f "${run_dir}files/wandb-synced.txt" ]]; then
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  echo "[sync_wandb] Syncing ${run_dir}..."
  if wandb sync --project "${PROJECT}" "${run_dir}" 2>&1 | tail -1; then
    SYNCED=$((SYNCED + 1))
  else
    echo "[sync_wandb] WARNING: failed to sync ${run_dir}"
  fi
done

echo "[sync_wandb] Done — synced ${SYNCED} run(s), skipped ${SKIPPED} already-synced."
