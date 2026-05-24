#!/usr/bin/env bash
# W&B auto-sync daemon — run ONCE on the login node.
# Compute nodes have no internet; this bridges offline runs to W&B in real time.
#
# Start:  nohup bash scripts/accv2026/wandb_daemon.sh &
# Stop:   kill $(cat .wandb_daemon.pid)
# Logs:   tail -f evaluations/accv2026/logs/wandb_daemon.log
#
# How it works:
#   /data/ is a shared NFS filesystem. Compute nodes write wandb/ runs as
#   "offline-run-*" directories. This daemon reads those files from the login
#   node (which has internet) and pushes them to wandb.ai every SYNC_INTERVAL
#   seconds — giving you live progress without any manual steps.

set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

SYNC_INTERVAL="${SYNC_INTERVAL:-60}"
PROJECT="${WANDB_PROJECT:-inforates-accv2026}"
LOG="evaluations/accv2026/logs/wandb_daemon.log"
PID_FILE=".wandb_daemon.pid"

mkdir -p evaluations/accv2026/logs

# Guard: only one daemon at a time
if [[ -f "${PID_FILE}" ]]; then
  old_pid=$(cat "${PID_FILE}")
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "[wandb-daemon] Already running (pid ${old_pid}). Exiting."
    exit 1
  fi
fi

echo $$ > "${PID_FILE}"
trap 'rm -f "${PID_FILE}"; echo "[wandb-daemon] Stopped." | tee -a "${LOG}"' EXIT

echo "[wandb-daemon] Started (pid $$) — syncing to project '${PROJECT}' every ${SYNC_INTERVAL}s" | tee -a "${LOG}"
echo "[wandb-daemon] Stop with: kill \$(cat .wandb_daemon.pid)" | tee -a "${LOG}"

while true; do
  # Find all offline runs — these are written by compute nodes
  mapfile -t RUNS < <(find wandb -maxdepth 1 -name "offline-run-*" -type d 2>/dev/null | sort)

  if [[ ${#RUNS[@]} -eq 0 ]]; then
    echo "[wandb-daemon] $(date '+%H:%M:%S') — no offline runs found, waiting..." | tee -a "${LOG}"
  else
    echo "[wandb-daemon] $(date '+%H:%M:%S') — syncing ${#RUNS[@]} run(s)" | tee -a "${LOG}"
    for run_dir in "${RUNS[@]}"; do
      # --include-synced: re-sync runs we already pushed (picks up new steps)
      # --no-mark-synced: don't mark as done so next loop re-checks for new data
      if wandb sync \
          --project "${PROJECT}" \
          --include-offline \
          --include-synced \
          --no-mark-synced \
          "${run_dir}" >> "${LOG}" 2>&1; then
        echo "[wandb-daemon]   OK  ${run_dir}" | tee -a "${LOG}"
      else
        echo "[wandb-daemon]   ERR ${run_dir} (will retry)" | tee -a "${LOG}"
      fi
    done
  fi

  sleep "${SYNC_INTERVAL}"
done
