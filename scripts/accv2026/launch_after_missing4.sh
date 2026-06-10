#!/bin/bash
# Waits for run_missing4.sh (SlowFast FineGym) to complete, then starts retrain_all.sh
cd /mnt/datasets/infoRates

echo "[$(date '+%H:%M:%S')] Waiting for run_missing4.sh (PID 2375493) to finish..."
while kill -0 2375493 2>/dev/null; do
    sleep 60
done
echo "[$(date '+%H:%M:%S')] run_missing4.sh finished. Starting retrain_all.sh..."
nohup bash scripts/accv2026/retrain_all.sh > evaluations/accv2026/logs/retrain_all.log 2>&1 &
echo "[$(date '+%H:%M:%S')] retrain_all.sh started (PID $!), logging to evaluations/accv2026/logs/retrain_all.log"
