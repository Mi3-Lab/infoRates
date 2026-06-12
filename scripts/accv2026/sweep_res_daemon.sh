#!/usr/bin/env bash
# Autonomous daemon: monitors training completion, submits resolution coverage×stride
# sweep jobs as checkpoints become available, then regenerates app CSVs.
#
# Usage: nohup bash scripts/accv2026/sweep_res_daemon.sh \
#          > evaluations/accv2026/logs/sweep_res_daemon.log 2>&1 &

set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

SBATCH_SCRIPT="scripts/accv2026/slurm_res_cov_stride.sbatch"
SWEEP_ROOT="evaluations/accv2026/coverage_stride_sweep"
SCRATCH_MODELS="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models"
LOG="evaluations/accv2026/logs/sweep_res_daemon.log"
MAX_SWEEP_JOBS=6   # stay under QOS=7
POLL_INTERVAL=120  # seconds between polls

# Training job IDs still running when daemon starts — wait for all to finish
TRAINING_JOBS=(144132 144206 144232 144331 144348 144355 144360)

MODELS=(r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba)
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)
RESOLUTIONS=(48 96 112 160 224)

ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

# ── Checkpoint detection ───────────────────────────────────────────────────
# Naming: accv2026_{model}_{dataset}_{res}px_e10[_v{N}]_h200
ckpt_exists() {
    local model=$1 dataset=$2 res=$3
    local transformer_models="videomamba timesformer vivit videomae"

    for p in "${SCRATCH_MODELS}"/accv2026_${model}_${dataset}_${res}px_e*_*h200; do
        [[ -d "$p" ]] || continue
        if echo "$transformer_models" | grep -qw "$model"; then
            [[ -f "${p}/accv_meta.json" ]] && echo "$p" && return 0
        else
            [[ -f "${p}/config.json" ]] && grep -q '"backend"' "${p}/config.json" \
                && echo "$p" && return 0
        fi
    done
    return 1
}

# ── Sweep completion detection ─────────────────────────────────────────────
sweep_done() {
    local model=$1 dataset=$2 res=$3
    local dir="${SWEEP_ROOT}/${model}_${dataset}_trainres${res}"
    # Done if sweep_summary.csv exists with 25 rows (5 cov × 5 stride)
    [[ -f "${dir}/sweep_summary.csv" ]] || return 1
    local n
    n=$(wc -l < "${dir}/sweep_summary.csv" 2>/dev/null || echo 0)
    [[ $n -ge 26 ]]  # header + 25 data rows
}

# ── Count active sweep jobs ────────────────────────────────────────────────
count_sweep_jobs() {
    local n
    n=$(squeue -u wesleyferreiramaia --name=res-sweep --noheader --format="%.10i" 2>/dev/null | wc -l)
    echo "${n:-0}"
}

# ── Count training jobs still running ─────────────────────────────────────
count_training() {
    local running=0
    for jid in "${TRAINING_JOBS[@]}"; do
        squeue -j "$jid" --noheader --format="%.10i" 2>/dev/null | grep -q "$jid" && running=$((running+1)) || true
    done
    echo $running
}

# ── Regenerate retrained_spatial.csv ──────────────────────────────────────
rebuild_retrained_spatial() {
    log "Rebuilding dashboard/data/retrained_spatial.csv ..."
    python3 - <<'PYEOF'
import json
from pathlib import Path

SCRATCH = Path("/scratch/wesleyferreiramaia/infoRates/fine_tuned_models")
OUT = Path("dashboard/data/retrained_spatial.csv")

rows = []
for d in sorted(SCRATCH.glob("accv2026_*px_e10_h200*")):
    name = d.name
    # Parse model, dataset, resolution from directory name
    # Format: accv2026_{model}_{dataset}_{res}px_e10_h200[_v2/_v3]
    import re
    m = re.match(r"accv2026_(.+?)_(.+?)_(\d+)px_e\d+_h200", name)
    if not m:
        continue
    model, dataset, res = m.group(1), m.group(2), int(m.group(3))

    # Get best_val_acc
    meta = d / "accv_meta.json"
    cfg  = d / "config.json"
    acc  = None
    if meta.exists():
        try:
            data = json.loads(meta.read_text())
            acc = data.get("best_val_acc") or data.get("val_acc")
        except Exception:
            pass
    elif cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            acc = data.get("best_val_acc")
        except Exception:
            pass

    if acc is None:
        continue
    rows.append((model, dataset, res, round(float(acc) * 100, 4)))

# Deduplicate: keep best acc per (model, dataset, res)
best = {}
for model, dataset, res, acc in rows:
    key = (model, dataset, res)
    if key not in best or acc > best[key]:
        best[key] = acc

lines = ["model,dataset,train_res,acc"]
for (model, dataset, res), acc in sorted(best.items()):
    lines.append(f"{model},{dataset},{res},{acc}")

OUT.write_text("\n".join(lines) + "\n")
print(f"Written {len(lines)-1} rows to {OUT}")
PYEOF
}

# ── Update sweep_summary.csv (base native-res sweep) ──────────────────────
rebuild_sweep_summary() {
    log "Rebuilding dashboard/data/sweep_summary.csv ..."
    python3 - <<'PYEOF'
import pandas as pd
from pathlib import Path

SWEEP_ROOT = Path("evaluations/accv2026/coverage_stride_sweep")
OUT = Path("dashboard/data/sweep_summary.csv")

rows = []
MODELS = {"r3d_18","mc3_18","r2plus1d_18","slowfast_r50","timesformer","vivit","videomae","videomamba"}
DATASETS = {"ucf101","ssv2","hmdb51","diving48","autsl","driveact","epic_kitchens"}

for csv in sorted(SWEEP_ROOT.glob("*/sweep_summary.csv")):
    folder = csv.parent.name
    # Base combos only: {model}_{dataset} (no _res or _trainres suffix)
    if "_res" in folder or "_trainres" in folder:
        continue
    ds = next((d for d in sorted(DATASETS, key=len, reverse=True) if folder.endswith(d)), None)
    if not ds:
        continue
    model = folder[:-(len(ds)+1)]
    if model not in MODELS:
        continue
    try:
        df = pd.read_csv(csv)
        df["model"] = model
        df["dataset"] = ds
        rows.append(df[["coverage","stride","top1","n","model","dataset"]])
    except Exception as e:
        print(f"  [WARN] {csv}: {e}")

if rows:
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"Written {len(out)} rows to {OUT}")
else:
    print("No rows found!")
PYEOF
}

# ══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════
log "=== sweep_res_daemon started ==="
log "Training jobs to wait for: ${TRAINING_JOBS[*]}"
log "Resolutions: ${RESOLUTIONS[*]}"
log "Models: ${MODELS[*]}"
log "Datasets: ${DATASETS[*]}"

mkdir -p evaluations/accv2026/logs

submitted_jobs=()

while true; do
    n_training=$(count_training)
    n_sweep=$(count_sweep_jobs)
    log "Training remaining: ${n_training} | Sweep jobs running: ${n_sweep}"

    # Submit sweep jobs for combos with available checkpoints
    pending_count=0
    submitted_this_round=0

    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for res in "${RESOLUTIONS[@]}"; do
                sweep_done "$model" "$dataset" "$res" && continue  # already done

                ckpt_path=$(ckpt_exists "$model" "$dataset" "$res" 2>/dev/null) || continue  # no checkpoint yet

                # Check if a sweep job is already queued/running for this combo
                dir="${SWEEP_ROOT}/${model}_${dataset}_trainres${res}"
                lock="${dir}/.submitted"
                [[ -f "$lock" ]] && continue  # already submitted

                pending_count=$((pending_count + 1))

                # Submit if under job limit
                n_sweep=$(count_sweep_jobs)
                if [[ $n_sweep -lt $MAX_SWEEP_JOBS ]]; then
                    mkdir -p "$dir"
                    touch "$lock"
                    jid=$(sbatch \
                        --output="evaluations/accv2026/logs/res-sweep-${model}-${dataset}-${res}px-%j.out" \
                        --error="evaluations/accv2026/logs/res-sweep-${model}-${dataset}-${res}px-%j.err" \
                        --export="ALL,MODEL=${model},DATASET=${dataset},TRAIN_RES=${res}" \
                        "$SBATCH_SCRIPT" 2>/dev/null | awk '{print $NF}')
                    log "  Submitted sweep ${model}/${dataset}@${res}px → job ${jid}"
                    submitted_jobs+=("$jid")
                    submitted_this_round=$((submitted_this_round + 1))
                    sleep 2
                fi
            done
        done
    done

    # Check if everything is done
    total_missing=0
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for res in "${RESOLUTIONS[@]}"; do
                sweep_done "$model" "$dataset" "$res" || total_missing=$((total_missing + 1))
            done
        done
    done

    log "Missing sweeps: ${total_missing} | Pending submissions: ${pending_count}"

    if [[ $total_missing -eq 0 ]]; then
        log "=== ALL SWEEPS COMPLETE ==="
        rebuild_retrained_spatial
        rebuild_sweep_summary
        log "=== DONE — App CSVs updated. Daemon exiting. ==="
        exit 0
    fi

    # If nothing submitted this round and no training left and no sweep jobs running,
    # something is stuck — log and keep trying
    if [[ $submitted_this_round -eq 0 && $n_training -eq 0 && $n_sweep -eq 0 && $total_missing -gt 0 ]]; then
        log "[WARN] ${total_missing} sweeps missing but nothing running/submitting. Missing checkpoints?"
        # List what's missing without a checkpoint
        for model in "${MODELS[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                for res in "${RESOLUTIONS[@]}"; do
                    sweep_done "$model" "$dataset" "$res" && continue
                    ckpt_exists "$model" "$dataset" "$res" &>/dev/null \
                        || log "  NO CHECKPOINT: ${model}/${dataset}@${res}px"
                done
            done
        done
    fi

    sleep $POLL_INTERVAL
done
