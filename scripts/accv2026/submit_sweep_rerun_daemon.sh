#!/usr/bin/env bash
# Daemon: rerun all invalid trainres sweeps for VideoMAE and VideoMamba.
# Uses get_checkpoint() (now selects by best val_acc, not highest version).
# Max 8 concurrent jobs across both partitions.
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

SBATCH_TRAINRES=scripts/accv2026/slurm_res_cov_stride.sbatch
SBATCH_NATIVE48=scripts/accv2026/slurm_native_at_48px.sbatch
SWEEP_ROOT=evaluations/accv2026/coverage_stride_sweep
MAX_TOTAL=8
PARTITIONS=(cenvalarc.gpu gpu)

ds_list=(autsl diving48 driveact epic_kitchens hmdb51 ssv2 ucf101)

# Pending jobs: (model dataset res sbatch_script)
declare -a PENDING=()

# VideoMAE @ 48, 96, 112, 160px (trainres sweep with retrained checkpoints)
# All use slurm_res_cov_stride.sbatch + --train-res flag so PE is correctly handled
for ds in "${ds_list[@]}"; do
    for res in 48 96 112 160; do
        dir="${SWEEP_ROOT}/videomae_${ds}_trainres${res}"
        [ -f "$dir/sweep_summary.csv" ] && [ "$(wc -l < "$dir/sweep_summary.csv")" -ge 26 ] && continue
        PENDING+=("videomae ${ds} ${res} ${SBATCH_TRAINRES}")
    done
done

# VideoMamba — trainres at multiple resolutions
declare -A vmamba_res
vmamba_res["diving48"]="96 112 160"
vmamba_res["driveact"]="112 160 224"
vmamba_res["epic_kitchens"]="48 96 112 160 224"
vmamba_res["hmdb51"]="96 112 160 224"
vmamba_res["ucf101"]="96 112"

for ds in "${!vmamba_res[@]}"; do
    for res in ${vmamba_res[$ds]}; do
        dir="${SWEEP_ROOT}/videomamba_${ds}_trainres${res}"
        [ -f "$dir/sweep_summary.csv" ] && [ "$(wc -l < "$dir/sweep_summary.csv")" -ge 26 ] && continue
        PENDING+=("videomamba ${ds} ${res} ${SBATCH_TRAINRES}")
    done
done

total_pending=${#PENDING[@]}
echo "[$(date '+%H:%M:%S')] Daemon starting — ${total_pending} sweep jobs to submit"

is_already_queued() {
    local model=$1 ds=$2 res=$3
    squeue -u wesleyferreiramaia --noheader 2>/dev/null | \
        grep -qE "sw-${model:0:4}.*${ds}|${ds}.*${model:0:4}"
}

running_total() {
    squeue -u wesleyferreiramaia --noheader 2>/dev/null | wc -l
}

submit_one() {
    local model=$1 ds=$2 res=$3 sbatch=$4
    local extra_args=""
    local part
    if [[ "$model" == "videomamba" ]]; then
        # H200 nodes are ONLY in cenvalarc.gpu partition
        part="cenvalarc.gpu"
        extra_args="--constraint=H200"
    else
        part=${PARTITIONS[$((RANDOM % ${#PARTITIONS[@]}))]}
    fi
    local job_id
    if [[ "$sbatch" == *native_at_48px* ]]; then
        job_id=$(sbatch --job-name="sw-${model:0:4}-${ds}-${res}" \
            --partition="$part" $extra_args \
            --export=ALL,MODEL="$model",DATASET="$ds" \
            "$sbatch" 2>&1 | awk '{print $NF}')
    else
        job_id=$(sbatch --job-name="sw-${model:0:4}-${ds}-${res}" \
            --partition="$part" $extra_args \
            --export=ALL,MODEL="$model",DATASET="$ds",TRAIN_RES="$res" \
            "$sbatch" 2>&1 | awk '{print $NF}')
    fi
    echo "[$(date '+%H:%M:%S')] Submitted ${model}/${ds}@${res}px → ${part} job ${job_id}"
}

is_done() {
    local model=$1 ds=$2 res=$3 sbatch=$4
    local dir
    if [[ "$sbatch" == *native_at_48px* ]]; then
        dir="${SWEEP_ROOT}/${model}_${ds}_native48"
    else
        dir="${SWEEP_ROOT}/${model}_${ds}_trainres${res}"
    fi
    [ -f "$dir/sweep_summary.csv" ] && [ "$(wc -l < "$dir/sweep_summary.csv")" -ge 26 ]
}

remaining=("${PENDING[@]}")
while [ ${#remaining[@]} -gt 0 ]; do
    new_remaining=()
    running=$(running_total)
    for job in "${remaining[@]}"; do
        read -r model ds res sbatch <<< "$job"
        if is_done "$model" "$ds" "$res" "$sbatch"; then
            echo "[$(date '+%H:%M:%S')] DONE: ${model}/${ds}@${res}px"
            continue
        fi
        if squeue -u wesleyferreiramaia --format="%j" --noheader 2>/dev/null | grep -qF "sw-${model:0:4}-${ds}-${res}"; then
            new_remaining+=("$job")
            continue
        fi
        if [ "$running" -lt "$MAX_TOTAL" ]; then
            submit_one "$model" "$ds" "$res" "$sbatch"
            running=$((running + 1))
            new_remaining+=("$job")
        else
            new_remaining+=("$job")
        fi
    done
    remaining=("${new_remaining[@]}")
    [ ${#remaining[@]} -eq 0 ] && break
    sleep 120
done

echo "[$(date '+%H:%M:%S')] All sweep reruns submitted and complete."
