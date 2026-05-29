#!/usr/bin/env bash
#SBATCH --job-name=cov-stride-sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
source .venv/bin/activate
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE=disabled   # no wandb for sweep — too many runs

# ── Sweep parameters (25 configs: 5 coverage × 5 stride) ──────────────────
COVERAGES=(10 25 50 75 100)
STRIDES=(1 2 4 8 16)

# ── Models and their native frame counts ──────────────────────────────────
# (budget = model_frames so model always gets a full input)
# MODEL_NAME  CKPT_DIR                                    RESIZE  MODEL_FRAMES  MANIFEST_PREFIX  SPLIT
declare -A MODEL_FRAMES=(
    [r3d_18]=16 [mc3_18]=16 [r2plus1d_18]=16
    [slowfast_r50]=32
    [timesformer]=8 [vivit]=32 [videomae]=16
)

# ── Datasets and their manifests ──────────────────────────────────────────
declare -A MANIFESTS=(
    [ucf101]="evaluations/accv2026/manifests/ucf101_val_20_per_class.csv"
    [ssv2]="evaluations/accv2026/manifests/somethingv2_val_20_per_class.csv"
    [autsl]="evaluations/accv2026/manifests/autsl_val_20_per_class.csv"
    [diving48]="evaluations/accv2026/manifests/diving48_val_20_per_class.csv"
    [hmdb51]="evaluations/accv2026/manifests/hmdb51_val_20_per_class.csv"
    [driveact]="evaluations/accv2026/manifests/driveact_val_20_per_class.csv"
    [epic_kitchens]="evaluations/accv2026/manifests/epic_kitchens_val_20_per_class.csv"
)
declare -A DATASET_NAMES=(
    [ucf101]="ucf101" [ssv2]="somethingv2"
    [autsl]="autsl" [diving48]="diving48"
    [hmdb51]="hmdb51" [driveact]="driveact"
    [epic_kitchens]="epic_kitchens"
)
declare -A SPLITS=(
    [ucf101]="validation" [ssv2]="validation"
    [autsl]="val" [diving48]="val"
    [hmdb51]="val" [driveact]="val" [epic_kitchens]="val"
)
declare -A RESIZES=(
    [r3d_18]=112 [mc3_18]=112 [r2plus1d_18]=112
    [slowfast_r50]=224
    [timesformer]=224 [vivit]=224 [videomae]=224
)

# ── What to run (MODEL DATASET pairs for THIS job — set via env vars) ────
MODELS_TO_RUN="${SWEEP_MODELS:-r3d_18 timesformer videomae}"
DATASETS_TO_RUN="${SWEEP_DATASETS:-ucf101 ssv2 autsl diving48}"

echo "=== Coverage×Stride Sweep — job ${SLURM_JOB_ID} ==="
echo "Models:   ${MODELS_TO_RUN}"
echo "Datasets: ${DATASETS_TO_RUN}"
echo "Configs:  ${#COVERAGES[@]} coverages × ${#STRIDES[@]} strides = 25"
nvidia-smi | head -6 || true

OUT_BASE="evaluations/accv2026/coverage_stride_sweep"
mkdir -p "${OUT_BASE}"

run_one() {
    local model=$1 dataset=$2 coverage=$3 stride=$4
    local mf=${MODEL_FRAMES[$model]}
    local resize=${RESIZES[$model]}
    local manifest=${MANIFESTS[$dataset]}
    local dname=${DATASET_NAMES[$dataset]}
    local split=${SPLITS[$dataset]}
    local ckpt="fine_tuned_models/accv2026_${model}_${dataset}_full_e10_a100"
    [[ "$model" == "timesformer" || "$model" == "vivit" || "$model" == "videomae" ]] \
        && ckpt="fine_tuned_models/accv2026_${model}_${dataset}_full_e10_h200"
    [[ "$dataset" == "ssv2" ]] \
        && ckpt="${ckpt/ssv2/ssv2}"  # keep as is, trailing pattern match handled below
    # Handle SSV2 special checkpoint names
    [[ "$model" == "r3d_18" && "$dataset" == "ssv2" ]] \
        && ckpt="fine_tuned_models/accv2026_r3d18_ssv2_full_e10_a100"
    [[ "$model" == "videomae" && "$dataset" == "ssv2" ]] \
        && ckpt="fine_tuned_models/accv2026_videomae_ssv2_full_e5_h200"

    [[ ! -d "$ckpt" ]] && echo "[SKIP] no checkpoint: $ckpt" && return

    local out_dir="${OUT_BASE}/${model}_${dataset}"
    mkdir -p "${out_dir}"

    local stem="${dname}_${split}_cov${coverage}_s${stride}_${model}"
    local summary="${out_dir}/${stem}_summary.csv"
    [[ -f "$summary" ]] && return  # already done

    python scripts/accv2026/eval_fixed_budget.py \
        --manifest  "${manifest}" \
        --dataset-name "${dname}" \
        --split "${split}" \
        --checkpoint "${ckpt}" \
        --budgets "${mf}" \
        --model-frames "${mf}" \
        --coverage "${coverage}" \
        --stride "${stride}" \
        --batch-size 32 \
        --resize "${resize}" \
        --output-dir "${out_dir}" \
        --no-wandb \
        2>/dev/null && echo "  ✓ ${model}/${dataset} C=${coverage}% s=${stride}"
}

for model in ${MODELS_TO_RUN}; do
    for dataset in ${DATASETS_TO_RUN}; do
        for cov in "${COVERAGES[@]}"; do
            for stride in "${STRIDES[@]}"; do
                run_one "$model" "$dataset" "$cov" "$stride"
            done
        done
    done
done

echo "=== Sweep COMPLETE — $(date) ==="
echo "Results in: ${OUT_BASE}"
find "${OUT_BASE}" -name "*_summary.csv" | wc -l | xargs echo "Summary files written:"
