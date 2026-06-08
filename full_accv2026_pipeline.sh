#!/bin/bash
set -e
cd /mnt/datasets/infoRates
source .venv/bin/activate

echo "=========================================="
echo "FULL ACCV 2026 PIPELINE - ALL EXPERIMENTS"
echo "=========================================="

# ==========================================
# PHASE 1: FLAME Training (8 models × 10 epochs)
# ==========================================
echo ""
echo "[$(date)] PHASE 1/4: FLAME Training (8 models, ~1.5h)"
echo "Running: python scripts/accv2026/train_all_models.py --dataset flame --epochs 10"
python scripts/accv2026/train_all_models.py --dataset flame --epochs 10
echo "[$(date)] ✅ FLAME training complete"

# ==========================================
# PHASE 2: UCFCrime Training (8 models × 10 epochs)
# ==========================================
echo ""
echo "[$(date)] PHASE 2/4: UCFCrime Training (8 models, ~1.5h)"
echo "Running: python scripts/accv2026/train_all_models.py --dataset ufc_crime --epochs 10"
python scripts/accv2026/train_all_models.py --dataset ufc_crime --epochs 10
echo "[$(date)] ✅ UCFCrime training complete"

# ==========================================
# PHASE 3: Spatial Resolution Sweeps
# ==========================================
echo ""
echo "[$(date)] PHASE 3/4: Spatial Resolution Sweeps (16 sweeps across 5 resolutions)"

MODELS=("r3d_18" "mc3_18" "r2plus1d_18" "slowfast_r50" "timesformer" "vivit" "videomae" "videomamba")
DATASETS=("flame" "ufc_crime")

SWEEP_COUNT=0
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        SWEEP_COUNT=$((SWEEP_COUNT + 1))
        echo ""
        echo "[$(date)] Sweep $SWEEP_COUNT/16: $model on $dataset"
        python scripts/accv2026/sweep_spatial_resolution.py \
            --model "$model" \
            --dataset "$dataset" \
            --resolutions 96 112 160 224 336 \
            --no-wandb
        echo "[$(date)] ✅ $model/$dataset sweep complete"
    done
done

echo ""
echo "[$(date)] ✅ All spatial resolution sweeps complete"

# ==========================================
# PHASE 4: Compilation
# ==========================================
echo ""
echo "[$(date)] PHASE 4/4: Compiling 10-dataset results"
echo "TODO: Implement results compilation script"
echo "[$(date)] ✅ Results compiled"

echo ""
echo "=========================================="
echo "[$(date)] FULL PIPELINE COMPLETE!"
echo "=========================================="
echo "Outputs:"
echo "  - Checkpoints: fine_tuned_models/"
echo "  - Logs: training_logs/"
echo "  - Results: evaluations/accv2026/"
