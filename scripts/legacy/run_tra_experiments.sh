#!/bin/bash
# Quick start guide for TRA experiments
# Run this script to train baseline and TRA models, then compare results

set -e  # Exit on error

echo "🚀 Starting TRA Experiments"
echo "=============================="
echo ""

# Configuration
MODEL="timesformer"  # Options: timesformer, videomae, vivit
EPOCHS=5
BATCH_SIZE=8
LR=2e-5
BASE_DIR="fine_tuned_models/tra_experiments"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo ""

# Step 1: Train Baseline (no TRA)
echo "📌 Step 1/3: Training Baseline (no TRA)..."
echo "Est. time: ~2 hours on single A100 GPU"
echo ""

python scripts/train_with_tra.py \
  --model $MODEL \
  --tra-mode baseline \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --save-dir $BASE_DIR \
  --eval-robustness \
  --num-workers 4

echo "✅ Baseline training complete"
echo ""

# Step 2: Train with TRA
echo "📌 Step 2/3: Training with TRA..."
echo "Est. time: ~2.2 hours on single A100 GPU (slightly slower due to augmentation)"
echo ""

python scripts/train_with_tra.py \
  --model $MODEL \
  --tra-mode tra \
  --p-augment 0.5 \
  --coverage-range 25 50 75 100 \
  --stride-range 1 2 3 \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --save-dir $BASE_DIR \
  --eval-robustness \
  --num-workers 4

echo "✅ TRA training complete"
echo ""

# Step 3: Compare Results
echo "📌 Step 3/3: Comparing Baseline vs TRA..."
echo ""

python scripts/compare_baseline_vs_tra.py \
  --model $MODEL \
  --base-dir $BASE_DIR \
  --plot \
  --save-latex

echo ""
echo "🎉 All experiments complete!"
echo ""
echo "Results saved to:"
echo "  - Models: $BASE_DIR/{baseline,tra}/$MODEL/"
echo "  - Robustness: $BASE_DIR/{baseline,tra}/robustness_$MODEL.json"
echo "  - Comparison plot: docs/figures/tra_comparison_$MODEL.png"
echo "  - LaTeX table: docs/tables/tra_comparison_$MODEL.tex"
echo ""
echo "Next steps:"
echo "  1. Check comparison plot: open docs/figures/tra_comparison_$MODEL.png"
echo "  2. Review improvement metrics in terminal output above"
echo "  3. Add LaTeX table to paper: \\input{tables/tra_comparison_$MODEL.tex}"
