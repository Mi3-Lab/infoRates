#!/bin/bash
# 🚀 Treinar com TRA - Kinetics-400
# GPU: H200 (já alocada)
# Logging: WandB
#
# Execute este script para treinar baseline e TRA no Kinetics-400

cd /home/wesleyferreiramaia/data/infoRates
source .venv/bin/activate

# Configuração
MODEL="timesformer"      # Opções: timesformer, videomae, vivit
DATASET="kinetics400"
EPOCHS=10
BATCH_SIZE=32            # H200 tem 141GB VRAM, pode usar batch maior
LR=2e-5
NUM_WORKERS=8            # H200 tem muitos cores, paralelize I/O

echo "=========================================="
echo "🎯 Treinamento TRA - Kinetics-400"
echo "=========================================="
echo "GPU: H200 (141GB VRAM)"
echo "Model: $MODEL"
echo "Dataset: Kinetics-400 (validation set split)"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Training: ~15.8k videos (80%)"
echo "Validation: ~4k videos (20%)"
echo "=========================================="
echo ""

# PASSO 1: Treinar BASELINE (sem TRA)
echo "📌 [1/3] Treinando BASELINE (sem TRA)..."
echo "Tempo estimado: ~2h na H200"
echo ""

python scripts/train_with_tra.py \
  --model $MODEL \
  --dataset $DATASET \
  --tra-mode baseline \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --num-workers $NUM_WORKERS \
  --save-dir fine_tuned_models/tra_experiments/kinetics400 \
  --eval-robustness \
  --wandb

if [ $? -ne 0 ]; then
    echo "❌ Erro no treinamento baseline!"
    exit 1
fi

echo ""
echo "✅ Baseline training complete!"
echo ""

# PASSO 2: Treinar com TRA
echo "📌 [2/3] Treinando com TRA..."
echo "Tempo estimado: ~2.3h na H200 (um pouco mais lento devido à augmentação)"
echo ""

python scripts/train_with_tra.py \
  --model $MODEL \
  --dataset $DATASET \
  --tra-mode tra \
  --p-augment 0.5 \
  --coverage-range 25 50 75 100 \
  --stride-range 1 2 4 8 16 \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --num-workers $NUM_WORKERS \
  --save-dir fine_tuned_models/tra_experiments/kinetics400 \
  --eval-robustness \
  --wandb

if [ $? -ne 0 ]; then
    echo "❌ Erro no treinamento TRA!"
    exit 1
fi

echo ""
echo "✅ TRA training complete!"
echo ""

# PASSO 3: Resultados
echo "=========================================="
echo "🎉 EXPERIMENTOS COMPLETOS!"
echo "=========================================="
echo ""
echo "📊 Resultados salvos em:"
echo "  - Modelos:"
echo "    • fine_tuned_models/tra_experiments/kinetics400/baseline/$MODEL/"
echo "    • fine_tuned_models/tra_experiments/kinetics400/tra/$MODEL/"
echo ""
echo "  - Robustness evaluations:"
echo "    • fine_tuned_models/tra_experiments/kinetics400/baseline/robustness_${MODEL}.json"
echo "    • fine_tuned_models/tra_experiments/kinetics400/tra/robustness_${MODEL}.json"
echo ""
echo "📈 Para comparar baseline vs TRA:"
echo "   python scripts/compare_baseline_vs_tra.py --model $MODEL --dataset kinetics400"
echo ""
echo "=========================================="
