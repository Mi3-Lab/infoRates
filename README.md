# InfoRates: Temporal Sampling for Action Recognition

This repository explores how temporal sampling (coverage and stride) affects action recognition across modern video models.

Quick start: see START_HERE.txt for commands, or the full UNIFIED_GUIDE.md for end‑to‑end docs.

Key entry points
- Training (multi-model, DDP-ready): scripts/train_ddp.sh → launches scripts/train_multimodel.py
- Evaluation (multi-model): scripts/run_eval_multimodel.py
- Legacy DDP eval of a saved model: scripts/run_eval.py and scripts/pipeline_eval.sh

Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Examples
```bash
# 2 GPUs, fine-tune VideoMAE
bash scripts/train_ddp.sh --model videomae --gpus 2 --epochs 5

# Evaluate all models with temporal sampling
python scripts/run_eval_multimodel.py --model all --batch-size 16
```

More details: UNIFIED_GUIDE.md