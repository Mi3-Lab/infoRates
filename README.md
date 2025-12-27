# InfoRates: Temporal Sampling for Action Recognition

This repository explores how temporal sampling (coverage and stride) affects action recognition across modern video models.

Quick start: see START_HERE.txt for commands, or the full UNIFIED_GUIDE.md for end‑to‑end docs.

Key entry points
- Training (multi-model, DDP-ready): scripts/data_processing/train_ddp.sh → launches scripts/data_processing/train_multimodel.py
- Evaluation (multi-model): scripts/evaluation/run_eval_multimodel.py
- Legacy DDP eval of a saved model: scripts/evaluation/run_eval.py and scripts/data_processing/pipeline_eval.sh

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

## Repository Structure

- `data/`: Raw datasets (UCF101, Kinetics400, HMDB51)
- `evaluations/`: Model evaluation results and plots
  - `kinetics400/`: Kinetics400 results by model
  - `ucf101/`: UCF101 results by model
- `scripts/`: Utility scripts
  - `data_processing/`: Data preparation, training scripts
  - `evaluation/`: Evaluation and testing scripts
  - `plotting/`: Plotting and statistical analysis scripts
- `src/`: Source code (models, analysis modules)
- `docs/`: Documentation and evaluation reports
- `fine_tuned_models/`: Saved model checkpoints
- `models/`: Pre-trained models