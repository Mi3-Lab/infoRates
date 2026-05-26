# InfoRates — Temporal Evidence Allocation for Video Recognition

Research code for ACCV 2026 submission.

**Core question:** how many frames does a video model actually need to make a correct prediction, and how does that vary across model families and dataset types?

We measure this with *fixed-budget evaluation*: force a model to decide using only `k` frames sampled uniformly from the video, and sweep `k ∈ {4, 8, 16, 32}`. The resulting accuracy-vs-frames curve — and its area (temporal AUC) — quantifies each model's temporal evidence demand. We then use this signal to build a confidence-based routing system that allocates frames adaptively at inference time without retraining.

**Run status:** [`docs/ACCV_2026_RUN_STATUS.md`](docs/ACCV_2026_RUN_STATUS.md)

---

## Repository layout

```
infoRates/
├── src/info_rates/          # Python package (models, evaluation, data utils)
│   ├── models/              # timesformer, videomae, vivit, torchvision_video, slowfast_video
│   ├── evaluation/          # benchmark.py — fixed-budget evaluator
│   └── data/                # dataset loaders: SSV2, UCF101, HMDB51, Diving48,
│                            #   EPIC-Kitchens, WLASL100
│
├── scripts/accv2026/        # ACCV 2026 experiment pipeline
│   ├── 02_run_fixed_budget_eval.py   # main evaluator entrypoint
│   ├── 05_compute_temporal_metrics.py
│   ├── 07_dataset_temporal_demand.py
│   ├── 08_compile_paper_results.py
│   ├── 09_plot_paper_figures.py
│   ├── 10_per_class_temporal_analysis.py
│   ├── 12_confidence_cascade.py      # cascade routing (k_low → k_high)
│   ├── 13_knapsack_confidence.py     # batch knapsack allocator (learned MLP)
│   ├── 14_plot_routing_comparison.py # unified routing figure
│   ├── 15_baseline_comparison.py     # FrameExit vs knapsack
│   ├── train_transformers.py         # VideoMAE, TimeSformer, ViViT
│   ├── train_torchvision.py          # R3D-18, MC3-18, R(2+1)D-18
│   ├── train_slowfast.py             # SlowFast R50
│   ├── run_*.sh                      # shell launchers (A100 / H200)
│   ├── slurm_*.sbatch                # Slurm batch scripts
│   └── run_post_completion_analyses.sh  # all post-training analyses
│
├── evaluations/accv2026/
│   ├── manifests/           # dataset manifests (tracked — small CSVs)
│   ├── fixed_budget/        # per-model results (gitignored — large)
│   ├── paper_results/       # compiled tables and figures (gitignored)
│   ├── confidence_cascade/  # cascade routing results (gitignored)
│   ├── knapsack_confidence/ # knapsack allocator results (gitignored)
│   └── logs/                # Slurm logs (gitignored)
│
└── docs/
    ├── ACCV_2026_RUN_STATUS.md      # ← current run status and results
    └── ACCV_2026_RESEARCH_PLAN.md
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e src/
export PYTHONPATH=src
```

Requires CUDA. Tested on A100-PCIE-40GB and H200 NVL.

---

## Models

| Model | Family | Frames | Training script |
|-------|--------|--------|-----------------|
| VideoMAE | Transformer | 16 | `train_transformers.py --model videomae` |
| TimeSformer | Transformer | 8 | `train_transformers.py --model timesformer` |
| ViViT | Transformer | 32 | `train_transformers.py --model vivit` |
| SlowFast R50 | SlowFast | 8+32 | `train_slowfast.py` |
| R(2+1)D-18 | 3D CNN | 16 | `train_torchvision.py --model r2plus1d_18` |
| R3D-18 | 3D CNN | 16 | `train_torchvision.py --model r3d_18` |
| MC3-18 | 3D CNN | 16 | `train_torchvision.py --model mc3_18` |

## Datasets

| Dataset | Classes | Domain | Status |
|---------|---------|--------|--------|
| SSV2 | 174 | Motion-centric (temporal) | ✅ All 7 models |
| UCF101 | 101 | Action (appearance) | ✅ 3 models |
| HMDB51 | 51 | Action (appearance) | ✅ 3 models |
| Diving48 | 48 | Fine-grained action | ✅ 3 models |
| EPIC-Kitchens | 97 | Egocentric (verb) | 🔄 Running |
| WLASL100 | 100 | Sign language | 🔄 Training |

---

## Running experiments

### Submit a training + eval job

Each sbatch script trains for 10 epochs, saves per-epoch checkpoints, and runs fixed-budget eval automatically at the end.

```bash
# Examples — multi-dataset
sbatch scripts/accv2026/slurm_h200_wlasl100_videomae.sbatch
sbatch scripts/accv2026/slurm_a100_wlasl100_r2plus1d.sbatch
sbatch scripts/accv2026/slurm_h200_vivit_full.sbatch   # SSV2
```

### Run all post-completion analyses

After training jobs finish, run once to generate all paper outputs:

```bash
bash scripts/accv2026/run_post_completion_analyses.sh
# with FDE routing:
bash scripts/accv2026/run_post_completion_analyses.sh --fde-routing
```

Outputs:
- `evaluations/accv2026/paper_results/` — tables and figures
- `evaluations/accv2026/confidence_cascade/` — cascade routing results
- `evaluations/accv2026/knapsack_confidence/` — knapsack allocator results

---

## Current results (fixed-budget, full training)

### SSV2 — temporally demanding

| Model | 4f | 8f | 16f | 32f |
|-------|----|----|-----|-----|
| VideoMAE | 21.0% | 39.5% | **52.3%** | 51.9% |
| TimeSformer | 31.8% | **42.3%** | 41.3% | 41.7% |
| SlowFast R50 | 6.6% | 15.2% | 33.3% | **49.5%** |
| R(2+1)D-18 | 12.6% | 24.3% | **42.6%** | 42.1% |
| ViViT | 8.4% | 17.5% | 30.5% | **38.3%** |
| R3D-18 | 9.8% | 19.7% | **37.1%** | 36.9% |
| MC3-18 | 8.2% | 18.8% | **34.5%** | 34.5% |

### UCF101 — appearance-driven (saturates early)

| Model | 4f | 8f | 16f | 32f |
|-------|----|----|-----|-----|
| VideoMAE | 81.4% | 91.4% | 95.4% | **95.5%** |
| R(2+1)D-18 | 70.0% | 81.6% | 88.6% | **89.0%** |
| SlowFast R50 | 50.1% | 66.2% | 81.3% | **87.6%** |

### HMDB51

| Model | 4f | 8f | 16f | 32f |
|-------|----|----|-----|-----|
| VideoMAE | 51.5% | 73.6% | 84.0% | **84.4%** |
| SlowFast R50 | 35.1% | 44.7% | 65.1% | **79.3%** |
| R(2+1)D-18 | 46.2% | 63.2% | 73.1% | **74.6%** |

### Diving48 — requires many frames

| Model | 4f | 8f | 16f | 32f |
|-------|----|----|-----|-----|
| SlowFast R50 | 5.8% | 14.5% | 26.4% | **50.5%** |
| VideoMAE | 8.6% | 27.6% | 48.6% | **49.9%** |
| R(2+1)D-18 | 8.6% | 16.8% | **35.3%** | 34.7% |

---

## W&B project

Project: `inforates-accv2026` — runs tagged with model, dataset, and Slurm job ID.
