# InfoRates — Temporal Evidence Allocation for Video Recognition

Research code for ACCV 2026 submission.

**Core question:** how many frames does a video model actually need to make a correct prediction, and how does that vary across model families?

We measure this with *fixed-budget evaluation*: force a model to decide using only `k` frames sampled uniformly from the video, and sweep `k` from small to large. The resulting accuracy-vs-frames curve — and its area (temporal AUC) — quantifies each model's temporal evidence demand.

---

## Repository layout

```
infoRates/
├── src/info_rates/          # Python package (models, evaluation, data utils)
│   ├── models/              # timesformer, videomae, vivit, torchvision_video, slowfast_video
│   ├── evaluation/          # benchmark.py — fixed-budget evaluator
│   └── data/                # SSV2, UCF101, HMDB51 dataset readers
│
├── scripts/accv2026/        # ACCV 2026 experiment pipeline
│   ├── 00_prepare_datasets.py
│   ├── 01_audit_datasets.py
│   ├── 02_make_manifests.py
│   ├── 02_run_fixed_budget_eval.py   # main evaluator entrypoint
│   ├── 04_compute_temporal_demand.py
│   ├── 05_compute_temporal_metrics.py
│   ├── 06_demand_vs_budget.py
│   ├── 07_build_comparison_table.py
│   ├── train_something.py            # transformer training (TimeSformer, VideoMAE, ViViT)
│   ├── train_torchvision.py          # 3D CNN training (R3D-18, MC3-18, R(2+1)D-18)
│   ├── train_slowfast.py             # SlowFast R50 training
│   ├── run_*.sh                      # shell launchers (A100 / H200)
│   ├── slurm_*.sbatch                # Slurm batch scripts
│   └── sync_wandb.sh                 # sync W&B runs after job finishes
│
├── evaluations/accv2026/
│   ├── manifests/           # dataset manifests (tracked — small CSVs)
│   ├── fixed_budget/        # per-model results (gitignored — large)
│   ├── metrics/             # comparison tables (gitignored — generated)
│   └── logs/                # Slurm logs (gitignored)
│
├── docs/                    # ACCV 2026 documentation
│   ├── ACCV_2026_EXPERIMENT_TRACKER.md
│   ├── ACCV_2026_RESEARCH_PLAN.md
│   ├── ACCV_2026_DATASET_PREP.md
│   ├── ACCV_2026_H200_RUNBOOK.md
│   ├── ACCV_2026_ARCHITECTURE_AND_SAMPLING_PROTOCOL.md
│   └── legacy/              # archived ECCV-era docs
│
└── scripts/legacy/          # archived scripts from ECCV phase
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e src/
```

Requires CUDA. Tested on A100-PCIE-40GB and H200 NVL.

---

## Models supported

| Model | Family | Input frames | Training script |
|---|---|---|---|
| TimeSformer | Transformer | 8 | `train_something.py --model timesformer` |
| VideoMAE | Transformer | 16 | `train_something.py --model videomae` |
| ViViT | Transformer | 32 | `train_something.py --model vivit` |
| R3D-18 | 3D CNN | 16 | `train_torchvision.py --model r3d_18` |
| MC3-18 | 3D CNN | 16 | `train_torchvision.py --model mc3_18` |
| R(2+1)D-18 | 3D CNN | 16 | `train_torchvision.py --model r2plus1d_18` |
| SlowFast R50 | SlowFast | 8+32 | `train_slowfast.py` |

All models are fine-tuned on Something-Something V2 (SSV2).

---

## Running experiments

### 1. Prepare dataset manifests

```bash
python scripts/accv2026/00_prepare_datasets.py
python scripts/accv2026/01_audit_datasets.py
python scripts/accv2026/02_make_manifests.py
```

### 2. Submit training jobs (Slurm)

```bash
# 3D CNN baselines on 2x A100
sbatch scripts/accv2026/slurm_a100_r2plus1d_full.sbatch
sbatch scripts/accv2026/slurm_a100_r3d18_full.sbatch
sbatch scripts/accv2026/slurm_a100_mc3_full.sbatch

# Transformers on H200
sbatch scripts/accv2026/slurm_h200_timesformer_full.sbatch
sbatch scripts/accv2026/slurm_h200_videomae_full.sbatch
sbatch scripts/accv2026/slurm_h200_vivit_full.sbatch
```

### 3. Sync W&B after jobs finish (run from login node)

```bash
bash scripts/accv2026/sync_wandb.sh
```

Compute nodes have no internet; W&B saves locally and this syncs everything at once.

### 4. Build comparison table

```bash
python scripts/accv2026/07_build_comparison_table.py
```

---

## Pilot results (1 epoch / 5-10k samples — pipeline validation only)

| Model | Family | 4 frames | 8 frames | 16 frames | Temporal AUC |
|---|---|---|---|---|---|
| VideoMAE | Transformer | 3.05% | 5.37% | 7.44% | 5.67% |
| R(2+1)D-18 | 3D CNN | 2.07% | 3.29% | 4.39% | 3.46% |
| TimeSformer | Transformer | 2.44% | 2.44% | 3.41% | 3.10% |
| MC3-18 | 3D CNN | 1.34% | 2.68% | 2.80% | 2.50% |
| R3D-18 | 3D CNN | 1.34% | 1.83% | 3.54% | 2.32% |
| ViViT | Transformer | 1.95% | 1.95% | 2.20% | 2.26% |

Low accuracy is expected: these are pipeline-validation pilots (1 epoch, tiny subset). Full runs use the complete SSV2 training set and 5 epochs.

---

## W&B project

Project: `inforates-accv2026` — runs are tagged with `train`/`eval`, model name, and Slurm job ID.
