# scripts/accv2026 — Script Catalog

Scripts for the ACCV 2026 experiment pipeline. Organized below by function.
Completed one-off scripts are in `archive/`.

---

## 1. Data Pipeline (numbered, run in order)

| Script | Purpose |
|--------|---------|
| `00_prepare_datasets.py` | Dataset path resolution and validation |
| `01_audit_datasets.py` | Verify video files, log corrupted/missing |
| `02_make_manifests.py` | Build 20-clip-per-class stratified validation splits |
| `04_compute_temporal_demand.py` | Compute TDS per dataset × model |
| `05_compute_temporal_metrics.py` | Per-clip temporal sensitivity metrics |
| `06_fde_adaptive_routing.py` | Frame-drop entropy routing (C4) |
| `07_dataset_temporal_demand.py` | Aggregate TDS stats across models |
| `08_compile_paper_results.py` | Collect sweep CSVs into `dashboard/data/` |
| `09_plot_paper_figures.py` | Generate paper figures (Figs 1–5) |
| `10_per_class_temporal_analysis.py` | Per-class sensitivity breakdown |
| `11_spectral_router.py` | Spectral-guided adaptive sampler |
| `12_confidence_cascade.py` | Confidence cascade routing |
| `13_knapsack_confidence.py` | Budget-constrained frame selection |
| `14_plot_routing_comparison.py` | Plot routing vs. baseline curves |
| `15_baseline_comparison.py` | AdaFocus / AR-Net / FrameExit comparison |
| `16_latency_analysis.py` | Per-model latency at each resolution |

---

## 2. Training

| Script | Purpose |
|--------|---------|
| `train_torchvision.py` | Train R3D-18, MC3-18, R(2+1)D-18 |
| `train_slowfast.py` | Train SlowFast-R50 |
| `train_transformers.py` | Train TimeSformer, ViViT, VideoMAE |
| `train_videomamba.py` | Train VideoMamba (requires `.venv_mamba`) |
| `train_all_models.py` | Orchestrator: calls the above 4 scripts |

---

## 3. Sweeps (main experiment runners)

| Script | Purpose |
|--------|---------|
| `sweep_coverage_stride.py` | **Core sweep** — evaluates (coverage, stride) grid at native resolution; reads checkpoints via `get_checkpoint()` |
| `sweep_spatial_resolution.py` | Spatial sweep — evaluates native checkpoint at each resolution; `--resolutions` arg; aggregate rebuilds from all `res*_summary.csv` |
| `sweep_coverage_stride_resolution.py` | Combined (resolution, coverage, stride) sweep for multi-resolution campaign |
| `sweep_all_models_coverage_stride_resolution.py` | Batch orchestrator for multi-model × multi-dataset |

---

## 4. Analysis & Statistics

| Script | Purpose |
|--------|---------|
| `analyze_tds_robustness.py` | TDS stability: family ablation (CNN-only ρ=0.976; Transformer-only ρ=1.000) + bootstrap CI (B=10,000) |
| `analyze_nyquist_spectral_v2.py` | **Spectral test v2** — 7 datasets × 5 resolutions = 35 points; ρ=−0.549, p=0.0006 |
| `analyze_nyquist_finegym_flowonly.py` | **Run on other PC** — FineGym optical flow cutoff freq; requires FineGym videos |
| `analyze_nyquist_spectral.py` | Spectral test v1 (n=7, underpowered; kept for reference) |
| `analyze_all_models_sweep.py` | Aggregate sweep results across all models |
| `analyze_trainres_sweep.py` | Analyze resolution-retrain sweep results |
| `merge_finegym_nyquist.py` | Merge `finegym_cutoff_freq.csv` into `nyquist_resolution_validation.csv` after FineGym run |
| `e2_variance_analysis.py` | Levene variance inflation analysis |
| `e3_spectral_analysis.py` | Optical flow spectral analysis (per dataset) |
| `e4_anova_analysis.py` | ANOVA η² effect size computation |
| `e5_taxonomy_analysis.py` | Action sensitivity taxonomy |
| `e7_entropy_routing.py` | Entropy routing evaluation |
| `e9_baseline_comparison.py` | Baseline routing comparison |
| `e10_clip_duration.py` | Clip duration vs. aliasing correlation |
| `benchmark_latency_by_resolution.py` | Measure inference latency at each resolution |
| `generate_paper_figures.py` | High-res paper figures (alternative to 09_) |

---

## 5. Evaluation

| Script | Purpose |
|--------|---------|
| `eval_native_models_at_48px.py` | Evaluate all native checkpoints at 48px (cross-res, no retrain) |
| `eval_p3_retrained.py` | Evaluate P3-retrained checkpoints at their training resolution |
| `eval_fixed_budget.py` | Fixed-frame-budget evaluation (all models) |

---

## 6. Data Preparation

| Script | Purpose |
|--------|---------|
| `preprocess_autsl.py` | AUTSL .avi → .mp4, normalize paths |
| `preprocess_driveact.py` | DriveAct clip extraction |
| `preprocess_ego4d.py` | Ego4D FHO segment extraction |
| `build_video_datasets.py` | Generic dataset builder |
| `create_full_manifests.py` | Build manifests for new datasets |
| `audit_new_datasets.py` | Verify new dataset integrity |
| `download_epic_clips.py` | Download EPIC-Kitchens validation clips |
| `download_finegym.py` | Download FineGym videos |
| `download_finegym_flow_sample.py` | Download FineGym sample for flow analysis |
| `integrate_finegym_to_dashboard.py` | Add FineGym rows to dashboard CSVs |
| `check_wandb_login.py` | Verify W&B authentication |

---

## 7. Slurm Job Templates

| Script | Purpose |
|--------|---------|
| `slurm_h200_single.sbatch` | **Main template** — single H200 GPU job (used by most submit daemons) |
| `slurm_coverage_stride_sweep.sbatch` | Temporal sweep job (1 GPU, ~4h) |
| `slurm_spatial_resolution_sweep.sbatch` | Spatial sweep job (1 GPU, ~2h) |
| `slurm_h200_resolution_retrain.sbatch` | Resolution retraining (1 GPU, ~3h) |
| `slurm_h200_retrain_all.sbatch` | Batch retraining template |
| `slurm_336px_2gpu.sbatch` | P3 retraining at 336px — 2 GPU DDP, batch=64 |
| `slurm_h200_cnn_224px.sbatch` | CNN 224px retraining |
| `slurm_native_at_48px.sbatch` | 48px inference on native checkpoints |
| `slurm_res_cov_stride.sbatch` | Multi-resolution coverage×stride sweep |
| `slurm_trainres_sweep.sbatch` | Training-resolution analysis sweep |
| `slurm_spatial_sweep.sbatch` | Spatial resolution sweep (alternative) |
| `slurm_h200_videomamba_*.sbatch` | VideoMamba-specific jobs (various configs) |
| `slurm_h200_videomae_epic_reeval.sbatch` | VideoMAE EPIC-Kitchens re-evaluation |
| `slurm_videomamba_autsl_deep.sbatch` | VideoMamba AUTSL deep sweep |

---

## 8. Daemon Submit Scripts

These scripts manage job queues with a concurrency limit (typically MAX=4-8).

| Script | Purpose |
|--------|---------|
| `submit_sweep_daemon.sh` | Main temporal sweep daemon (8 models × 8 datasets) |
| `submit_sweep_rerun_daemon.sh` | Rerun failed sweep configs |
| `submit_spatial_48px_daemon.sh` | 48px inference daemon (7 datasets) |
| `submit_p3_336px_2gpu.sh` | P3 retraining 336px daemon (MAX=4 concurrent) |
| `submit_retrain_fixes_daemon.sh` | Anomaly retraining daemon |
| `submit_spatial_sweep_all_datasets.sh` | Spatial sweep across all datasets |
| `submit_missing_trainres224_daemon.sh` | Missing 224px training-res configs |
| `submit_master_daemon.sh` | Master orchestrator |
| `master_auto_submit.sh` | Legacy master launcher |
| `auto_submit_all.sh` | Auto-submit all pending jobs |
| `auto_submit_final.sh` | Final pass auto-submit |
| `submit_native_at_48px.sh` | Submit 48px eval for all native checkpoints |
| `submit_combined_sweep.sh` | Combined sweep submission |
| `submit_missing_jobs.sh` | Resubmit missing/failed jobs |
| `submit_h200_all_resolutions.sh` | All-resolution H200 submit |
| `submit_a100_retrain.sh` | A100 retraining campaigns |
| `submit_ego4d_retrain.sh` | Ego4D retraining |

---

## 9. VideoMamba-specific

| Script | Purpose |
|--------|---------|
| `train_videomamba.py` | VideoMamba training (requires `.venv_mamba`) |
| `slurm_h200_videomamba_all_datasets.sbatch` | VideoMamba sweep all 8 datasets |
| `slurm_h200_videomamba_dataset.sbatch` | VideoMamba single-dataset job |
| `slurm_h200_videomamba_eval.sbatch` | VideoMamba evaluation |
| `submit_videomamba_all_parallel.sh` | Submit all VideoMamba jobs |
| `submit_videomamba_retrain_v2.sh` | VideoMamba v2 checkpoint retraining |
| `submit_videomamba_pending_evals.sh` | Resume pending VideoMamba evals |

---

## archive/

Scripts that completed their one-time purpose and will not be run again.
Kept for reproducibility reference only — do not use directly.

---

## Common Patterns

**To submit a new sweep job:**
```bash
# Single job
sbatch slurm_h200_single.sbatch MODEL=timesformer DATASET=ssv2 TRAIN_RES=224

# Daemon (MAX=4 concurrent)
nohup bash submit_sweep_daemon.sh > evaluations/accv2026/logs/daemon.log 2>&1 &
```

**To add FineGym spectral data (requires other PC):**
```bash
# On the other PC (where FineGym videos are):
python3 scripts/accv2026/analyze_nyquist_finegym_flowonly.py
# Then copy evaluations/accv2026/e3_spectral/finegym_cutoff_freq.csv back here
python3 scripts/accv2026/merge_finegym_nyquist.py
```

**Python environment:**
- Most scripts: `source .venv/bin/activate`
- VideoMamba only: `source .venv_mamba/bin/activate`
