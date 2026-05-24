# ACCV 2026 Experiment Tracker

This tracker separates historical ECCV artifacts, reusable results, untrusted preliminary results, and the new ACCV experiment track.

## Status Legend

- **Trusted**: can be used for analysis after checking protocol language.
- **Usable With Caveat**: useful for development or supporting evidence, but not enough for a main ACCV claim.
- **Untrusted**: should not be used in the paper without re-running or auditing.
- **Pending**: planned but not completed.

## Historical Paper Artifact

| Artifact | Status | Notes |
|---|---|---|
| `Mi3_Lab_Research_InfoRates.pdf` | Usable With Caveat | ECCV submission. Use for context only; do not reuse the overstrong Nyquist/TRA framing. |

## Existing Result Files

### UCF101

| Artifact | Status | Notes |
|---|---|---|
| `evaluations/ucf101/timesformer/fine_tuned_timesformer_ucf101_temporal_sampling.csv` | Trusted | Completed 25-config temporal sampling grid. Good for metric development. |
| `evaluations/ucf101/videomae/fine_tuned_videomae_ucf101_temporal_sampling.csv` | Trusted | Completed 25-config temporal sampling grid. Good for metric development. |
| `evaluations/ucf101/vivit/fine_tuned_vivit_ucf101_temporal_sampling.csv` | Trusted | Completed 25-config temporal sampling grid. Good for metric development. |
| `evaluations/ucf101/*/*_per_class_testset.csv` | Trusted | Good for per-class sensitivity analysis. Verify split/protocol before paper use. |
| `evaluations/ucf101/*/per_class_aliasing_drop.csv` | Trusted | Good derived artifact for sensitivity ranking. |

### Kinetics-400

| Artifact | Status | Notes |
|---|---|---|
| `evaluations/kinetics400/*/*_temporal_sampling.csv` | Usable With Caveat | Useful for analysis, but protocol uses K4TestSet/Kaggle-style data. Avoid central claims unless revalidated on canonical validation. |
| `evaluations/kinetics400/*/*_per_class.csv` | Usable With Caveat | Same protocol caveat as above. |
| `data/Kinetics400_data/k4testset/*` | Usable With Caveat | Keep as historical/supporting data, not primary ACCV evidence. |

### Spectral Analysis

| Artifact | Status | Notes |
|---|---|---|
| `src/info_rates/analysis/spectral_analysis.py` | Usable With Caveat | Reusable implementation for temporal-demand proxy, but claims should be softened. |
| `evaluations/spectral_*` | Usable With Caveat | Good for prototyping figures and metrics; verify whether each folder is synthetic, demo, or real before paper use. |
| `docs/SPECTRAL_ANALYSIS.md` | Usable With Caveat | Useful background; rewrite language away from direct Nyquist validation. |

### TRA

| Artifact | Status | Notes |
|---|---|---|
| `src/info_rates/training/temporal_augmentation.py` | Usable With Caveat | Code may be reusable as a baseline, but sampling semantics need tests. |
| `scripts/train_with_tra.py` | Usable With Caveat | Reuse only after evaluator is consolidated. |
| `docs/baseline_tra_comparison_data.md` | Untrusted | Contains known anomalies and post-fix degradation. Do not cite as result. |
| `docs/DATA_ANOMALY_ANALYSIS.md` | Trusted | Trusted as an audit document explaining why old TRA numbers should not be used. |
| `docs/TRA_SUMMARY.md` | Untrusted | Contains expected results and old claims; use only as implementation history. |

## Dataset Readiness

| Dataset | Local Status | ACCV Role | Required Action |
|---|---|---|---|
| UCF101 | Present, evaluated | Supporting dataset | Keep for continuity and metric validation. |
| Kinetics-400 | Present with K4TestSet-style data | Supporting only | Revalidate on canonical validation or demote. |
| HMDB51 | Present | Optional breadth dataset | Build/evaluate only if cheap. |
| Something-Something V2 | Downloaded/extracted to scratch; 220,847 videos available; train/validation audit has 0 missing files | Primary temporal dataset | Probe a pilot subset and freeze manifest paths before training. |
| Diving48 | RGB videos downloaded/extracted to scratch; OpenMMLab/PYSKL annotation pkl downloaded; manifest generated from pkl | Primary/secondary temporal dataset | Prefer official V2 JSONs if UCSD/OpenDataLab access becomes available; current pkl manifest is usable with caveat. |
| FineGym | Directories prepared; data not downloaded yet | Optional stretch dataset | Defer until Something-Something V2 and Diving48 are stable. |

## New ACCV Experiment Track

All new ACCV outputs should go under:

```text
evaluations/accv2026/
```

Recommended subdirectories:

```text
evaluations/accv2026/
  manifests/
  metrics/
  ucf101/
  somethingv2/
  diving48_or_finegym/
  hmdb51/
```

## Current Plan Snapshot

Last updated: 2026-05-22.

### What We Are Doing Now

We are not yet producing final ACCV tables. We are building a reliable ACCV pipeline that can survive real SSV2 videos, Slurm jobs, W&B tracking, corrupted-video edge cases, and model-specific temporal inputs.

The current pilot stage answers:

- Can the SSV2 data be decoded robustly?
- Can TimeSformer and VideoMAE train and save checkpoints on the cluster?
- Can fixed-budget evaluation produce per-sample CSVs, summary CSVs, W&B runs, and temporal metrics?
- Does accuracy change monotonically enough with temporal evidence to justify full runs?

### What Is Already Done

- SSV2 train/validation manifest is ready.
- W&B tracking works from Slurm jobs.
- Batch jobs survive notebook disconnects.
- Decoder fallback is implemented; unreadable videos are not deleted.
- Fixed-budget evaluator runs and writes sample/summary CSVs.
- `model_input_frames` is now tracked separately from evidence `processed_frames`.
- TimeSformer 10k/1-epoch pilot produced a checkpoint and fixed-budget curve.
- VideoMAE 10k/1-epoch DDP pilot produced a checkpoint and fixed-budget curve.
- ViViT 5k/1-epoch DDP pilot produced a checkpoint and fixed-budget curve.
- TorchVision 3D CNN support is implemented for `r3d_18`, `mc3_18`, and `r2plus1d_18`.
- R3D-18, MC3-18, and R(2+1)D-18 smoke tests passed on SSV2.
- R3D-18, MC3-18, and R(2+1)D-18 5k/1-epoch pilots produced checkpoints and fixed-budget curves.

### Current Pilot Results

| Model | Checkpoint | Eval Grid | Top-1 Trend | Status |
|---|---|---:|---|---|
| TimeSformer | `fine_tuned_models/accv2026_timesformer_ssv2_10k_e1` | `4,8,16,32` | 2.44%, 2.44%, 3.41%, 3.29% | Useful pilot; grid predates the cleaner `2,4,8` protocol |
| VideoMAE | `fine_tuned_models/accv2026_videomae_ssv2_10k_e1_a100ddp` | `4,8,16` | 3.05%, 5.37%, 7.44% | Useful pilot; clean architecture-valid grid |
| ViViT | `fine_tuned_models/accv2026_vivit_ssv2_5k_e1_a100ddp` | `4,8,16,32` | 1.95%, 1.95%, 2.20%, 2.68% | Useful pilot; 32-frame transformer reference |
| R3D-18 | `fine_tuned_models/accv2026_r3d18_ssv2_5k_e1_a100ddp` | `4,8,16` | 1.34%, 1.83%, 3.54% | First completed non-transformer pilot; 2xA100 train, single-A100 eval |
| MC3-18 | `fine_tuned_models/accv2026_mc3_18_ssv2_5k_e1_a100` | `4,8,16` | 1.34%, 2.68%, 2.80% | Completed non-transformer pilot; single-A100 train/eval |
| R(2+1)D-18 | `fine_tuned_models/accv2026_r2plus1d_18_ssv2_5k_e1_a100` | `4,8,16` | 2.07%, 3.29%, 4.39% | Completed non-transformer pilot; best 3D CNN pilot so far |

These accuracies are low because the models trained for only one epoch on small SSV2 subsets. Their purpose is pipeline validation, not paper performance.

### Immediate Next Steps

1. **Freeze the protocol**
   - Use evidence budget plus model input length.
   - Use architecture-valid grids.
   - Stop reporting any model comparison without `model_input_frames` and latency.

2. **Run paper-usable SSV2 training**
   - Start with VideoMAE, TimeSformer, and ViViT full/substantially larger SSV2 runs.
   - Use 2-3 epochs first, then decide whether longer training is worth the GPU time.
   - Evaluate on `somethingv2_val_20_per_class.csv`, not only `val_5_per_class`.

3. **Add stronger non-transformer baselines before claiming breadth**
   - TorchVision 3D CNN pilots now cover R3D-18, MC3-18, and R(2+1)D-18.
   - Next target: SlowFast R50.
   - Second target: X3D-S or X3D-M.
   - Use PyTorchVideo/PySlowFast adapters rather than forcing them through Hugging Face.

4. **Build the actual ACCV contribution**
   - Temporal robustness metrics are already started.
   - Next method step is a cheap temporal-demand estimator.
   - Then test adaptive budget allocation against fixed-budget baselines.

5. **Only then write the paper tables**
   - Main table: SSV2 temporal evidence curves across transformer and non-transformer families.
   - Secondary table: Diving48 or another temporal dataset.
   - Supporting table: UCF101/Kinetics historical continuity with caveats.

### What Not To Do

- Do not present 1-epoch/10k results as final evidence.
- Do not keep the ACCV submission transformer-only.
- Do not claim equal fairness by frame count alone.
- Do not center the paper on Nyquist language.
- Do not make Kinetics/Kaggle protocol a main result.

## Multi-Node Execution Runbook

The immediate multi-node GPU plan is documented in `docs/ACCV_2026_H200_RUNBOOK.md`.

Current compute resources:

| Resource | Node | GPUs | Role |
|---|---|---|---|
| H200 allocation | `gnode027` | H200 NVL | Full SSV2 TimeSformer and high-memory runs |
| A100 allocation | `gnode002` | 2x A100-PCIE-40GB | DDP pilots, VideoMAE pilot, repeated evaluation |

Tracking and persistence:

| Item | Status | Notes |
|---|---|---|
| W&B package | Ready | Installed in `.venv` |
| W&B login | Verified | `wandb.login(verify=True)` succeeds on `gnode002` and `gnode027` via `/home/wesleyferreiramaia/.netrc`; `wandb status` may still show `"api_key": null` |
| W&B phase tracking | Patched | Future run names, tags, and configs include `phase=train/eval` and `slurm_job_id`; evaluation W&B runs now start at the beginning of evaluation instead of only after all budgets finish |
| Slurm batch scripts | Ready | Use `sbatch` so jobs survive notebook sleep/disconnect |
| Submit helper | Ready | `scripts/accv2026/submit_accv2026_jobs.sh` submits H200 stage 1 and A100 VideoMAE pilot after W&B login |
| Slurm logs | Ready | `evaluations/accv2026/logs/%x-%j.out` and `%x-%j.err` |

Submitted jobs:

| Job ID | Job Name | Partition | Purpose | Status at Submission |
|---|---|---|---|---|
| `70270` | `accv26-h200-s1` | `cenvalarc.gpu` | H200 stage 1: smoke + TimeSformer SSV2 10k pilot + fixed-budget metrics | Running on `gnode027` |
| `70271` | `accv26-a100-vmae` | `gpu` | 2xA100 VideoMAE SSV2 10k DDP pilot + fixed-budget metrics | Failed before decoder fallback fix; logs in `evaluations/accv2026/logs/accv26-a100-vmae-70271.*` |
| `70276` | `accv26-a100-vmae` | `gpu` | 2xA100 VideoMAE SSV2 10k DDP pilot rerun after decoder fallback fix | Failed during fixed-budget evaluation after successfully saving `fine_tuned_models/accv2026_videomae_ssv2_10k_e1_a100ddp`; failure exposed VideoMAE fixed-frame evaluation mismatch, now fixed |
| `70277` | `accv26-h200-s1` | `cenvalarc.gpu` | H200 stage 1 rerun after evaluator W&B/fallback fixes | Slurm state `FAILED` after a shell-script tail error, but useful artifacts were produced: checkpoint, fixed-budget summary, W&B eval run, and recovered temporal metrics |
| `70295` | `accv26-a100-vmae-eval` | `gpu` | Recovery job for fixed-budget evaluation of the already saved VideoMAE 10k checkpoint | Completed; uses `budgets=4,8,16` and `model_frames=16` |
| `70312` | `accv26-a100-vivit` | `gpu` | 2xA100 ViViT SSV2 5k/1-epoch pilot + fixed-budget metrics | Slurm state `FAILED` after outputs were written because the run script was edited while the shell process was already running; checkpoint, fixed-budget summary, W&B train/eval runs, and recovered temporal metrics exist |

W&B/Slurm interpretation:

- W&B runs are phase-level records, not whole-job records.
- A `train-*` W&B run marked `Finished` means the checkpoint phase ended successfully.
- The same Slurm job may still be running the `eval-*` phase; use `squeue`, `sacct`, or `evaluations/accv2026/logs/%x-%j.*` for whole-job state.
- Future jobs include `phase` and `slurm_job_id` in W&B config, plus `train`/`eval` and `job-<id>` tags.
- Future evaluation runs are opened before budget evaluation begins and log `status/eval_started` immediately, then `status/eval_finished` at the end.

Prepared jobs:

| Script | Purpose | Notes |
|---|---|---|
| `scripts/accv2026/slurm_a100_vivit_pilot.sbatch` | 2xA100 ViViT SSV2 5k/1-epoch pilot + fixed-budget eval | Run after a tiny ViViT smoke test passes |

ViViT smoke test:

- Ran on `gnode002` with 4 train samples, 4 validation samples, batch size 1, and `--no-wandb`.
- Passed model loading, SSV2 decoding fallback, training, validation, and saving.
- Smoke checkpoint: `fine_tuned_models/smoke_vivit_ssv2_4`.

ViViT A100 pilot artifacts:

- Job: `70312`, Slurm state `FAILED` after useful artifacts were produced.
- Checkpoint: `fine_tuned_models/accv2026_vivit_ssv2_5k_e1_a100ddp`
- Training result: train loss `4.8570`, validation loss `4.7455`, validation accuracy `0.0810`.
- Summary: `evaluations/accv2026/fixed_budget/vivit_ssv2_5k_e1_a100ddp/somethingv2_validation_accv2026_vivit_ssv2_5k_e1_a100ddp_fixed_budget_summary.csv`
- Metrics: `evaluations/accv2026/fixed_budget/vivit_ssv2_5k_e1_a100ddp/temporal_metrics.csv`
- Fixed-budget top-1 on the 5-per-class SSV2 validation manifest: `0.0195` at 4 frames, `0.0195` at 8 frames, `0.0220` at 16 frames, `0.0268` at 32 frames.
- Temporal robustness AUC: `0.022648`; critical frame budget: `32`.
- W&B train run: `a100ddp-vivit-ssv2-5k-e1`; W&B eval run: `a100-eval-vivit-ssv2-5k-e1`.
- Follow-up fix: run scripts now define `ACCV_JOB_ID` with a nested default so future edits or manual launches do not fail under `set -u`.

Recovered H200 artifacts:

- Checkpoint: `fine_tuned_models/accv2026_timesformer_ssv2_10k_e1`
- Summary: `evaluations/accv2026/fixed_budget/timesformer_ssv2_10k_e1/somethingv2_validation_accv2026_timesformer_ssv2_10k_e1_fixed_budget_summary.csv`
- Metrics: `evaluations/accv2026/fixed_budget/timesformer_ssv2_10k_e1/temporal_metrics.csv`
- Result caveat: this run used the old evaluation grid `4,8,16,32` before the architecture-valid TimeSformer grid was patched to `2,4,8`.

Runtime fixes from first A100 attempt:

- `70271` failed because `decord.get_batch()` crashed on several SSV2 `.webm` files.
- The old loader also deleted unreadable files; this has been removed.
- `src/info_rates/models/timesformer.py` now falls back to OpenCV decoding when `decord` fails.
- `src/info_rates/evaluation/benchmark.py` now uses the same OpenCV fallback for fixed-budget evaluation.
- `scripts/accv2026/02_run_fixed_budget_eval.py` now logs W&B metrics from the real summary columns (`budget`, `top1`, `top5`).
- A real 2xA100 DDP smoke test passed on `gnode002` with VideoMAE, 32 train samples, and 16 validation samples.
- The `torchvision` warning is not fatal; it only means Hugging Face falls back to the slow image processor.
- The fixed-budget evaluator now records `model_input_frames` separately from `processed_frames`, so model-specific frame adaptation is explicit in future summaries.

Purpose of one-epoch jobs:

- The 1-epoch/10k jobs are not final ACCV evidence.
- Their role is to validate the full execution path: W&B, Slurm survival, 2xA100 DDP, H200 training, checkpoint saving, corrupt-video handling, and fixed-budget evaluation.
- After this validation, launch longer paper-usable runs from the same scripts with full SSV2 and more epochs.
- These jobs deliberately run short because full runs should only start after the training/evaluation path survives real SSV2 videos on the target GPUs.
- Problems found by the short jobs: W&B summary-key mismatch, SSV2 `decord` failures, unsafe deletion of unreadable videos, and VideoMAE's fixed 16-frame input requirement during fixed-budget evaluation.

Additional evaluation fix:

- VideoMAE has fixed temporal position embeddings for 16 input frames.
- Fixed-budget evaluation now separates evidence budget from model input length: e.g. `budget=4` selects 4 evidence frames and pads/repeats to the model's required 16 frames.
- A local VideoMAE evaluation smoke test with `budget=4`, `model_frames=16`, and 4 samples passed after the fix.
- Future runs use model-valid budget grids by default: TimeSformer uses `2, 4, 8`; VideoMAE uses `4, 8, 16`.
- `scripts/accv2026/slurm_a100_eval_videomae_10k.sbatch` recovers the VideoMAE evaluation from the already saved `70276` checkpoint.

Architecture expansion:

- Added `docs/ACCV_2026_ARCHITECTURE_AND_SAMPLING_PROTOCOL.md`.
- Installed `torchvision==0.23.0+cu128`, `pytorchvideo`, `fvcore`, and `iopath` in the active `.venv` after the transformer pilots finished.
- Added a TorchVision 3D CNN baseline path with `r3d_18`, `mc3_18`, and `r2plus1d_18` support.
- First non-transformer target completed: TorchVision R3D-18 pretrained on Kinetics-400 and fine-tuned on SSV2 5k/1 epoch.
- Second non-transformer target completed: TorchVision MC3-18 pretrained on Kinetics-400 and fine-tuned on SSV2 5k/1 epoch.
- Third non-transformer target completed: TorchVision R(2+1)D-18 pretrained on Kinetics-400 and fine-tuned on SSV2 5k/1 epoch.
- Next non-transformer target after the TorchVision CNN family: PyTorchVideo/PySlowFast SlowFast R50, then X3D-S or X3D-M.
- Purpose: directly answer the ECCV criticism that the study only evaluated transformer models.
- ViViT remains in the transformer set as the 32-frame transformer reference. It is useful for testing whether higher native temporal input changes the evidence-budget curve.

TorchVision 3D CNN baseline implementation:

- Model adapter: `src/info_rates/models/torchvision_video.py`
- Training script: `scripts/accv2026/03_train_torchvision_video.py`
- R3D-specific run script: `scripts/accv2026/run_a100_ssv2_r3d18_pilot_ddp.sh`
- Generic TorchVision run script: `scripts/accv2026/run_a100_ssv2_torchvision_pilot.sh`
- MC3 wrapper script: `scripts/accv2026/run_a100_ssv2_mc3_pilot.sh`
- R(2+1)D wrapper script: `scripts/accv2026/run_a100_ssv2_r2plus1d_pilot.sh`
- Slurm script: `scripts/accv2026/slurm_a100_r3d18_pilot.sbatch`
- MC3 Slurm wrapper: `scripts/accv2026/slurm_a100_mc3_pilot.sbatch`
- R(2+1)D Slurm wrapper: `scripts/accv2026/slurm_a100_r2plus1d_pilot.sbatch`
- Fixed-budget evaluator now accepts dict-style processor outputs as well as Hugging Face `BatchFeature` outputs.
- R3D-18 smoke tests passed on `gnode002`: 4 train videos, 4 validation videos, checkpoint save, checkpoint reload, and fixed-budget evaluation over budgets `4,8,16`.
- MC3-18 smoke tests passed on `gnode002`: 4 train videos, 4 validation videos, checkpoint save, and fixed-budget evaluation over budgets `4,8,16`.
- R(2+1)D-18 smoke tests passed on `gnode002`: 4 train videos, 4 validation videos, checkpoint save, and fixed-budget evaluation over budgets `4,8,16`.
- Completed R3D-18 run: `interactive70263` on `gnode002`, trained with 2xA100, `max_train_samples=5000`, `max_val_samples=1000`, `batch_size=16`, `num_frames=16`, `input_size=112`; training finished with train loss `4.7888`, validation loss `4.5582`, validation accuracy `0.1030`.
- R3D-18 fixed-budget result on the 5-per-class SSV2 validation manifest: `0.0134` at 4 frames, `0.0183` at 8 frames, `0.0354` at 16 frames; temporal robustness AUC `0.023171`; critical frame budget `16`.
- R3D-18 artifacts: `fine_tuned_models/accv2026_r3d18_ssv2_5k_e1_a100ddp`, `evaluations/accv2026/fixed_budget/r3d18_ssv2_5k_e1_a100ddp/somethingv2_validation_accv2026_r3d18_ssv2_5k_e1_a100ddp_fixed_budget_summary.csv`, and `evaluations/accv2026/fixed_budget/r3d18_ssv2_5k_e1_a100ddp/temporal_metrics.csv`.
- Completed MC3-18 clean restart: `interactive70263-mc3-restart` on `gnode002`, using GPU 1, `max_train_samples=5000`, `max_val_samples=1000`, `batch_size=16`, `num_frames=16`, `input_size=112`.
- MC3-18 fixed-budget result on the 5-per-class SSV2 validation manifest: `0.0134` at 4 frames, `0.0268` at 8 frames, `0.0280` at 16 frames; temporal robustness AUC `0.025000`; critical frame budget `8`.
- MC3-18 artifacts: `fine_tuned_models/accv2026_mc3_18_ssv2_5k_e1_a100`, `evaluations/accv2026/fixed_budget/mc3_18_ssv2_5k_e1_a100/somethingv2_validation_accv2026_mc3_18_ssv2_5k_e1_a100_fixed_budget_summary.csv`, and `evaluations/accv2026/fixed_budget/mc3_18_ssv2_5k_e1_a100/temporal_metrics.csv`.
- Completed R(2+1)D-18 run: `interactive70263-r2plus1d` on `gnode002`, using GPU 0, `max_train_samples=5000`, `max_val_samples=1000`, `batch_size=16`, `num_frames=16`, `input_size=112`; training finished with train loss `4.7355`, validation loss `4.5151`, validation accuracy `0.1150`.
- R(2+1)D-18 fixed-budget result on the 5-per-class SSV2 validation manifest: `0.0207` at 4 frames, `0.0329` at 8 frames, `0.0439` at 16 frames; temporal robustness AUC `0.034553`; critical frame budget `16`.
- R(2+1)D-18 artifacts: `fine_tuned_models/accv2026_r2plus1d_18_ssv2_5k_e1_a100`, `evaluations/accv2026/fixed_budget/r2plus1d_18_ssv2_5k_e1_a100/somethingv2_validation_accv2026_r2plus1d_18_ssv2_5k_e1_a100_fixed_budget_summary.csv`, and `evaluations/accv2026/fixed_budget/r2plus1d_18_ssv2_5k_e1_a100/temporal_metrics.csv`.
- W&B R3D train run: `train-a100ddp-r3d18-ssv2-5k-e1-jobinteractive70263`.
- W&B R3D eval run: `eval-a100-r3d18-ssv2-5k-e1-jobinteractive70263`.
- W&B MC3 train run: `train-a100-mc3-ssv2-5k-e1-restart-jobinteractive70263`.
- W&B R(2+1)D train run: `train-a100-r2plus1d-ssv2-5k-e1-jobinteractive70263`.

Execution priority:

1. Login to W&B once from the login node.
2. Submit H200 stage 1 for SSV2 TimeSformer smoke + 10k pilot.
3. Submit A100 VideoMAE DDP pilot in parallel.
4. If H200 stage 1 passes, submit H200 full SSV2 TimeSformer.
5. Use A100 for additional pilots/evaluations while H200 runs full training.
6. Defer Diving48 training/evaluation until the SSV2 pipeline is proven.

## Planned Experiments

| ID | Experiment | Dataset | Models | Status | Output |
|---|---|---|---|---|---|
| E01 | Convert existing grids to robustness metrics | UCF101, Kinetics-400 | TimeSformer, VideoMAE, ViViT | Pending | `evaluations/accv2026/metrics/` |
| E02 | Trusted fixed-budget evaluator smoke test | UCF101 subset | TimeSformer | Pending | smoke CSV + sampling test report |
| E03 | Something-Something V2 dataset audit | Something-Something V2 | n/a | Complete: 193,690 train/validation rows, 0 missing | `evaluations/accv2026/manifests/` |
| E03b | Diving48 dataset audit | Diving48 | n/a | Complete from OpenMMLab/PYSKL pkl: 16,997 annotated videos, 0 missing, 47 observed labels because label 30 is absent from the pkl | `evaluations/accv2026/manifests/` |
| E04 | Fixed-budget sweep on temporal dataset | Something-Something V2 subset/full | TimeSformer, VideoMAE, ViViT | Running pilots | accuracy-budget CSVs + W&B runs |
| E04b | Non-transformer architecture adapter | Something-Something V2 smoke subset | R3D-18, MC3-18, R(2+1)D-18, SlowFast, X3D | TorchVision CNN pilots complete; SlowFast and X3D pending | adapter smoke tests + pilot scripts |
| E05 | Temporal-demand score correlation | Something-Something V2 | same as E04 | Pending | demand-vs-critical-budget tables |
| E06 | Adaptive budget baseline | Something-Something V2 | same as E04 | Pending | adaptive-vs-fixed table |
| E07 | Optional dense-to-sparse consistency | primary temporal dataset | best model pair | Pending | ablation table |

## Paper-Readiness Gates

| Gate | Pass Condition | Status |
|---|---|---|
| Dataset Gate | At least one temporal-reasoning dataset is usable. | Passed for SSV2; Diving48 usable with annotation caveat |
| Metric Gate | Robustness metrics produce interpretable rankings. | Pending |
| Estimator Gate | Temporal-demand score correlates with observed critical budget. | Pending |
| Method Gate | Adaptive sampling improves accuracy at equal compute or reduces compute at equal accuracy. | Pending |

## Notes For Writing

- Treat TRA as a baseline, not as the main method.
- Treat UCF101/Kinetics as supporting evidence, not the central temporal reasoning proof.
- Use "temporal evidence allocation" rather than "Nyquist validation."
- Do not make efficiency claims until latency/compute measurement is fixed.
