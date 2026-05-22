# ACCV 2026 Multi-Node GPU Runbook

This is the fast execution plan for the H200 and 2xA100 windows. The goal is to turn the prepared datasets into trustworthy first results without wasting GPU time on experiments that cannot support the paper.

## Current State

- Something-Something V2 is ready on scratch and linked from `data/Something_data`.
- Diving48 RGB videos are ready on scratch and linked from `data/Diving48_data`.
- ACCV manifests are ready under `evaluations/accv2026/manifests/`.
- The fixed-budget evaluator and temporal robustness metrics are implemented.
- The first paper-usable target is Something-Something V2, because it directly answers the ECCV criticism about temporal reasoning datasets.
- Diving48 is useful as a second temporal dataset, but the current manifest is built from the OpenMMLab/PYSKL annotation pkl and has an annotation caveat.
- W&B is installed and verified through `/home/wesleyferreiramaia/.netrc` on `gnode002` and `gnode027`.

## Node Strategy

| Node Type | Best Use | Reason |
|---|---|---|
| H200 NVL | Full SSV2 TimeSformer run and high-memory sweeps | Much larger memory headroom; best for the first paper-usable full run. |
| 2xA100 40GB | DDP pilots, VideoMAE pilot, repeated evaluation jobs | Easier to obtain and good for parallel model breadth. |
| Login node | `sbatch`, `squeue`, file checks, W&B login only | Do not run training or video decoding here. |

## W&B Setup

Run this once from the login node. This stores the API key in your home directory so Slurm jobs keep logging even if your notebook sleeps.

```bash
cd /data/wesleyferreiramaia/infoRates
source .venv/bin/activate
wandb login
wandb status
```

`wandb status` may still show:

```text
"api_key": null
```

This is acceptable when `wandb login` reports credentials loaded from `.netrc`. The production check is:

```bash
python scripts/accv2026/check_wandb_login.py
```

All ACCV jobs use:

```bash
export WANDB_PROJECT=inforates-accv2026
export WANDB_MODE=online
```

The run scripts intentionally fail fast if W&B online mode is requested but `wandb.login(verify=True)` fails. This prevents a batch job from hanging at an interactive login prompt.

W&B records each phase separately. A training run marked `Finished` can be correct while the Slurm job is still running fixed-budget evaluation. Future run scripts include `phase=train/eval` and `slurm_job_id` in W&B config, and add `train`/`eval` plus `job-<id>` tags. Future evaluation runs are also opened before budget evaluation begins, so W&B shows an active `eval-*` phase while it runs. Use `squeue`, `sacct`, and the Slurm log files for whole-job state.

## Surviving Notebook Sleep

Prefer `sbatch` for long runs. A submitted Slurm batch job is owned by the scheduler, not by your local laptop SSH session.

Useful commands:

```bash
squeue -u "$USER"
tail -f evaluations/accv2026/logs/<job-name>-<job-id>.out
tail -f evaluations/accv2026/logs/<job-name>-<job-id>.err
```

If you already have an interactive allocation, `ssh gnode002` or `ssh gnode027` works for manual checks, but long training should still go through `sbatch` whenever possible.

## GPU Session Setup

Run this inside an allocated GPU node only for manual checks. For production, use the Slurm scripts below.

```bash
cd /data/wesleyferreiramaia/infoRates
source .venv/bin/activate

export PYTHONPATH=src
export HF_HOME=/scratch/wesleyferreiramaia/infoRates/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export WANDB_PROJECT=inforates-accv2026
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME" evaluations/accv2026/fixed_budget fine_tuned_models
nvidia-smi
```

## Submit Jobs

Submit the H200 stage-1 pilot:

```bash
sbatch scripts/accv2026/slurm_h200_stage1.sbatch
```

Submit the H200 full TimeSformer run:

```bash
sbatch scripts/accv2026/slurm_h200_timesformer_full.sbatch
```

Submit the A100 TimeSformer DDP pilot:

```bash
sbatch scripts/accv2026/slurm_a100_timesformer_pilot.sbatch
```

Submit the A100 VideoMAE DDP pilot:

```bash
sbatch scripts/accv2026/slurm_a100_videomae_pilot.sbatch
```

Recommended parallel schedule:

1. Login to W&B once.
2. Submit `slurm_h200_stage1.sbatch`.
3. Submit `slurm_a100_videomae_pilot.sbatch`.
4. If H200 stage 1 passes, submit `slurm_h200_timesformer_full.sbatch`.
5. Use A100 for additional pilots/evaluations while H200 runs the full model.

After W&B login, the first two jobs can be submitted together with:

```bash
bash scripts/accv2026/submit_accv2026_jobs.sh
```

Current submitted jobs:

| Job ID | Job Name | Purpose |
|---|---|---|
| `70270` | `accv26-h200-s1` | H200 stage 1: smoke + TimeSformer SSV2 10k pilot + fixed-budget metrics. Running on `gnode027`. |
| `70271` | `accv26-a100-vmae` | 2xA100 VideoMAE SSV2 10k DDP pilot + fixed-budget metrics. Failed before decoder fallback fix. |
| `70276` | `accv26-a100-vmae` | 2xA100 VideoMAE SSV2 10k DDP pilot rerun after decoder fallback fix. Failed during fixed-budget evaluation after saving `fine_tuned_models/accv2026_videomae_ssv2_10k_e1_a100ddp`; use `slurm_a100_eval_videomae_10k.sbatch` to recover evaluation. |
| `70277` | `accv26-h200-s1` | H200 stage 1 rerun after evaluator W&B/fallback fixes. Slurm state `FAILED` after a shell-script tail error, but useful artifacts were produced: checkpoint, fixed-budget summary, W&B eval run, and recovered temporal metrics. |
| `70295` | `accv26-a100-vmae-eval` | Recovery fixed-budget evaluation for the saved VideoMAE 10k checkpoint. Running on `gnode003` with `budgets=4,8,16` and `model_frames=16`. |

Recovered H200 artifacts:

- `fine_tuned_models/accv2026_timesformer_ssv2_10k_e1`
- `evaluations/accv2026/fixed_budget/timesformer_ssv2_10k_e1/somethingv2_validation_accv2026_timesformer_ssv2_10k_e1_fixed_budget_summary.csv`
- `evaluations/accv2026/fixed_budget/timesformer_ssv2_10k_e1/temporal_metrics.csv`
- Caveat: this H200 evaluation used the pre-patch grid `4,8,16,32`; future TimeSformer runs use `2,4,8` with `model_frames=8`.

The H200 batch job is now running on `gnode027`.

Decoder note:

- The first A100 attempt exposed SSV2 `.webm` files where `decord.get_batch()` fails.
- The training dataset now falls back to OpenCV decoding and never deletes unreadable files.
- The fixed-budget evaluator now uses the same OpenCV fallback.
- A 2xA100 DDP smoke test with VideoMAE completed successfully after the fix.
- The `torchvision` warning is only a performance warning; training still runs with the slow processor.

One-epoch scope:

- The 1-epoch/10k jobs are validation jobs, not final paper evidence.
- They verify the end-to-end system before spending the scarce H200 window on full runs.
- Once validation passes, launch the full TimeSformer and VideoMAE runs.
- They also intentionally expose dataset/decoder/model-shape failures cheaply. In this pass they found four issues before full training: W&B metric naming, `decord` failures on SSV2 `.webm`, unsafe deletion of unreadable videos, and VideoMAE fixed-frame evaluation.

VideoMAE fixed-frame note:

- VideoMAE expects 16 input frames because of fixed temporal position embeddings.
- For temporal-budget evaluation, the evaluator now selects the requested evidence budget and pads/repeats frames to the model's required input length.
- Example: `budget=4` means 4 distinct evidence frames, delivered to VideoMAE as 16 frames after padding.
- Future evaluation scripts use architecture-valid grids: TimeSformer `2, 4, 8`; VideoMAE `4, 8, 16`.

## Priority 0: CPU/GPU Smoke Checks

These checks are not paper results. They only prove that decoding, model loading, batching, and output writing work on the GPU node.

```bash
PYTHONPATH=src python -m compileall -q \
  src/info_rates/evaluation/benchmark.py \
  src/info_rates/metrics/temporal_robustness.py \
  scripts/accv2026/02_run_fixed_budget_eval.py \
  scripts/accv2026/04_compute_temporal_metrics.py
```

```bash
python scripts/accv2026/02_run_fixed_budget_eval.py \
  --manifest evaluations/accv2026/manifests/somethingv2_val_5_per_class.csv \
  --dataset-name somethingv2 \
  --split validation \
  --model timesformer \
  --num-labels 174 \
  --budgets 4 8 \
  --batch-size 16 \
  --max-samples 32 \
  --output-dir evaluations/accv2026/fixed_budget/smoke_timesformer_ssv2
```

Stop and fix infrastructure only if this fails. Do not interpret the accuracy because this run uses a randomly initialized classification head for SSV2.

## Priority 1: First Useful SSV2 Checkpoint

This is the fastest useful checkpoint. It is not the final paper model, but it can produce real temporal sensitivity curves.

```bash
python scripts/train_something.py \
  --data-root data/Something_data \
  --model timesformer \
  --epochs 1 \
  --batch-size 32 \
  --lr 2e-5 \
  --num-workers 8 \
  --max-train-samples 10000 \
  --max-val-samples 2000 \
  --save-path fine_tuned_models/accv2026_timesformer_ssv2_10k_e1 \
  --no-wandb
```

If the job is stable and GPU memory is low, raise `--batch-size` to `48` or `64`. If it runs out of memory, lower it to `16` and keep going.

Evaluate the checkpoint:

```bash
python scripts/accv2026/02_run_fixed_budget_eval.py \
  --manifest evaluations/accv2026/manifests/somethingv2_val_5_per_class.csv \
  --dataset-name somethingv2 \
  --split validation \
  --checkpoint fine_tuned_models/accv2026_timesformer_ssv2_10k_e1 \
  --budgets 2 4 8 \
  --model-frames 8 \
  --batch-size 32 \
  --output-dir evaluations/accv2026/fixed_budget/timesformer_ssv2_10k_e1
```

Compute temporal metrics:

```bash
python scripts/accv2026/04_compute_temporal_metrics.py \
  --summary evaluations/accv2026/fixed_budget/timesformer_ssv2_10k_e1/somethingv2_validation_accv2026_timesformer_ssv2_10k_e1_fixed_budget_summary.csv \
  --output evaluations/accv2026/fixed_budget/timesformer_ssv2_10k_e1/temporal_metrics.csv
```

Pass condition: the summary file exists, temporal AUC is computed, and accuracy increases sensibly as frame budget grows.

## Priority 2: First Paper-Usable SSV2 Run

Start this as soon as Priority 1 passes.

```bash
python scripts/train_something.py \
  --data-root data/Something_data \
  --model timesformer \
  --epochs 3 \
  --batch-size 48 \
  --lr 2e-5 \
  --num-workers 12 \
  --save-path fine_tuned_models/accv2026_timesformer_ssv2_full_e3 \
  --no-wandb
```

If the H200 handles `--batch-size 64`, use it. If the dataloader becomes the bottleneck, try `--num-workers 16`.

Evaluate on the stronger balanced validation subset:

```bash
python scripts/accv2026/02_run_fixed_budget_eval.py \
  --manifest evaluations/accv2026/manifests/somethingv2_val_20_per_class.csv \
  --dataset-name somethingv2 \
  --split validation \
  --checkpoint fine_tuned_models/accv2026_timesformer_ssv2_full_e3 \
  --budgets 2 4 8 \
  --model-frames 8 \
  --batch-size 32 \
  --output-dir evaluations/accv2026/fixed_budget/timesformer_ssv2_full_e3
```

```bash
python scripts/accv2026/04_compute_temporal_metrics.py \
  --summary evaluations/accv2026/fixed_budget/timesformer_ssv2_full_e3/somethingv2_validation_accv2026_timesformer_ssv2_full_e3_fixed_budget_summary.csv \
  --output evaluations/accv2026/fixed_budget/timesformer_ssv2_full_e3/temporal_metrics.csv
```

## Priority 3: Second Architecture

Run this after TimeSformer has a valid curve. VideoMAE gives architectural breadth while staying within the existing code.

```bash
python scripts/train_something.py \
  --data-root data/Something_data \
  --model videomae \
  --epochs 2 \
  --batch-size 24 \
  --lr 2e-5 \
  --num-workers 12 \
  --save-path fine_tuned_models/accv2026_videomae_ssv2_full_e2 \
  --no-wandb
```

```bash
python scripts/accv2026/02_run_fixed_budget_eval.py \
  --manifest evaluations/accv2026/manifests/somethingv2_val_20_per_class.csv \
  --dataset-name somethingv2 \
  --split validation \
  --checkpoint fine_tuned_models/accv2026_videomae_ssv2_full_e2 \
  --budgets 4 8 16 \
  --model-frames 16 \
  --batch-size 24 \
  --output-dir evaluations/accv2026/fixed_budget/videomae_ssv2_full_e2
```

```bash
python scripts/accv2026/04_compute_temporal_metrics.py \
  --summary evaluations/accv2026/fixed_budget/videomae_ssv2_full_e2/somethingv2_validation_accv2026_videomae_ssv2_full_e2_fixed_budget_summary.csv \
  --output evaluations/accv2026/fixed_budget/videomae_ssv2_full_e2/temporal_metrics.csv
```

## Priority 4: Diving48 Support Run

Only start this after at least one clean SSV2 model exists. This is supporting evidence, not the first result to chase.

For now, use Diving48 for fixed-budget evaluation only after a proper Diving48 training path is added. Do not evaluate a random head and call it a result.

Target outputs:

```text
evaluations/accv2026/fixed_budget/diving48_*/
fine_tuned_models/accv2026_*_diving48_*/
```

## What To Watch During Runs

- Keep `nvidia-smi` open in another shell if possible.
- If GPU utilization is low and memory is low, increase `--batch-size`.
- If GPU utilization is low but memory is high, increase `--num-workers` first.
- If the dataloader crashes on a corrupt video, preserve the error log and continue with a filtered manifest rather than deleting large chunks of data.
- Every paper-usable result needs both the per-sample CSV and the summary CSV from the evaluator.

## Result Validity Rules

- Smoke runs are infrastructure-only.
- A valid SSV2 result requires a checkpoint trained with `num_labels=174`.
- A valid Diving48 result requires a checkpoint trained with the Diving48 label space used by its manifest.
- The main ACCV claim should be about temporal evidence allocation and robustness curves, not Nyquist recovery or TRA as a new method.
- UCF101 and Kinetics-400 should stay as historical/supporting evidence unless re-run with clean protocols.
