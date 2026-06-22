# VideoMamba3: Adapting Mamba-3 State Space Models for Efficient Video Understanding

## Abstract Draft

Video understanding models must integrate long-range temporal structure while remaining efficient over dense spatiotemporal token streams. We introduce VideoMamba3, a video classification backbone that adapts the Mamba-3 state space model to the VideoMamba architecture. VideoMamba3 replaces the original bidirectional Mamba mixer with a Mamba-3-style recurrent mixer using trapezoidal discretization, data-dependent complex rotations, and optional low-rank MIMO state updates. We evaluate the resulting design across fixed temporal budgets and report accuracy/latency trade-offs relative to VideoMamba and transformer video baselines.

## Claims To Validate

- VideoMamba3 trains end-to-end in the existing ACCV video pipeline.
- Mamba-3 dynamics improve temporal modeling over VideoMamba under matched frame budgets.
- The complex/RoPE and MIMO variants provide measurable gains over trapezoidal-only SSMs.
- The current pure-PyTorch scan is correct but slow; production-scale efficiency requires fused or chunked scan kernels.

## Implemented Variants

- `trapezoidal`: Mamba-3-style trapezoidal state update without complex rotation.
- `complex`: trapezoidal update plus data-dependent RoPE over Q/K state channels.
- `mimo`: complex variant with low-rank MIMO projections.

## Current Smoke Result

Run `72135` on H200 completed successfully:

- Dataset: UCF101 subset
- Model: `VideoMamba3-tiny`
- Variant: `complex`
- Input: `112x112`, `2` frames
- Train/val samples: `128/32`
- Epochs: `1`
- Train loss: `2.3193`
- Validation loss: `1.1055`
- Validation accuracy: `1.0000`
- Runtime: `11m20s`

This is a functional smoke test, not a publishable accuracy result.

## Validation First

The current priority is not release packaging. VideoMamba3 still needs a clean
validation stack before any public checkpoint:

- controlled forward latency;
- training-step latency;
- peak allocated and reserved VRAM;
- throughput in videos/s and tokens/s;
- fixed-budget evaluation on real validation manifests;
- matched VideoMamba baseline comparisons.

Release/Hugging Face export remains available as infrastructure, but it is
not a paper claim and should not be used until the validation gates pass.

## Experiment Plan

1. Run system validation on H200: inference latency, train-step latency, VRAM.
2. Promote `complex` as the primary candidate because it beat `trapezoidal` and `mimo` in the first UCF101 ablation.
3. Train a fast candidate: `depth=8`, `112x112`, `4` frames, larger UCF101 subset.
4. Run fixed-budget eval against VideoMamba at matched input shape and dataset.
5. Only then scale to `224x224`, `8+` frames, and larger datasets.

## Engineering Roadmap

- Reference implementation: pure-PyTorch Mamba-3 scan, correct and inspectable.
- Fast candidate: `complex`, `depth=8`, `112x112`, `4` frames.
- Validation gate: latency, train-step VRAM, fixed-budget evaluation, matched VideoMamba baseline.
- Performance gate: replace Python scan with fused/chunked recurrence if the accuracy trend is strong.
- Release gate: public weights only after real validation.

## Commands

Submit ablations:

```bash
bash scripts/accv2026/submit_videomamba3_ablation_h200.sh
```

Submit latency benchmark:

```bash
sbatch scripts/accv2026/slurm_h200_videomamba3_benchmark.sbatch
```

Submit system validation:

```bash
sbatch scripts/accv2026/slurm_h200_videomamba3_system_validation.sbatch
```

Submit fast UCF101 candidate:

```bash
sbatch scripts/accv2026/slurm_h200_videomamba3_fast_ucf101.sbatch
```

Compile tables:

```bash
python scripts/accv2026/18_compile_videomamba3_results.py
```

Outputs:

- `evaluations/accv2026/videomamba3/training_summary.csv`
- `evaluations/accv2026/videomamba3/latency_benchmark.csv`
- `evaluations/accv2026/videomamba3/system_validation*.csv`
- `evaluations/accv2026/videomamba3/paper_tables.md`
