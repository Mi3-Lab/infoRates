# ACCV 2026 Architecture and Temporal Sampling Protocol

This note answers two reviewer-facing questions:

1. How do we compare models that expect different numbers of input frames?
2. How do we address the ECCV criticism that the study only used transformer models?

## Core Decision

Do not force every architecture into the same tensor length and do not treat native frame count as equal compute.

The ACCV protocol should separate:

- **Evidence budget**: number of distinct decoded frames selected from the original video.
- **Model input length**: number of frames actually passed to the model after padding, repetition, pathway construction, or native preprocessing.
- **Temporal span**: source-video duration covered by the selected frames.
- **Compute proxy**: model input frames, pathway frames, inference time, and optionally FLOPs.

This keeps the scientific question clean:

> How much distinct temporal evidence does a model need, given its own architecture-specific input interface?

## Why Model-Specific Frame Counts Are Acceptable

Video action models are not interchangeable frame consumers. TimeSformer, VideoMAE, ViViT, SlowFast, X3D, and I3D encode time differently.

Forcing all models to the same frame count can create artificial failures:

- TimeSformer checkpoints are commonly configured around 8-frame clips.
- VideoMAE checkpoints commonly expect 16-frame clips because of fixed temporal position embeddings.
- SlowFast consumes two temporal rates at once, e.g. slow pathway plus fast pathway.
- X3D and I3D consume 3D convolutional clips with architecture-specific temporal kernels and strides.

Therefore, the evaluator should report both:

- `budget`: distinct evidence frames selected.
- `model_input_frames`: frames actually consumed by the model after adaptation.

The current evaluator now records both `mean_processed_frames` and `mean_model_input_frames`. In the current code path, `processed_frames` means evidence frames before model adaptation.

## Two Complementary Evaluation Views

### View A: Evidence-Budget Robustness

Question:

> If the video sensor or sampler exposes only K distinct frames, how robust is each model?

Protocol:

- Decode exactly K distinct evidence frames when possible.
- Adapt those K frames to the model's required input shape.
- Report accuracy, top-5, confidence, latency, processed evidence frames, and model input frames.

Recommended grids:

| Model | Native Input | Evidence Budgets |
|---|---:|---|
| TimeSformer | 8 | 2, 4, 8 |
| VideoMAE | 16 | 4, 8, 16 |
| ViViT | 32 | 4, 8, 16, 32 |
| SlowFast | 8 slow + 32 fast | 4, 8, 16, 32 evidence frames; convert to pathways |
| X3D | typically 16 or 32 depending checkpoint | 4, 8, 16 or checkpoint-native maximum |
| I3D / R(2+1)D | typically 16, 32, or 64 depending checkpoint | checkpoint-native grid |

This view is the main temporal evidence claim.

### View B: Compute-Normalized Robustness

Question:

> At equal or measured compute, which model gives the best accuracy-budget trade-off?

Protocol:

- Report mean inference time and decoded/evidence frames.
- Use measured GPU latency from the evaluator.
- Add FLOPs only if a stable profiler can be integrated without delaying the main experiments.

This view prevents unfair claims such as "VideoMAE is worse at 4 frames" without acknowledging that it still receives a padded 16-frame input.

## Architecture Expansion Plan

The ACCV paper should not remain transformer-only. At minimum, include one strong non-transformer family.

### Required Families

| Family | Candidate | Why It Matters | Implementation Path |
|---|---|---|---|
| Transformer | TimeSformer or VideoMAE | Continuity with ECCV results | Existing Hugging Face path |
| Two-pathway 3D CNN | SlowFast R50 | Explicit slow/fast temporal-rate design | PyTorchVideo or PySlowFast |
| Efficient 3D CNN | X3D-S or X3D-M | Tests efficient temporal modeling | PyTorchVideo or PySlowFast |

### Optional Families

| Family | Candidate | Why It Matters |
|---|---|---|
| Inflated 3D CNN | I3D / 3D ResNet | Reviewer directly asked for I3D/C3D-style baselines |
| Temporal-shift CNN | TSM | Strong 2D-CNN temporal baseline |
| Hybrid / multiscale | MViT or Video Swin | Strong modern architecture breadth |

## Recommended Minimum Paper Matrix

For ACCV, the smallest credible matrix is:

| Dataset | Transformer | Non-Transformer | Efficient Model |
|---|---|---|---|
| Something-Something V2 | VideoMAE or TimeSformer | SlowFast R50 | X3D-S/M |
| Diving48 | best transformer from SSV2 | SlowFast or X3D | optional |
| UCF101 | reuse existing transformers | add one non-transformer if cheap | optional |

This directly addresses the ECCV criticism:

- not only UCF101/Kinetics
- not only transformers
- not only a benchmark table
- includes temporally demanding datasets
- includes architecture families with different temporal inductive biases

## Implementation Notes

The current `ModelFactory` lists `slowfast` and `x3d`, but those entries are not currently loadable through the Hugging Face `AutoModelForVideoClassification` path.

Next implementation steps:

1. Add an architecture adapter interface:
   - `prepare_inputs(frames, device)`
   - `forward(inputs)`
   - `save/load checkpoint`
   - `native_input_frames`
2. Keep Hugging Face models behind the existing adapter.
3. Add PyTorchVideo adapters for SlowFast and X3D.
4. Add a 16-sample smoke test for each new model before any full training.
5. Only after smoke tests pass, launch SSV2 subset pilots.

Do not install PyTorchVideo dependencies while long-running jobs are using the same environment. Prepare scripts first, then install/test in a clean window or clone environment.

## Paper Language

Use:

- "evidence budget"
- "model input length"
- "architecture-specific temporal interface"
- "compute-normalized accuracy"
- "temporal evidence allocation"

Avoid:

- "all models receive exactly the same frames" unless this is literally true
- "Nyquist limit" as a direct explanation for recognition accuracy
- "fair by same frame count" without latency or input-shape caveats

## Decision For Current Runs

The current TimeSformer and VideoMAE jobs are still useful as infrastructure and first temporal-evidence pilots.

However, they should not be presented as the final model comparison until:

1. non-transformer baselines are added,
2. fixed-budget tables include `model_input_frames`,
3. results are grouped by architecture family,
4. compute or latency is reported next to accuracy.
