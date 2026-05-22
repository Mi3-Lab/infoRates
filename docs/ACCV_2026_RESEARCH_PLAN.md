# ACCV 2026 Research Plan

## Working Title

**How Much Time Does an Action Model Need? Measuring and Adapting Temporal Evidence in Video Recognition**

Alternative shorter title:

**Temporal Evidence Allocation in Video Action Recognition**

## Core Position

The ECCV version was perceived as a useful empirical benchmark, but not as a sufficiently new research contribution. For ACCV, the work should remain grounded in benchmarking, but the benchmark cannot be the whole story.

The new paper should ask a sharper question:

> Can video action models estimate how much temporal evidence they need before spending computation?

This reframes the project from "temporal sampling affects accuracy" to "current action models are poorly calibrated under temporal evidence budgets, and we can measure and improve this behavior."

## Current Repository Audit

### Assets Already Available

- Main ECCV submission PDF:
  - `Mi3_Lab_Research_InfoRates.pdf`
- Existing temporal sampling results:
  - `evaluations/ucf101/*/*_temporal_sampling.csv`
  - `evaluations/kinetics400/*/*_temporal_sampling.csv`
- Per-class temporal sensitivity results:
  - `evaluations/ucf101/*/*_per_class_testset.csv`
  - `evaluations/kinetics400/*/*_per_class.csv`
  - `evaluations/ucf101_per_class_sensitivity.csv`
- Existing plots:
  - `evaluations/comparative/`
  - `docs/images/`
  - `images/`
- Existing analysis documentation:
  - `docs/COMPREHENSIVE_RESULTS_ANALYSIS.md`
  - `docs/SPECTRAL_ANALYSIS.md`
  - `docs/TEMPORAL_ROBUSTNESS_AUGMENTATION.md`
  - `docs/baseline_tra_comparison_data.md`
  - `docs/DATA_ANOMALY_ANALYSIS.md`
- Existing training and evaluation code:
  - `scripts/data_processing/train_multimodel.py`
  - `scripts/evaluation/run_eval_multimodel.py`
  - `scripts/train_something.py`
  - `scripts/train_with_tra.py`
  - `src/info_rates/analysis/evaluate.py`
  - `src/info_rates/analysis/spectral_analysis.py`
  - `src/info_rates/training/temporal_augmentation.py`
  - `src/info_rates/data/something.py`

### Datasets Present Locally

- UCF101 is supported and has completed results.
- Kinetics-400 is supported, but the current paper used a Kaggle K4TestSet-style test protocol. This should not be a central claim in the ACCV version unless the protocol is made canonical and reproducible.
- HMDB51 data appears present locally and can be used as a small additional dataset if needed.
- Something-Something V2 labels are present, and `scripts/train_something.py` exists. However, the local videos appear incomplete: there are about 18.6k video files against about 168k train annotations. This is useful for prototyping, but ACCV experiments need either a complete dataset or a clearly defined subset protocol.

### Current Technical Issues To Fix Before New Experiments

1. **TRA result integrity**
   - `docs/baseline_tra_comparison_data.md` and `docs/DATA_ANOMALY_ANALYSIS.md` already document suspicious duplicated values and post-fix degradation.
   - TRA must be demoted from main contribution to baseline unless revalidated cleanly.

2. **Evaluation code consistency**
   - There are multiple evaluation paths:
     - `src/info_rates/analysis/evaluate.py`
     - `src/info_rates/analysis/evaluate_fixed.py`
     - `scripts/evaluation/run_eval_multimodel.py`
   - The ACCV work needs one trusted evaluator with a tested sampling function.

3. **Sampling semantics**
   - The old framing mixes coverage, stride, frame count, and temporal resolution.
   - ACCV should define all sampling variables precisely:
     - temporal coverage: duration/window fraction observed
     - sampling stride: spacing between candidate frames
     - inference budget: number of frames actually processed
     - effective FPS: source FPS divided by stride, when FPS is known

4. **Latency and compute metrics**
   - Current result summaries show `avg_time = 0.0000s`, so efficiency claims are not reliable.
   - ACCV needs measured latency, decoded-frame count, processed-frame count, and ideally FLOPs or approximate model compute.

5. **Nyquist language**
   - The ECCV paper overstates Nyquist-Shannon as a direct theory of action recognition.
   - ACCV should use softer language:
     - "frequency-inspired"
     - "temporal evidence rate"
     - "motion-bandwidth proxy"
     - "consistent with undersampling effects"

## New Scientific Thesis

Uniform temporal sampling is a poor interface for video recognition because it assigns the same temporal budget to every video regardless of temporal demand. Some actions are recognizable from sparse context; others require dense temporal evidence. Existing models and sampling recipes do not explicitly measure this demand.

The ACCV paper should introduce:

1. A benchmark protocol for temporal evidence allocation.
2. Metrics that summarize the full accuracy-budget curve.
3. A simple but strong adaptive sampling baseline that reallocates temporal budget per video.
4. A broader evaluation on datasets that require true temporal reasoning.

## Proposed Contributions

### Contribution 1: Temporal Evidence Benchmark

Extend the current coverage-by-stride stress test into a benchmark that measures the full accuracy-budget behavior of action models.

This should include:

- fixed uniform sampling baselines
- coverage and stride sweeps
- budgeted frame-count sweeps
- per-class sensitivity
- per-video temporal demand estimates
- accuracy-compute trade-off curves

The benchmark remains part of the paper, but it becomes the measurement instrument, not the whole contribution.

### Contribution 2: Temporal Robustness Metrics

Add reusable metrics that reviewers can understand quickly:

- **Temporal Robustness AUC**
  - Area under the accuracy-vs-budget curve.
  - Higher means the model remains accurate as temporal budget shrinks.

- **Critical Frame Budget**
  - Minimum frame budget required to retain a target fraction of dense accuracy, e.g. 95%.

- **Temporal Sensitivity Score**
  - Accuracy drop between dense and sparse budgets, computed per class or per video.

- **Budgeted Accuracy**
  - Accuracy at fixed average frame budgets, e.g. 4, 8, 16, 32 frames.

- **Temporal Calibration Error**
  - Difference between predicted temporal demand and observed critical budget.

### Contribution 3: Temporal Information Rate Estimator

Introduce a lightweight estimator of temporal demand.

Possible signals:

- frame-difference energy
- optical-flow magnitude statistics
- feature velocity from a frozen encoder
- token variance across time
- pose/keypoint velocity, if pose extraction is feasible

Recommended first implementation:

1. Start with frame-difference energy because it is cheap and easy to reproduce.
2. Add optical-flow metrics as a stronger variant using the existing spectral analysis module.
3. Add feature velocity only if time permits.

This estimator should not be framed as a perfect physical frequency estimator. It is a practical temporal-demand proxy.

### Contribution 4: Adaptive Temporal Evidence Allocation

Build a simple adaptive sampler that uses the temporal-demand score to choose the number of frames or stride per video.

Initial baselines:

- fixed 8 frames
- fixed 16 frames
- fixed 32 frames
- random budget with same average compute
- confidence-based adaptive budget
- TRA-style temporal augmentation

Proposed method:

- Compute temporal-demand score from a cheap preview.
- Allocate larger frame budget to high-demand videos.
- Allocate smaller frame budget to low-demand videos.
- Keep average compute fixed.

The method should be intentionally simple. The point is to prove that the benchmark exposes a real, exploitable failure mode.

### Contribution 5: Dense-to-Sparse Temporal Consistency

If time permits, add a training objective where a dense teacher guides sparse/adaptive views.

This can be implemented as:

- KL divergence between dense-view logits and sparse-view logits
- contrastive consistency between dense and sparse features
- class-balanced consistency weighted more heavily for temporally sensitive classes

This is the optional "strong method" component. If it works, it becomes the main method. If not, it can stay as an ablation or future work.

## Experimental Scope

### Required Datasets

1. **Something-Something V2**
   - Primary dataset because it reduces the reviewer concern that UCF101 and Kinetics are background-biased.
   - If full data is not immediately available, define a reproducible subset protocol.

2. **Diving48 or FineGym**
   - Use at least one if feasible.
   - These datasets strengthen the claim that temporal evidence matters beyond static scene context.

3. **UCF101**
   - Keep as continuity with the ECCV results.
   - Use it as a sanity-check and historical benchmark, not as the primary evidence.

4. **Kinetics-400**
   - Use only with a clearly reproducible validation protocol.
   - Avoid making K4TestSet/Kaggle central.

5. **HMDB51**
   - Optional small dataset for breadth.

### Required Model Families

The ECCV reviewers criticized the transformer-only scope. ACCV should include at least one non-transformer model.

Required:

- TimeSformer or VideoMAE
- SlowFast
- I3D or TSM

Nice to have:

- VideoSwin or MViT
- X3D as a lightweight efficient model

The existing `ModelFactory` already lists `slowfast` and `x3d`, but those entries need verification because Hugging Face support for those exact model IDs may not work out of the box. If they fail, use PyTorchVideo or MMAction2 rather than forcing them through the current Hugging Face loader.

### Architecture and Sampling Protocol

The detailed protocol is now in `docs/ACCV_2026_ARCHITECTURE_AND_SAMPLING_PROTOCOL.md`.

Key decision:

- Compare **evidence budget** separately from **model input length**.
- Report both `processed_frames` and `model_input_frames`.
- Use architecture-valid grids:
  - TimeSformer: `2, 4, 8`
  - VideoMAE: `4, 8, 16`
  - ViViT: `4, 8, 16, 32`
  - SlowFast/X3D: checkpoint-native grids after adapter smoke tests
- Add at least SlowFast and X3D to avoid a transformer-only ACCV submission.

ViViT remains part of the transformer set. It should not replace non-transformer baselines, but it is valuable because its 32-frame native input tests whether a larger temporal input interface changes the temporal evidence curve.

## Repository Organization Plan

Do not rewrite the whole repository before the science is clear. Instead, create a clean ACCV track alongside the existing ECCV artifacts.

Recommended structure:

```text
docs/
  ACCV_2026_RESEARCH_PLAN.md
  ACCV_2026_EXPERIMENT_TRACKER.md
  ACCV_2026_PAPER_OUTLINE.md

src/info_rates/
  metrics/
    temporal_robustness.py
    temporal_demand.py
  sampling/
    adaptive.py
    temporal.py
  evaluation/
    benchmark.py

scripts/accv2026/
  01_build_manifests.py
  02_run_fixed_budget_eval.py
  03_run_adaptive_eval.py
  04_compute_temporal_metrics.py
  05_make_paper_tables.py

evaluations/accv2026/
  ucf101/
  somethingv2/
  diving48_or_finegym/
  hmdb51/
```

Do not move existing result files yet. New work should write to `evaluations/accv2026/` so old ECCV artifacts remain intact and reproducible.

## Step-By-Step Execution Plan

### Phase 0: Freeze The ECCV Version

Goal: preserve the previous state while preventing old claims from contaminating the ACCV version.

Tasks:

- Keep `Mi3_Lab_Research_InfoRates.pdf` as historical reference.
- Mark TRA results as preliminary/untrusted unless re-run.
- Do not reuse the old Nyquist-heavy language.
- Document which CSVs are considered trusted.

Deliverable:

- `docs/ACCV_2026_EXPERIMENT_TRACKER.md` with trusted/untrusted status for every major result file.

Success criteria:

- We can tell which results are safe to cite and which are only exploratory.

### Phase 1: Build The New Metrics

Goal: transform the benchmark from a grid of accuracies into a temporal robustness measurement framework.

Tasks:

- Implement Temporal Robustness AUC.
- Implement Critical Frame Budget.
- Implement Temporal Sensitivity Score.
- Implement Budgeted Accuracy.
- Implement plots for accuracy-budget curves.
- Validate metrics on existing UCF101 and Kinetics CSVs.

Deliverable:

- `src/info_rates/metrics/temporal_robustness.py`
- `scripts/accv2026/04_compute_temporal_metrics.py`
- metrics tables under `evaluations/accv2026/`

Success criteria:

- Existing UCF101/Kinetics results can be converted into ACCV-style metrics without rerunning models.

### Phase 2: Build A Trusted Evaluator

Goal: remove ambiguity from the sampling and evaluation protocol.

Tasks:

- Select one evaluation entry point.
- Write tests for frame selection under coverage, stride, and fixed budget.
- Ensure labels are correctly aligned with video paths.
- Record decoded frame count, processed frame count, latency, and accuracy.
- Add checkpoint/resume support.

Deliverable:

- `src/info_rates/evaluation/benchmark.py`
- `scripts/accv2026/02_run_fixed_budget_eval.py`
- tests for sampling and label alignment

Success criteria:

- A small smoke test produces deterministic frame indices and correct labels.
- Latency is nonzero and measured consistently.

### Phase 3: Dataset Readiness

Goal: prepare temporally meaningful datasets before method work.

Tasks:

- Audit Something-Something V2 completeness.
- If incomplete, either finish acquisition or define a reproducible subset:
  - class-balanced
  - enough samples per class
  - train/val split preserved
  - missing/corrupt videos logged
- Decide whether to use Diving48 or FineGym as the second temporal dataset.
- Build manifests with columns:
  - `video_path`
  - `label`
  - `label_id`
  - `dataset`
  - `split`
  - `fps`
  - `num_frames`
  - `duration`

Deliverable:

- `evaluations/accv2026/manifests/`
- dataset audit report

Success criteria:

- At least one temporal-reasoning dataset is ready for full evaluation.

### Phase 4: Baseline Model Matrix

Goal: address the transformer-only criticism and establish credible baselines.

Minimum model matrix:

| Model | Family | Role |
|---|---|---|
| TimeSformer or VideoMAE | Transformer | continuity with ECCV |
| SlowFast | 3D CNN / two-stream temporal | reviewer-requested baseline |
| I3D or TSM | CNN temporal baseline | reviewer-requested baseline |

Tasks:

- Verify model loading and preprocessing.
- Run dense baseline.
- Run fixed-budget sweeps.
- Run coverage-stride sweeps only where useful.
- Save logits if storage permits; this enables later calibration and distillation analysis.

Deliverable:

- fixed-budget and coverage-stride CSVs under `evaluations/accv2026/{dataset}/{model}/`

Success criteria:

- Each primary dataset has at least one transformer and one non-transformer model evaluated.

### Phase 5: Temporal Demand Estimator

Goal: predict how much temporal evidence a video needs before expensive inference.

Tasks:

- Implement frame-difference temporal-demand score.
- Implement optical-flow temporal-demand score using existing spectral code.
- Correlate each score with observed critical budget.
- Evaluate per-video and per-class correlations.
- Compare against random and confidence-only baselines.

Deliverable:

- `src/info_rates/metrics/temporal_demand.py`
- temporal-demand CSVs and scatter plots

Success criteria:

- Temporal-demand score correlates meaningfully with observed critical budget on at least one temporal dataset.

### Phase 6: Adaptive Sampling Baseline

Goal: show that temporal demand is actionable.

Tasks:

- Implement a rule-based adaptive sampler:
  - low temporal demand: small frame budget
  - medium temporal demand: medium frame budget
  - high temporal demand: large frame budget
- Match average compute against fixed-budget baselines.
- Evaluate accuracy at equal average frame count.
- Evaluate robustness on temporally sensitive classes.

Deliverable:

- `src/info_rates/sampling/adaptive.py`
- `scripts/accv2026/03_run_adaptive_eval.py`
- adaptive-vs-fixed comparison tables

Success criteria:

- Adaptive sampling improves accuracy at the same average frame budget, or reduces frame budget at the same accuracy.

### Phase 7: Optional Dense-to-Sparse Consistency

Goal: add a stronger learning contribution if time allows.

Tasks:

- Use dense-view predictions as teacher targets.
- Train sparse/adaptive student views with KL consistency.
- Compare:
  - standard fine-tuning
  - TRA
  - dense-to-sparse consistency
  - adaptive sampling with and without consistency

Deliverable:

- training script extension or new ACCV script
- ablation table

Success criteria:

- Consistency improves temporal robustness AUC or critical budget without harming dense accuracy.

### Phase 8: Paper Writing

Goal: write the ACCV paper around the new question, not around ECCV review fixes.

Recommended paper outline:

1. Introduction
   - Ask: how much temporal evidence does an action model need?
   - Explain why fixed sampling is wasteful and sometimes unsafe.

2. Related Work
   - Video action recognition
   - Efficient video inference
   - Adaptive frame selection
   - Robustness and temporal reasoning

3. Temporal Evidence Benchmark
   - Define budgets, coverage, stride, metrics.

4. Temporal Demand Estimation
   - Define motion/feature-based estimators.

5. Adaptive Temporal Evidence Allocation
   - Simple controller and optional consistency training.

6. Experiments
   - Datasets, models, protocols.
   - Main accuracy-budget results.
   - Per-class/per-video analysis.
   - Adaptive sampling comparisons.
   - Ablations.

7. Limitations
   - Temporal demand proxies are imperfect.
   - Sensor FPS and compression artifacts matter.
   - Adaptive sampling does not reconstruct missing information.

8. Conclusion
   - Temporal evidence should be allocated, not fixed.

## Claims To Avoid

Avoid:

- "We prove Nyquist-Shannon applies to action recognition."
- "TRA mitigates aliasing."
- "Temporal augmentation recovers lost information."
- "UCF101/Kinetics alone establish temporal reasoning."
- "Kaggle Kinetics test protocol is canonical."

Use instead:

- "frequency-inspired temporal demand"
- "undersampling-consistent failure modes"
- "temporal evidence allocation"
- "adaptive budget under equal compute"
- "dense views can regularize sparse views, but cannot recover absent evidence"

## Minimum Viable ACCV Paper

If time is tight, the minimum viable version is:

1. Trusted evaluator.
2. New robustness metrics.
3. Something-Something V2 subset or full evaluation.
4. One transformer and one non-transformer baseline.
5. Temporal-demand score.
6. Rule-based adaptive sampler.
7. UCF101 as supporting evidence.

This version is already much stronger than the ECCV version because it contributes a question, metrics, and an actionable baseline.

## Strong ACCV Paper

The strong version adds:

1. Full Something-Something V2 or a large, well-defined subset.
2. Diving48 or FineGym.
3. SlowFast plus I3D/TSM.
4. Dense-to-sparse consistency training.
5. Calibration analysis showing models do not know when they need more frames.
6. Equal-compute comparisons against adaptive inference baselines.

## Immediate Next Actions

1. Create `docs/ACCV_2026_EXPERIMENT_TRACKER.md`.
2. Implement temporal robustness metrics on existing CSVs.
3. Audit Something-Something V2 completeness and decide full vs subset.
4. Consolidate evaluation into one trusted ACCV evaluator.
5. Run a small fixed-budget pilot on Something-Something V2 with one model.
6. Implement the first temporal-demand estimator.
7. Compare fixed budget vs adaptive budget on the pilot.

## Decision Gates

### Gate 1: Dataset Gate

Proceed only if at least one temporal-reasoning dataset is usable.

Pass condition:

- Something-Something V2 full or reproducible subset is ready, or Diving48/FineGym is ready.

### Gate 2: Metric Gate

Proceed only if the new metrics reveal a clear difference between models or classes.

Pass condition:

- Temporal Robustness AUC and Critical Frame Budget produce interpretable rankings.

### Gate 3: Estimator Gate

Proceed only if temporal-demand scores correlate with observed critical budget.

Pass condition:

- Positive correlation on at least one temporal dataset and sensible qualitative examples.

### Gate 4: Method Gate

Proceed with method claims only if adaptive sampling wins at equal compute.

Pass condition:

- Adaptive budget improves accuracy at fixed average frame count or reduces frame count at fixed accuracy.

If Gate 4 fails, the paper can still be submitted as a stronger benchmark-and-analysis paper, but the title and claims should emphasize measurement rather than method.
