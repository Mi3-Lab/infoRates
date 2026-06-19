# ACCV 2026 / WACV 2027 Research Plan

**Last updated:** 2026-05-24

---

## Title

**ACCV 2026 submission:**
> *Temporal Demand-Aware Video Recognition: Estimation, Measurement, and Adaptive Frame Allocation*

**WACV 2027 extended version:**
> *Learning to Budget Temporal Evidence: Adaptive Frame Allocation for Efficient and Accurate Video Recognition*

---

## Why the ECCV Version Failed — and What We Fix

All three ECCV 2026 reviewers raised the same structural objections:

| Reviewer | Criticism | Fix |
|---|---|---|
| All | Transformer-only model set | Add 3D CNNs (R3D-18, MC3-18, R(2+1)D-18) + SlowFast R50 → 7 models across 3 families |
| All | UCF101/Kinetics are background-conditioned | SSV2 as primary dataset + Diving48 as secondary — both require temporal reasoning |
| Multiple | TRA is just data augmentation, not a method | TRA becomes baseline; **adaptive allocation becomes the method** |
| Multiple | Nyquist claim is overstated | Drop Nyquist language entirely; use "temporal demand" throughout |
| Multiple | Findings are expected / not novel | New: cross-family architecture analysis + temporal demand estimator + adaptive method |

The ECCV paper measured temporal sensitivity. The ACCV/WACV paper **exploits** it: if we can estimate a video's temporal demand before expensive inference, we can allocate frames adaptively — giving more frames to hard videos and fewer to easy ones — and achieve higher accuracy at the same or lower average compute.

**This is the SOTA contribution: a method that beats fixed-budget uniform sampling in accuracy-per-frame efficiency.**

---

## Core Thesis

> Temporal evidence demand varies dramatically across video actions, yet every deployed video model assigns the same frame budget to every input. We show this is inefficient and demonstrate that a lightweight temporal demand estimator enables adaptive frame allocation that matches or exceeds uniform-sampling accuracy with systematically fewer frames on low-demand videos.

Three verifiable claims form the paper:

1. **Demand varies**: Fixed-budget accuracy curves differ sharply across action categories and across architecture families — some models/actions need 32 frames, others converge at 4.
2. **Demand is predictable**: Frame-difference energy computed from a cheap 4-frame preview correlates with the observed critical frame budget.
3. **Prediction is actionable**: An adaptive sampler routing high-demand videos to large budgets and low-demand videos to small budgets achieves higher accuracy than any fixed budget at the same average frame count.

---

## Infrastructure Already Built

Everything below exists and has been committed:

| Component | Script / Module | Status |
|---|---|---|
| SSV2 data loader | `src/info_rates/data/something.py` | Ready |
| Fixed-budget evaluator | `scripts/accv2026/02_run_fixed_budget_eval.py` | Ready |
| Temporal metrics | `scripts/accv2026/05_compute_temporal_metrics.py` | Ready |
| Manifest builder | `scripts/accv2026/01_build_manifests.py` | Ready |
| 3D CNN models | `src/info_rates/models/torchvision_video.py` | Ready |
| SlowFast model | `src/info_rates/models/slowfast_video.py` | Ready |
| Training (torchvision) | `scripts/accv2026/train_torchvision.py` | Ready (bad-video filter + per-epoch checkpoints + resume) |
| Training (SlowFast) | `scripts/accv2026/train_slowfast.py` | Ready (DDP _set_static_graph fix) |
| Training (transformers) | `scripts/accv2026/train_timesformer.py` | Ready |
| A100 pilot launcher | `scripts/accv2026/run_a100_ssv2_slowfast_pilot.sh` | Ready |
| W&B live logging | All scripts use `WANDB_MODE=online` | Ready |

**Models being trained right now (or completed):**
- R3D-18, MC3-18, R(2+1)D-18 on SSV2 (full, 5 epochs)
- SlowFast R50 — pilot done (4.2% on 5k, expected; needs full training)
- TimeSformer, VideoMAE, ViViT — pilots done; full training queued

---

## Three Contributions (Paper Structure)

### Contribution 1 — Temporal Evidence Benchmark

A reproducible, cross-architecture, cross-family benchmarking protocol for temporal evidence demand.

**What we measure:**

- Fixed-budget accuracy at k ∈ {4, 8, 16, 32} frames per video
- Accuracy-budget curve (7 models × 2 datasets)
- **Temporal AUC**: area under accuracy-vs-budget curve (higher = temporally efficient)
- **Critical Budget**: minimum k retaining ≥ 95% of dense accuracy
- **Per-class Temporal Sensitivity**: variance in accuracy across budgets, per action class

**Model matrix (7 models, 3 families):**

| Model | Family | Frames |
|---|---|---|
| TimeSformer-Base | Transformer | 8 |
| VideoMAE-Base | Transformer | 16 |
| ViViT-B/16 | Transformer | 32 |
| R3D-18 | 3D CNN | 16 |
| MC3-18 | 3D CNN (mixed conv) | 16 |
| R(2+1)D-18 | 3D CNN (factored) | 16 |
| SlowFast R50 | Two-stream | 32 slow / 64 fast |

**Datasets:**

| Dataset | Why |
|---|---|
| Something-Something V2 | 174 classes, 168k train, requires genuine temporal order |
| Diving48 | 48 fine-grained dive phases, pure temporal discrimination, explicitly recommended by reviewer gj2t |

UCF101 kept as supporting appendix for continuity with prior work, not as primary claim.

**Key findings we expect to show:**

- Architecture family predicts temporal demand more reliably than dataset alone
- Transformers with patch-based attention have lower critical budget than 3D CNNs on identical inputs
- SlowFast's two-stream design naturally handles temporal demand differently from uniform 3D CNNs
- Per-class: fine-grained phase actions (Diving48 phases, SSV2 "moving left vs right") need dense budgets; static-appearance actions do not

---

### Contribution 2 — Temporal Demand Estimation

A lightweight, training-free estimator that predicts a video's critical frame budget before inference.

**Why this is novel:** No prior video recognition paper proposes a pre-inference proxy for temporal demand that is then used to route compute. Existing efficient video methods (AdaFrame, AR-Net, LiteEval) either require training a policy network or assume the full video is available — neither is a cheap pre-inspection estimator.

**The estimator (implementation priority order):**

**Tier 1 — Frame-Difference Energy (cheap, interpretable, start here):**

```
FDE(v) = (1 / T-1) * Σ_{t=1}^{T-1} || f_{t+1} - f_t ||_2^2 / (H * W * C)
```

Computed on 4 downsampled frames decoded at low resolution (64×64). Cheap enough to run in the DataLoader worker. Gives a scalar per video that we correlate with the observed critical budget.

**Tier 2 — Temporal Variance Score (feature-space version):**

Extract features from a frozen lightweight encoder (MobileNetV3 or ResNet-18) at 4 keyframes. Compute variance across feature vectors. Normalized to [0, 1]. Better than raw pixel diff because it is illumination-invariant.

**Tier 3 — Optical Flow Magnitude (strongest proxy, already partially coded in spectral module):**

Sum of flow magnitudes at sparse temporal samples. Correlates with perceived motion complexity. Use the existing `src/info_rates/analysis/spectral_analysis.py` infrastructure.

**Validation protocol:**

1. Run fixed-budget eval (k=4,8,16,32) for all models on SSV2 val set.
2. For each video, define `critical_budget(v)` = smallest k where model accuracy ≥ 0.95 × dense accuracy.
3. Compute Spearman correlation between estimator score and `critical_budget(v)`.
4. Show scatter plots and per-class breakdowns.

**Success threshold:** Spearman ρ > 0.35 on SSV2 or Diving48. Even a weak but consistent correlation justifies routing compute.

---

### Contribution 3 — Adaptive Temporal Allocation

A test-time method that uses the demand estimator to select per-video frame budgets, matching or exceeding fixed-budget accuracy at lower average frame count.

**The method (no retraining required — inference-time only):**

```
score = FDE(video, 4 frames at 64x64)          # ~5ms on CPU, runs in dataloader

if score < θ_low:    budget = 4 frames
elif score < θ_high: budget = 16 frames
else:                budget = 32 frames
```

Thresholds θ_low, θ_high calibrated on the SSV2 val set to match the average frame count of the 16-frame fixed baseline.

**Evaluation: equal-compute comparison**

The only fair comparison for an adaptive method: fix the average frame count and compare accuracy.

| Method | Avg frames | SSV2 Top-1 | Diving48 Top-1 |
|---|---|---|---|
| Fixed 4 | 4.0 | baseline | baseline |
| Fixed 16 | 16.0 | baseline | baseline |
| Fixed 32 | 32.0 | baseline | baseline |
| Adaptive (ours) | ~16.0 | **target: ≥ Fixed 16** | **target: ≥ Fixed 16** |

The adaptive method wins if it matches Fixed-16 accuracy while actually using fewer frames on easy videos (paid for by fewer frames on hard videos too), OR if it beats Fixed-16 at the same average cost.

**Ablation table (required):**

| Variant | Estimator | Budget routing | Avg frames |
|---|---|---|---|
| Oracle | Ground-truth critical budget | Optimal routing | Variable |
| Adaptive-FDE | Frame-diff energy | Threshold | ~16 |
| Adaptive-random | Random | Same distribution | ~16 |
| Fixed-4 | None | Always 4 | 4 |
| Fixed-16 | None | Always 16 | 16 |
| Fixed-32 | None | Always 32 | 32 |

Oracle upper-bounds the method. Random routing shows that just budgeting helps, but accurate demand estimation helps more.

---

## Differences from ECCV That Reviewers Will Notice

| Dimension | ECCV version | ACCV/WACV version |
|---|---|---|
| Models | Transformers only | 7 models, 3 families |
| Datasets | UCF101, Kinetics | SSV2 (primary), Diving48 |
| Contribution type | Benchmark + TRA | Benchmark + estimator + **adaptive method** |
| Nyquist language | Heavy | Dropped entirely |
| TRA | Main contribution | Baseline to beat |
| New claim | Temporal sensitivity varies | Demand is predictable and allocation is actionable |
| Compute control | None | Equal-average-frame comparisons throughout |

---

## ACCV 2026 — Target Paper (Minimum and Strong Versions)

**Estimated ACCV 2026 deadline: ~July 14, 2026** (check camera-ready schedule)

### Minimum ACCV Paper (ship by deadline even if resources are tight)

Requires completing training on at least:
- R3D-18 + R(2+1)D-18 (full SSV2, in progress)
- TimeSformer or VideoMAE (full SSV2)
- SlowFast R50 pilot → upgrade to full if possible

Minimum experiments:
1. Fixed-budget eval on SSV2 val with 3+ models
2. Temporal AUC and Critical Budget tables
3. FDE estimator correlation on SSV2
4. Adaptive allocation vs Fixed-16 on SSV2

This alone is a substantially stronger paper than ECCV: new datasets, non-transformer models, a real method with a quantitative win.

### Strong ACCV Paper

Adds:
1. All 7 models fully trained and evaluated on SSV2
2. Diving48 evaluation (requires dataset acquisition — see below)
3. Temporal Variance Score (Tier 2 estimator) as ablation of FDE
4. Per-class temporal sensitivity analysis with visualizations
5. Comparison against AdaFrame / AR-Net / LiteEval-style baselines from literature (no retraining needed — compare our adaptive protocol against their published numbers where possible)

---

## WACV 2027 — SOTA Extension

**Estimated WACV 2027 deadline: ~September 2026**

WACV 2027 is the venue if ACCV 2026 is rejected, or if we decide to produce a significantly stronger paper with 6 weeks more compute time.

The WACV version adds one critical component that ACCV lacks time for:

### Learned Temporal Demand Estimator

Replace the heuristic FDE threshold routing with a small learned network:

- Input: 4 frames at 64×64 from early in the video
- Architecture: lightweight 2D CNN (ResNet-18 pretrained, 4 frames stacked as 12-channel input) → scalar score
- Training signal: pseudo-labels from the critical budget observed during fixed-budget eval
- Objective: predict critical_budget(v) as a regression target
- Total parameters: ~11M, trained in <1 hour on 1 GPU

This converts the adaptive method from a calibrated heuristic into a **trainable, end-to-end compatible module**.

### WACV Contribution Structure

1. **Benchmark** (identical to ACCV, already published/submitted)
2. **Analysis** (strengthened with Diving48 + full per-class breakdown)
3. **Adaptive method** (identical to ACCV but now validated with learned estimator)
4. **Learned estimator ablation**: FDE vs Temporal Variance vs Learned — show learned estimator closes the gap with Oracle routing

This gives WACV a clear "we trained something new" story without requiring retraining all 7 backbone models.

---

## Dataset Acquisition: Diving48

Diving48 is explicitly requested by reviewer gj2t and is a small dataset (~18k videos). Acquisition steps:

1. Download from official source (Academic Torrents or Charades hosting — check current availability)
2. Add to `src/info_rates/data/` as `diving48.py` following the same interface as `something.py`
3. Add to manifest builder `01_build_manifests.py`
4. Add to fixed-budget evaluator `02_run_fixed_budget_eval.py`

Fine-tuning each model on Diving48 train split (~8k videos, 48 classes) takes <1 hour on 1 A100.

**Priority: after SSV2 full runs are complete.** Do not block ACCV training on Diving48 acquisition.

---

## Execution Timeline

**Today: May 24, 2026**

| Week | Dates | Goal |
|---|---|---|
| 1 | May 24 – May 30 | Complete R(2+1)D-18 full training (job 70582); relaunch SlowFast full; check TimeSformer/VideoMAE status |
| 2 | May 31 – Jun 6 | Complete all 7 SSV2 full trainings; run fixed-budget eval on SSV2 val for all models |
| 3 | Jun 7 – Jun 13 | Compute Temporal AUC, Critical Budget, per-class sensitivity; implement FDE estimator |
| 4 | Jun 14 – Jun 20 | Implement adaptive allocation; run equal-compute comparison; acquire Diving48 |
| 5 | Jun 21 – Jun 27 | Fine-tune on Diving48; run fixed-budget + adaptive on Diving48; draft Results section |
| 6 | Jun 28 – Jul 5 | Draft full paper; generate all figures and tables |
| 7 | Jul 6 – Jul 13 | Polish, proofread, submit to ACCV 2026 |
| **Fallback** | Aug – Sep 2026 | If ACCV rejected: add learned estimator, resubmit to WACV 2027 |

---

## Immediate Next Actions (Week 1)

1. **Monitor job 70582** (R(2+1)D-18 full SSV2) — check accuracy progression via W&B
2. **Launch SlowFast full training** on 2×A100 with the pilot script upgraded to full SSV2 (remove `--max-train-samples`)
3. **Launch TimeSformer full training** if not already running (check W&B)
4. **Implement FDE estimator** in `src/info_rates/metrics/temporal_demand.py`
5. **Acquire Diving48** — check dataset availability and download pipeline

---

## Code To Write (Implementation Priority Order)

### Priority 1 — FDE Estimator (required for ACCV)

File: `src/info_rates/metrics/temporal_demand.py`

```python
def frame_diff_energy(video_path: str, num_probe_frames: int = 4, size: int = 64) -> float:
    """Compute normalized frame-difference energy as temporal demand proxy."""
    # decode num_probe_frames evenly spaced frames at low resolution
    # return mean squared pixel difference normalized to [0, 1]
```

Outputs a scalar per video. Fast enough to precompute for the entire SSV2 val set in <10 minutes on CPU.

### Priority 2 — Adaptive Sampler (required for ACCV)

File: `src/info_rates/sampling/adaptive.py`

```python
class AdaptiveTemporalSampler:
    def __init__(self, estimator, thresholds, budgets):
        ...
    def select_budget(self, video_path: str) -> int:
        score = self.estimator(video_path)
        return budgets[bisect(thresholds, score)]
```

### Priority 3 — Equal-Compute Eval Script (required for ACCV)

File: `scripts/accv2026/03_run_adaptive_eval.py`

Takes a model checkpoint + sampler config, evaluates at adaptive budgets, reports accuracy and average frame count. Must be directly comparable to output of `02_run_fixed_budget_eval.py`.

### Priority 4 — Diving48 Data Module (required for strong ACCV / WACV)

File: `src/info_rates/data/diving48.py`

Same interface as `something.py`. Manifests go in `evaluations/accv2026/manifests/`.

### Priority 5 — Learned Estimator (WACV only, do not block ACCV)

File: `src/info_rates/metrics/learned_demand_estimator.py`

Trained as regression from pseudo-labeled critical budgets. Requires completed fixed-budget eval as prerequisite.

---

## Decision Gates (Unchanged from Prior Plan)

| Gate | Pass Condition | Action if Fail |
|---|---|---|
| Model Gate | ≥4 of 7 models trained on SSV2 | Submit ACCV with available models, rerun all for WACV |
| Estimator Gate | Spearman ρ > 0.35 between FDE and critical_budget | Use Tier-2/Tier-3 estimator; do not claim Tier-1 works if it doesn't |
| Method Gate | Adaptive ≥ Fixed-16 accuracy at same average frames | Submit as analysis paper if method fails; still publishable |
| Diving48 Gate | Dataset acquired and ≥1 model fine-tuned | Submit without Diving48 for ACCV; add for WACV |

---

## Claims To Make

- "Architecture family is a stronger predictor of temporal demand than dataset category alone."
- "Frame-difference energy computed from 4 low-resolution probe frames predicts critical budget with Spearman ρ = X on SSV2."
- "Adaptive temporal allocation achieves [Y]% accuracy with [Z] average frames, compared to [Y-ε]% with [Z'] frames for fixed-budget sampling."
- "Temporal AUC reveals that SlowFast and R(2+1)D-18 are more temporally efficient than transformers at equal average frame count on SSV2."

## Claims To Avoid

- "Nyquist-Shannon applies to action recognition." (never again)
- "TRA mitigates aliasing." (TRA is now just a baseline)
- "UCF101 or Kinetics establish temporal reasoning." (only SSV2 and Diving48 are primary)
- "Our method requires retraining the backbone." (the whole point is it is inference-time only)
- "Our estimator perfectly predicts demand." (claim correlation, not perfection)

---

## Paper Outline

1. **Introduction** — The mismatch between uniform sampling and unequal temporal demand; why this is inefficient and fixable.
2. **Related Work** — Video recognition; efficient video inference (AdaFrame, AR-Net, LiteEval, FrameExit); temporal reasoning datasets; adaptive computation.
3. **Temporal Evidence Benchmark** — Protocol, models, datasets, metrics (Temporal AUC, Critical Budget, Temporal Sensitivity).
4. **Temporal Demand Estimation** — FDE and Tier-2 estimators; correlation analysis; per-class qualitative examples.
5. **Adaptive Temporal Allocation** — Method; equal-compute evaluation; ablation table (oracle / adaptive-FDE / adaptive-random / fixed).
6. **Experiments** — SSV2 main results; Diving48 results; per-class breakdown; architecture family comparison.
7. **Limitations** — Estimator is a proxy, not ground truth; assumes fixed backbone; temporal demand may shift with viewing angle or cut.
8. **Conclusion** — Temporal evidence should be estimated and allocated, not uniformly fixed.

---

## Repository Structure (Current and Planned)

```
scripts/accv2026/
  01_build_manifests.py           # Done
  02_run_fixed_budget_eval.py     # Done
  03_run_adaptive_eval.py         # TODO: Priority 3
  05_compute_temporal_metrics.py  # Done
  train_torchvision.py            # Done (with all robustness fixes)
  train_slowfast.py               # Done (DDP _set_static_graph fix)
  train_timesformer.py            # Done

src/info_rates/
  data/
    something.py                  # Done
    diving48.py                   # TODO: Priority 4
  models/
    torchvision_video.py          # Done
    slowfast_video.py             # Done
    timesformer.py                # Done
  metrics/
    temporal_demand.py            # TODO: Priority 1
    temporal_robustness.py        # Done (Temporal AUC etc.)
  sampling/
    adaptive.py                   # TODO: Priority 2

evaluations/accv2026/
  manifests/                      # SSV2 val manifests exist
  fixed_budget/                   # Output dir for 02_run_fixed_budget_eval.py
  adaptive/                       # Output dir for 03_run_adaptive_eval.py (TODO)
  metrics/                        # Output dir for 05_compute_temporal_metrics.py
```

---

## What Makes This SOTA

A paper earns "SOTA" in efficient video recognition if it:

1. Defines the right metric — **Temporal AUC** is new and captures accuracy-efficiency trade-off in one number
2. Covers the right architecture families — 3 families × 2 temporal datasets closes the "transformers only" criticism
3. Provides a method that wins — **adaptive allocation beats fixed-16 at equal compute** — this is the headline result
4. The method is practical — no extra training, no privileged access to labels at test time, works on any backbone

The adaptive allocation is lightweight by design: it is intentionally simple so the win cannot be dismissed as "a bigger model." If a 5ms FDE estimator routing frames to existing models beats uniform sampling, the insight generalizes.

This is not a new backbone paper. It is an **inference strategy** paper. The contribution is understanding demand and acting on it. That is publishable at ACCV and competitive at WACV/CVPR if the method gate passes cleanly.
