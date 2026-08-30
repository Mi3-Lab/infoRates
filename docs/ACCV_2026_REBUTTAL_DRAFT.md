# ACCV 2026 — Rebuttal Draft (Submission 230)

**Status:** working draft. Every number below has been reproduced from raw data;
`[PENDING]` marks results still computing. Scores: XdvJ 5 / vdwp 4 / pxtb 1.

**Length:** ACCV caps the rebuttal — confirm the exact limit before finalising.
This draft is deliberately over-length; the priority order for cutting is given
at the end.

---

## Framing decision

R2 and R3 both argue the protocol does not measure aliasing. **They are right**,
and we verified the mechanism in our own code. The rebuttal leads with that
concession, in our own words, before any reviewer has to press it — then shows
the concession is more informative than the original claim.

Do **not** argue the point. Two independent reviewers converged on it, and a
third (XdvJ) asked the question whose honest answer is the same thing.

---

## Opening (shared, ~150 words)

> We thank the reviewers. R2 and R3 question whether our protocol measures
> temporal aliasing. Investigating their objection, we found they are correct,
> and we can now state the mechanism precisely.
>
> `select_frame_indices` builds `candidates = arange(0, window, stride)` and then
> re-uniformizes them with `linspace` to the model's frame budget. At fixed
> coverage the selected frames therefore span the same window at **every**
> stride. Stride only changes the input once `ceil(window/stride) < budget`, and
> the deficit is filled by **repeating the last candidate frame**.
>
> Expressed in physical units this is unmistakable. Median seconds between
> sampled frames, coverage 100%:
>
> | dataset | s=1 | s=2 | s=4 | s=8 | s=16 |
> |---|---|---|---|---|---|
> | UCF-101 | 0.395 | 0.395 | 0.395 | 0.395 | 0.617 |
> | AUTSL | 0.125 | 0.125 | 0.131 | 0.252 | 0.475 |
>
> Four of the five columns are the same sampling rate. The axis we labelled
> "stride" is, over most of its range, not a sampling-rate axis at all.

---

## The corrected account (shared, ~200 words)

Three measurements, all on the submitted data:

1. **The effect lives entirely in the padding regime.** Restricting to clips
   that never pad, the stride 1→16 drop collapses from **+10.7pp to −0.5pp**, in
   **21 of 21** (model, dataset) pairs with a usable unpadded subset. On AUTSL,
   SSv2 and DriveAct *no* clip escapes padding at stride 16.

2. **Padding gates the effect; content sets its size.** Spearman(padding rate,
   TDS) = **+0.964**. But among the three datasets that are 100% saturated, TDS
   still ranges 21.9–53.0pp — a 31pp spread padding cannot explain. Temporal
   demand is real; our instrument for it was not.

3. **The architecture ranking tracked frame budget.** Spearman(input budget,
   published stride drop) = **+0.849**, p=0.008. The two "robust" models are
   exactly the two with `budget=8`.

**What survives.** Re-running with every architecture given the *same k distinct
frames at the same positions*: accuracy at matched evidence still correlates with
the published robustness ordering (Spearman **+0.821**, p=0.023). The ordering is
real; the **3–5× magnitude was the budget artifact**. C2 needs rewriting, not
withdrawal:

> *from* "architecture X degrades 3–5× less under stride"
> *to* "at equal evidence budget, architecture X delivers more accuracy per
> distinct frame"

---

## R3 — vdwp (4, confidence 5, expert). Highest-value target.

Their blocker is the definition of aliasing. Lead with the concession above,
then:

- **Frame rate.** Correct, and worse than stated: the fps table the duration
  analysis uses is hardcoded and wrong for three datasets. Measured from the
  files: HMDB-51 **30** (not 25), AUTSL **30** (not 25), EPIC-Kitchens **50**
  (not 60); FineGym is absent from the dict and silently defaults to 25. At a
  fixed nominal stride the real sampling interval varies **3.9×** across
  datasets, so same-stride comparisons compare different physical conditions. We
  will re-express the axis in seconds throughout.
- **Kinetics-400.** Added, no training required — every architecture is already
  K400-pretrained, and we verified the manifest label order matches each model's
  `id2label`. Preliminary: the curve is nearly flat from k=2 for R3D-18, placing
  K400 at the low-demand end beside UCF-101. This sharpens the paper's point:
  the field's default sampling recipes were tuned on the benchmark where
  sampling matters least. `[PENDING full 8-model run]`
- **Table 5 / FineGym.** Regenerated for all 8 datasets from one pool, FineGym
  included (Low ≤50.4pp, High >71.2pp, 33/32). See the caveat below — this row
  cannot simply be appended.
- **Table 7 fine-tuning may not be significant.** Confirmed. Fine-tuning at the
  *native* resolution, where there is nothing to adapt, already gains **+6 to
  +10.6pp** for 7 of 8 models. The headline +39.2pp is ≈32pp resolution
  adaptation plus ≈7pp additional training. We will report both components.
- **Model ordering across figures.** Will fix.

## R1 — XdvJ (5). Must not slip.

- **How frames / span / interval vary with stride and coverage.** This is the
  question whose answer is the mechanism above; we will state it explicitly in
  Sec. 3.2 with the index table.
- **Multi-stride augmentation ablation.** Running. Two arms, both trained with
  random (coverage, stride) exposure: one using the *published* sampler
  (padding included), one using uniform resampling with no frozen tail. The
  baseline arm is the published checkpoint. Prediction from the mechanism: the
  published cliff is **not** trainable, because no recipe lets a model extract 16
  frames of evidence from 4. `[PENDING]`
- **Convergence at 48/96px.** Extracted from 730 training logs. The concern is
  partly justified: at 48px **52.8%** of runs still peak at the final epoch,
  the highest of any resolution (16–27% elsewhere). But the accuracy still being
  gained in the last three epochs is **1.38pp**, the *smallest* of any
  resolution — the runs were creeping, not climbing. We will report this as a
  stated limitation.
- **Table 7 without-finetuning column.** Added.
- **Sec. 3.3 resize + PE interpolation for zero-shot.** Will move from S8.
- **SSv2 in Table S19.** Will add the discussion contrasting SSv2's discrete
  state-change evidence with continuous-motion domains.

## R2 — pxtb (1). Likely unmovable; answer for the AC's benefit.

- **Routing.** Both objections correct. The rule thresholds `max_c p(c|v)`, so
  "entropy routing" is a misnomer and will be renamed. Selecting τ on a
  calibration half and scoring on the held-out half over 1000 stratified splits,
  the headline SSv2 cell goes from **+0.21pp to −0.03pp, 95% CI [−1.07, +0.83]**.
  Across all 56 pairs the mean gain over fixed dense inference is **−3.5pp**. The
  abstract's "halves temporal budgets without accuracy loss" and the title's
  "Mitigating" do not survive as general claims and will be removed.
- **Statistics.** Correct that the published ANOVA treats 25 cells scoring the
  same clips as independent. Re-run as a within-subject design (clip = subject,
  each effect against its own subject×factor error term), partial η² is
  coverage **0.292**, stride **0.232**, interaction **0.070**. "Coverage is king
  at 2.2× stride" becomes **1.26×**. The per-model stride ordering is
  **unchanged** (TimeSformer 0.124 → SlowFast 0.331).
- **One correction, politely.** The stride×coverage interaction *is* reported —
  supplementary S13, grid decomposition, FineGym 15.0% of variance, RMSE 8.3pp.
  We accept the narrower version (not in the ANOVA model, only two datasets) and
  now provide it for all 56 pairs, where it reaches 0.271 on SlowFast/AUTSL.
- **TDS is a weak metric.** Each of the three sub-objections is testable and none
  changes the ranking: dropping the positive part is **numerically identical**
  (it never fires); integrating the whole curve as an AUC gives Spearman
  **+1.000**; normalising by baseline accuracy or by headroom above chance gives
  **+0.952**, swapping only adjacent pairs. FineGym > AUTSL > SSv2 is identical
  under all five definitions.
- **Hardware scaling.** Correct. The ×1.5 / ×10 / ×50 device factors are
  estimates, not measurements, and will be removed.
- **"8,000 configurations".** Total distinct grid evaluations are 13,875, but the
  specific 8×8×5×5×5 factorial is not complete. We will state the actual counts.

---

## Caveats to disclose rather than hide

- **Table 5 is not reproducible.** No model pool we tested reproduces the printed
  thresholds under absolute drop; relative drop matches 3 of 7 datasets almost
  exactly (SSv2 59.7/76.9 vs 67.5/82.3, UCF 1.4/7.5 vs 1.5/8.6, HMDB 8.4/28.5 vs
  9.6/30.1), suggesting the printed values are relative percentages mislabelled
  as "pp". The repo's own `taxonomy_summary.csv` agrees with our recomputation,
  not with the paper. Needs a decision from whoever generated it.
- **VideoMamba is missing** from the matched-evidence analysis: `mamba-ssm 2.3.1`
  is a CUDA 13 build while its environment ships torch cu128. It is one of the
  two `budget=8` models, so its absence weakens exactly the comparison that
  matters most.
- **FineGym is missing** from all re-analysis: source videos and checkpoints were
  lost. Its 55.9pp TDS is the paper's most-promoted number and, at a median of
  ~37 source frames, the most extreme instance of the padding artifact.

---

## Cut order if over length

Keep 1–4 whatever happens; they are the spine.

1. The mechanism (index table + seconds table)
2. Unpadded re-analysis (21/21) and the ρ=0.964 / 31pp residual pair
3. Matched-evidence survival (ρ=+0.821) and the C2 rewrite
4. Routing held-out numbers
5. Repeated-measures ANOVA
6. fps corrections
7. Convergence at 48px
8. TDS metric variants
9. Kinetics-400
10. Table 7 native-resolution control
