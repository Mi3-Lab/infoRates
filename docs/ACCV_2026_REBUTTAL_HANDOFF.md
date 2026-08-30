# ACCV 2026 Rebuttal — Handoff for the FineGym Machine

**Read this whole file before running anything.** It is written for a Claude
session starting cold on the machine that still holds the FineGym videos and
FineGym checkpoints. Everything else is being run on the Pinnacles HPC cluster.

Submission 230, *"Measuring and Mitigating Spatiotemporal Aliasing in Video
Action Recognition: A Cross-Architecture Analysis at Scale."* Reviews are in:
**5 (weak accept) / 4 (borderline accept) / 1 (reject)**. Rebuttal window is one
week from 2026-08-27.

---

## 1. Why this work exists

Two reviewers argue the paper does not measure aliasing. **They are right**, and
we verified the mechanism in the code. This is the single most important thing
to understand before touching anything.

### The protocol

`src/info_rates/evaluation/benchmark.py::select_frame_indices`:

```python
window     = round(total_frames * coverage / 100)
candidates = arange(0, window, stride)
if len(candidates) >= budget:
    pick = linspace(0, len(candidates) - 1, budget)   # re-uniformizes!
    return candidates[pick]
pad = full(budget - len(candidates), candidates[-1])  # repeats the LAST frame
return concatenate([candidates, pad])
```

### What that actually does

Because `linspace` re-spreads the candidates over the whole window, **at fixed
coverage the selected frames span the same interval at every stride**. For a
400-frame clip at `budget=16`:

| stride | distinct frames | span | mean gap |
|---:|---:|---|---:|
| 1 | 16 | [0, 399] | 26.6 |
| 16 | 16 | [0, 384] | 25.6 |

Stride changes almost nothing. It only bites once
`ceil(window / stride) < budget`, and then the deficit is filled by **repeating
the last frame**. For a 60-frame clip at stride 16 the model receives
`[0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 48, 48, 48]` — 4 distinct
frames and a frozen tail.

So the "stride" axis is not a sampling-density axis. It is a
*how-much-of-the-input-degenerates* axis, and it only activates on short clips.

### The evidence

1. **Padding rates at stride 16, coverage 100%** — AUTSL, SSv2 and DriveAct are
   at **100%** for every model; UCF-101 (the lowest-TDS dataset) is the lowest.
2. **Paired test** — restricted to clips that never pad, the stride 1→16 drop
   collapses from **+10.7pp to −0.5pp**, in **21 of 21** (model, dataset) pairs
   that have a usable unpadded subset.
3. **Frame budget confound** — the two "robust" models (TimeSformer, VideoMamba)
   are exactly the two with `budget=8`; the fragile ones include both
   `budget=32` models. `Spearman(budget, avg drop) = 0.849, p = 0.0077`.
4. **But it is not the whole story** — `Spearman(%padding, TDS) = +0.964`, yet
   among the three datasets that are 100% saturated the TDS still ranges from
   21.9 to 53.0pp. **Padding gates the effect; content modulates its magnitude.**

That last point is the rebuttal's central honest claim, and the `matched`
experiment below is what quantifies it.

Reproduce all of this with:

```bash
.venv/bin/python scripts/accv2026/rebuttal_padding_diagnostic.py
```

---

## 2. Bugs found in the repo — check these on your machine too

These were live in `scripts/accv2026/sweep_coverage_stride.py`. They are fixed
here; **your copy may still have them.** Diff before running.

### 2.1 `_interp_pos_embed` import (fatal)

The HF-transformer branch imported `_interp_pos_embed` from
`info_rates.models.model_factory` unconditionally. That name does not exist —
only `ModelFactory._interpolate_pos_embed`, with a different signature. Any
TimeSformer / ViViT / VideoMAE load crashed. Fixed by making the import lazy so
it only runs on the non-native-resolution path.

### 2.2 Checkpoint / input resolution mismatch (silent, ~15pp)

`get_checkpoint` preferred `accv2026_{model}_{dataset}_224px_e10_h200` for every
model. For the three 112px CNNs that returned a **224px-retrained checkpoint**
while the sweep feeds 112px frames. `r3d_18/ucf101` scored 66.0% instead of
81.2%. `PROGRESS.md:111` claims this was fixed — it was fixed in a *different*
script, never here.

Fixed by ordering candidates so the native-resolution checkpoint wins.
**Acceptance test** (this must pass before you trust any number):

```
matched mode, k = budget  ==  the published cov100/s1 accuracy
r3d_18 / ucf101:  k16 = 81.2%   vs published 81.2%   ✅
```

### 2.3 FineGym CNN checkpoints cannot resolve — **this one is yours**

`SPECIAL_CKPTS` spells out FineGym checkpoints for SlowFast, TimeSformer, ViViT,
VideoMAE and VideoMamba, but **not** for `r3d_18`, `mc3_18`, `r2plus1d_18`. A
short-name fallback `accv2026_{model}_{dataset}` was added. Verify your FineGym
CNN checkpoints match one of:

```
accv2026_{model}_finegym                     # short form, what the others use
accv2026_{model}_finegym_full_e10_a100       # native 112px, standard naming
accv2026_{model}_finegym_112px_e10_h200
```

If they are named otherwise, add them to `SPECIAL_CKPTS` rather than renaming
directories.

### 2.4 Not a bug, but do not trust it

`README.md` reports 48px family accuracies (CNN 48.2%, Transformer 10.4%) that
**contradict** `dashboard/data/p3_results.csv` (24.6% / 50.3%) and have the
ordering inverted. The README predates the bicubic-PE fix; 10.4% is the
collapsed value S8 describes as "the wrong approach". `p3_results.csv` is
authoritative. These stale numbers appear **only** in the README, not in the
submitted paper.

---

## 3. What you need to run

Everything for the 7 non-FineGym datasets is running on the cluster. **FineGym
is the only gap**, because its videos and checkpoints were lost there and only
exist on your machine.

### 3.1 Sanity first

```bash
# does the checkpoint resolve, and does it reproduce the paper?
python - <<'PY'
import sys; sys.path.insert(0,'src'); sys.path.insert(0,'scripts/accv2026')
from sweep_coverage_stride import get_checkpoint, MODEL_CFG
for m in MODEL_CFG:
    print(f'{m:14s} resize={MODEL_CFG[m]["resize"]:3d} -> {get_checkpoint(m,"finegym").name}')
PY
```

Then run one model and check `matched` k=budget against the published FineGym
stride-1 accuracy in the table below. If it does not match within ~1pp, stop and
report — do not proceed.

| model | published FineGym stride-1 | drop at stride 16 |
|---|---:|---:|
| R3D-18 | 71.5% | 61.4 |
| MC3-18 | 64.2% | 51.1 |
| R(2+1)D | 76.8% | 64.2 |
| SlowFast-R50 | 78.2% | 71.5 |
| TimeSformer | 66.4% | 36.1 |
| ViViT | 69.9% | 55.2 |
| VideoMAE | 78.0% | 67.8 |
| VideoMamba | 72.7% | 40.8 |

### 3.2 The four sweeps

```bash
for MODE in matched span antialias covstride; do
  for M in r3d_18 mc3_18 r2plus1d_18 slowfast_r50 \
           timesformer vivit videomae videomamba; do
    python scripts/accv2026/rebuttal_matched_evidence_sweep.py \
      --model "$M" --dataset finegym --mode "$MODE"
  done
done
```

Use `--limit 64` for a smoke test first. Completed pairs are skipped
automatically, so re-running only fills gaps. VideoMamba needs `.venv_mamba`.

**What each mode is for:**

| mode | design | answers |
|---|---|---|
| `matched` | every model gets the **same k distinct frames** at the same positions (k ∈ 1,2,4,8,16), resampled to its own input length | do architectures still differ once evidence is equal? kills the budget confound |
| `span` | k = native budget, drawn from a **centered** window of width W | varies true sampling rate k/W — the axis the paper never varied |
| `antialias` | same k, same span, **point-sampled vs temporally box-filtered** | the only causal test of the Nyquist hypothesis — see below |
| `covstride` | the submitted 5×5 grid, but adapting by **uniform resampling instead of repeat-last** | how much of the "cliff" was the frozen tail |

**`antialias` deserves special attention.** It is the one experiment that can
decide whether any of this is aliasing. Both arms deliver k samples over the same
span; `point` takes the frame at each position, `box` takes the mean of the frames
in the window around it, which low-passes the clip so no energy survives above the
Nyquist limit the sampling imposes. If the loss is aliasing, removing that energy
**must** recover accuracy — that is the defining signature, and the same test used
for spatial aliasing in anti-aliased CNNs. If the loss is missing evidence,
filtering cannot help, because it adds no evidence. Either outcome is publishable:
one rescues the Nyquist framing with a direct measurement, the other gives the
reframing a causal basis instead of an argument.

Only `k ≤ 8` is strictly matched across all models, since the smallest budget is
8. Larger k is recorded but only usable by models with `budget ≥ k`.

**Methodological caveat to keep in the write-up:** with a fixed frame budget,
sampling rate = k/W, so density cannot be varied without changing k or W. True
temporal aliasing is *not isolable* by frame subsampling alone. This is the
strongest argument for dropping the Nyquist framing.

### 3.3 FineGym-specific facts that matter

FineGym clips have a **median of ~37 source frames** (S8), the shortest of any
dataset. At `budget=16, stride=16` that gives `ceil(37/16) = 3` distinct frames
plus 13 copies — **99.2% of clips pad**. FineGym's headline 55.9pp TDS, the
paper's most-promoted finding, is the most extreme instance of the artifact.
Getting FineGym through `matched` is therefore the highest-value single result
in the whole rebuttal.

### 3.4 VideoMamba — also yours, and it is not optional

**VideoMamba cannot run on the cluster at all.** Both environments are broken:

```
.venv        ModuleNotFoundError: No module named 'mamba_ssm'
.venv_mamba  ImportError: libcudart.so.13: cannot open shared object file
```

`.venv_mamba` ships torch 2.8.0+cu128 and only `libcudart.so.12`, but the
installed `mamba-ssm 2.3.1` wheel was compiled against **CUDA 13** — the residue
of the fake-nvcc build workaround. Checked and ruled out:

- `nvidia-cuda-runtime-cu13` on PyPI is a 1.2 kB placeholder, not the runtime.
- Neither `state-spaces/mamba` nor `Dao-AILab/causal-conv1d` publishes a
  **py39 + torch2.8** wheel. Releases stop at torch 2.7 for cp39.
- Rebuilding from source uses the system nvcc, which is CUDA 13 only, and would
  reproduce the same mismatch.

The only local fix is downgrading torch to 2.7 in `.venv_mamba`, which risks
breaking the environment that produced the paper's published VideoMamba numbers.
Not worth it a week before the deadline.

**So VideoMamba runs on your machine, alongside FineGym.** Verify first:

```bash
.venv_mamba/bin/python -c "import mamba_ssm; print('ok')"
```

If that works, run VideoMamba across **all 8 datasets** in all three modes, not
just FineGym:

```bash
for MODE in matched span antialias covstride; do
  .venv_mamba/bin/python scripts/accv2026/rebuttal_matched_evidence_sweep.py \
    --model videomamba --dataset <each of the 8> --mode "$MODE"
done
```

This matters more than it looks. VideoMamba and TimeSformer are the paper's two
"robust" architectures, and they are also **exactly the two models with
`budget=8`** — the confound at the heart of the rebuttal. Without VideoMamba the
matched-evidence test only has one of the two, and the central question ("do
architectures still differ once evidence is equal?") cannot be answered
properly. It is the single highest-value item on your machine after FineGym.

Note the sbatch here activates `.venv`, not `.venv_mamba`; adapt accordingly.

### 3.5 Multi-stride augmentation ablation, if you have training capacity

Reviewer XdvJ's primary doubt is whether the cliff reflects a training-inference
sampling mismatch rather than anything architectural. `train_with_tra.py` answers
it with two arms, both trained with random (coverage, stride) exposure:

```bash
python scripts/accv2026/train_with_tra.py \
    --model timesformer --dataset finegym --arm paper --epochs 10
# then --arm fixed
```

`--arm paper` augments using the **published** sampler, padding and all;
`--arm fixed` uses uniform resampling with no frozen tail. The baseline arm needs
no training — the published checkpoint is it. The script dispatches to
`train_torchvision` for the CNNs and `train_transformers` otherwise, since their
`make_loader` signatures differ.

Evaluate the resulting checkpoints with the sweep, **not** by their validation
accuracy: the validation loader is unaugmented, so `val_acc` is measured under
dense sampling and says nothing about stride robustness.

```bash
python scripts/accv2026/rebuttal_matched_evidence_sweep.py \
    --model timesformer --dataset finegym --mode covstride \
    --checkpoint .../accv2026_timesformer_finegym_tra_paper --tag tra_paper
```

`--tag` keeps each arm in its own output directory; without it the arms overwrite
each other silently.

**Prediction from the mechanism:** the `paper` arm should *not* recover the
cliff, because no training recipe lets a model extract 16 frames of evidence from
4. If it does recover, the padding diagnosis is incomplete and we need to know
before a reviewer finds out.

### 3.6 Also worth doing on your machine

- **Taxonomy Table 5.** The paper's Table 5 omits FineGym (reviewer vdwp flagged
  it). `evaluations/accv2026/e5_taxonomy/finegym_class_taxonomy.csv` already has
  it: Low ≤ 50.4pp, High > 71.2pp, 33 High / 32 Low. **Warning:** the published
  Table 5 thresholds are **not reproducible** from the current 8-model
  artifacts (AUTSL: paper 66.9/73.9 vs computed 46.7/57.3). They look like they
  came from a CNN-only pool. Regenerate the whole table, do not patch one row.
- **FineGym fps.** `e10_clip_duration.py` hardcodes approximate fps and FineGym
  is absent from the dict, so it silently defaults to 25. Measured real values
  elsewhere: HMDB-51 is 30 (not 25), AUTSL is 30 (not 25), EPIC is 50 (not 60).
  Measure FineGym's actual fps and report it.

---

## 4. Where results go

```
evaluations/accv2026/rebuttal_sweeps/{mode}/{model}_finegym/
    per_clip.csv    per-clip outcomes, needed for paired analysis
    summary.csv     accuracy per config
evaluations/accv2026/rebuttal/
    *.csv           diagnostic outputs
```

Ship back **both** files per pair — `per_clip.csv` is what makes the paired
tests possible; `summary.csv` alone is not enough.

---

## 5. Reviewer-by-reviewer state

### Reviewer pxtb — Reject (confidence 4)

Core objections are correct. Two are already settled:

- **Routing.** Correct that the rule thresholds `max_c p(c|v)`, not entropy —
  "entropy routing" is a misnomer and must be renamed. Correct that it is tuned
  and scored on the same split. Held-out (1000 stratified 50/50 splits): the
  headline TimeSformer/SSv2 cell goes from **+0.21pp to −0.03pp, 95% CI
  [−1.07, +0.83]**. Across 56 pairs the mean gain vs fixed dense is **−3.5pp**.
  The abstract's "halves temporal budgets without accuracy loss" and the title's
  "Mitigating" do not survive as general claims.
- **One factual error to correct politely.** They write that the stride×coverage
  interaction "is also not reported". It *is* — `supplementary.tex:793`, grid
  decomposition, FineGym 15.0% of variance, RMSE 8.3pp. Concede the narrower
  version (it is not in the ANOVA model and covers only 2 datasets) and commit
  to adding the interaction term for all 64 pairs.

Still open: repeated-measures statistics, TDS as curve AUC rather than an
endpoint difference, measured end-to-end compute.

### Reviewer vdwp — Borderline accept (confidence **5**, expert)

The most movable reviewer. Wants: the aliasing definition fixed (the `span`
sweep is the answer), stride expressed in **seconds** rather than frames since
fps varies 12–50 across datasets, Kinetics-400 included, FineGym added to
Table 5, consistent model ordering across figures.

Their Table 7 suspicion is **confirmed**: fine-tuning at the *native* resolution
— where there is no resolution to adapt to — already gains **+6 to +10.6pp** for
7 of 8 models. So the headline "+39.2pp from target-resolution fine-tuning" is
really ≈32pp resolution + ≈7pp extra training.

Kinetics-400 is viable on the cluster with **zero training**: 8000 clips are on
disk and the manifest's label order was verified identical to the pretrained
`id2label`. Being handled there, not by you.

### Reviewer XdvJ — Weak accept (confidence 4)

Wants a multi-stride augmentation ablation (needs training), convergence curves
at 48/96px (W&B logs exist), the no-finetune column in Table 7 (**done**), §3.3
text on resize + PE interpolation, and a note on SSv2 in Table S19.

Their question about how frame count, temporal span and source-frame interval
vary with stride and coverage is answered exactly by §1 of this document.

---

## 6. Strategy

Concede precisely, then reframe. The padding threshold is itself a real,
unreported, deployment-relevant finding: **there is a critical stride, a
function of clip length and frame budget, at which a fixed-budget pipeline
degenerates — and it is dataset-dependent.** That claim is correct and useful.
"We measured temporal aliasing" is not.

Do not oversell, do not hide the confound, and do not let any number into the
rebuttal that has not been reproduced from raw data.
