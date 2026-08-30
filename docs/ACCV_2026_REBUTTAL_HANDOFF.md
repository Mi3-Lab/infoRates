# ACCV 2026 — FineGym Handoff

For a Claude session starting cold on the machine that holds the FineGym videos
and checkpoints. Everything else has been completed on the Pinnacles cluster.

**Submission 230.** Reviews: XdvJ 5 (weak accept) / vdwp 4 (borderline, conf. 5)
/ pxtb 1 (reject). **Rebuttal deadline: 2 September 2026.**

---

## 0. Read this first: FineGym is NOT on the rebuttal critical path

The rebuttal is written, fits the one-page limit, and does not cite any FineGym
result that does not already exist. The one FineGym item a reviewer asked for
(vdwp: "Table 5 doesn't have FineGym") was answered from per-class data already
on disk — no new runs needed.

So the work below is for the **revised paper**, which matters only if the paper
is accepted or resubmitted. **Do not let it delay the rebuttal PDF.** If you have
limited time before 2 September, do nothing here and help with the rebuttal
instead.

---

## 1. Context: what the rebuttal argues

Two reviewers questioned whether the temporal sweep measures aliasing. It does
not, and we characterise what it does measure.

`select_frame_indices` re-uniformises the strided candidate pool to each model's
fixed frame budget `B`, so at fixed coverage the selected frames span the same
window at every stride; the input only changes once `ceil(window/s) < B`, and
the shortfall is filled by repeating the last frame. In seconds between sampled
frames, UCF-101 reads 0.395 at s=1, 2, 4 **and** 8.

Four independent measurements, all reproducible from `scripts/accv2026/`:

| measurement | value |
|---|---|
| stride 1→16 drop on clips that never pad | **−0.5pp** vs +10.7pp overall, 21/21 pairs |
| Spearman(padding rate, TDS) | **+0.964** |
| Spearman(frame budget, published drop) | **+0.849**, p=0.008 |
| same, under matched evidence | **+0.231**, p=0.58 |

What survives: TDS is invariant under five metric definitions (ρ≥0.95), and the
architecture ordering holds at matched evidence (**ρ=0.857, p=0.007**, 8 models).
Withdrawn: the 3–5× magnitude, which tracked input length.

Reproduce: `.venv/bin/python scripts/accv2026/rebuttal_padding_diagnostic.py`

---

## 2. Tone: do not over-concede

This matters more than any experiment here. The two positive reviewers each gave
a *specific, addressable* reservation, and both have now been addressed with
data. They have written grounds to raise their scores. An over-apologetic
framing can undo that by making a terminological problem look structural.

Rules of thumb used in `paper/rebuttal.tex`, keep them:
- Lead with what is unchanged, not with the correction.
- "imprecise", not "wrong". vdwp himself wrote "not well defined".
- Tie every new result to the request that prompted it — ACCV permits new
  analyses only "when they are explicitly requested by reviewers", and forbids
  "introducing new contributions".
- Never volunteer a problem no reviewer raised.

---

## 3. Bugs fixed here — check your copy before running anything

Diff `scripts/accv2026/sweep_coverage_stride.py` against yours.

1. **`_interp_pos_embed` import** — the name does not exist in `model_factory`
   (only `ModelFactory._interpolate_pos_embed`, different signature). The import
   was unconditional, so every TimeSformer/ViViT/VideoMAE load crashed. Now lazy.
2. **Checkpoint/resolution mismatch (silent, ~15pp)** — `get_checkpoint`
   preferred `*_224px_e10_h200` for every model, handing the 112px CNNs a
   224px-trained checkpoint. Now the native-resolution candidate wins.
3. **Unloadable checkpoints returned** — a directory can exist and still fail to
   load: `videomamba/diving48`'s `accv_meta.json` was renamed to `.bak` to retire
   it after it collapsed to 6.8% val_acc. `get_checkpoint` now validates
   loadability and falls through.
4. **FineGym CNN checkpoints unresolvable — yours** — `SPECIAL_CKPTS` names
   FineGym checkpoints for SlowFast and the Transformers but not for `r3d_18`,
   `mc3_18`, `r2plus1d_18`. A short-name fallback `accv2026_{model}_{dataset}`
   was added. Verify yours match one of:
   `accv2026_{model}_finegym`, `..._finegym_full_e10_a100`,
   `..._finegym_112px_e10_h200`. If not, add them to `SPECIAL_CKPTS` rather than
   renaming directories.
5. **VideoMamba is fine** — an earlier version of this file said its environment
   was broken. It is not: `mamba_ssm` imports on a GPU node and fails only on the
   login node, where the CUDA runtime it links against is absent. All four sweeps
   are complete here for all 8 datasets. **Nothing about VideoMamba is yours.**

---

## 4. Acceptance test — run this before trusting any number

`matched` at k = the model's frame budget samples k frames uniformly over the
clip, which is what `cov100/s1` does. The right checkpoint reproduces the
published value within ~1pp. This test caught two wrong-checkpoint bugs here; do
not skip it.

```bash
python scripts/accv2026/rebuttal_matched_evidence_sweep.py \
    --model r3d_18 --dataset finegym --mode matched --limit 200
```

| model | published FineGym cov100/s1 |
|---|---:|
| R3D-18 | 71.5% |
| MC3-18 | 64.2% |
| R(2+1)D | 76.8% |
| SlowFast-R50 | 78.2% |
| TimeSformer | 66.4% |
| ViViT | 69.9% |
| VideoMAE | 78.0% |
| VideoMamba | 72.7% |

If it does not match, stop and report. Do not proceed.

---

## 5. The runs, in priority order

```bash
for MODE in matched span covstride; do
  for M in r3d_18 mc3_18 r2plus1d_18 slowfast_r50 \
           timesformer vivit videomae videomamba; do
    python scripts/accv2026/rebuttal_matched_evidence_sweep.py \
      --model "$M" --dataset finegym --mode "$MODE"
  done
done
```

Completed pairs are skipped, so re-running only fills gaps. VideoMamba needs
`.venv_mamba` and a GPU node.

| mode | what it is for | priority |
|---|---|---|
| `matched` | same k distinct frames for every architecture; removes the frame-budget confound | **high** — this is the C2 argument |
| `span` | k pinned, window width varies; separates context from density | **high** — this is the R3 answer |
| `covstride` | published grid without repeat-last padding; isolates the frozen-tail share | medium |
| `antialias` | point vs box-filtered at same k and span | **skip** — inconclusive by construction (models were fine-tuned on sharp frames, so filtered input is out-of-distribution) and deliberately excluded from the rebuttal |

FineGym matters disproportionately here: its clips have a median of ~37 source
frames, the shortest of any dataset, so at budget 16 and stride 16 it delivers
~3 distinct frames plus 13 repeats — 99.2% of clips pad. It is the most extreme
instance of the effect and the paper's most-promoted number (TDS 55.9pp).

---

## 6. Where results go, and what to send back

```
evaluations/accv2026/rebuttal_sweeps/{mode}/{model}_finegym/
    per_clip.csv    per-clip outcomes — REQUIRED, the paired tests need it
    summary.csv     accuracy per config
```

Send **both** files per pair. `summary.csv` alone cannot support the paired
analyses.

Then, here, the numbers are folded in with:
```bash
.venv/bin/python scripts/accv2026/rebuttal_analyze_matched.py
.venv/bin/python scripts/accv2026/rebuttal_analyze_span.py
.venv/bin/python scripts/accv2026/rebuttal_verify_numbers.py   # must stay green
```

---

## 7. Known issue you may hit, and should not "fix" silently

The **Table 5 thresholds in the submitted paper are not reproducible** from the
repo's artifacts. Recomputing tertiles of per-class drop over the 8-model pool
gives AUTSL 46.7/57.3 against the printed 66.9/73.9; relative-drop definitions
match 3 of 7 datasets, suggesting the printed values are relative percentages
labelled as "pp". The repo's own `taxonomy_summary.csv` agrees with the
recomputation, not with the paper.

**No reviewer raised this**, and the rebuttal deliberately does not mention it.
Do not add it. It needs a decision from whoever generated the original table.
