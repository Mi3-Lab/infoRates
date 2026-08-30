#!/usr/bin/env python3
"""Is any of the measured loss actually aliasing?

An interventional test, and the only one here that targets aliasing directly --
but see the limitation below, which makes it conclusive in one direction only.
Both arms deliver the same k samples over the same span; they differ only in
what each sample is:

  point  the frame at position t_i
  box    the mean of the frames in the window of width T/k centred on t_i,
         i.e. the clip is low-pass filtered before it is sampled, so no energy
         survives above the Nyquist limit k/2T that the sampling imposes

Aliasing has a defining signature: removing above-Nyquist energy before sampling
*recovers* accuracy, because the folded energy that was corrupting the sampled
signal is gone. This is the same intervention used to demonstrate spatial
aliasing in anti-aliased CNNs, applied on the temporal axis.

IMPORTANT LIMITATION -- read before interpreting.

The models were fine-tuned on sharp frames. A box-filtered clip is therefore
out-of-distribution for them, and they would lose accuracy on it even if no
aliasing existed at all. The anti-aliased-CNN literature avoids this by applying
the filter during training as well; we apply it only at inference. The test is
consequently asymmetric in what it can establish:

  box > point   INFORMATIVE. A gain despite the distribution-shift handicap is
                strong evidence that above-Nyquist energy was corrupting the
                sampled signal, i.e. genuine aliasing.
  box < point   AMBIGUOUS. Consistent with "no aliasing", but equally consistent
                with "blurred input is out-of-distribution". This test cannot
                separate the two, and a negative result here must not be
                reported as evidence against aliasing.

Removing the ambiguity requires fine-tuning on box-filtered inputs so that both
arms are in-distribution. Until that is run, treat a negative result as weak
corroboration only, and rest the argument on the span sweep and the augmentation
ablation, neither of which has this confound.


Usage:  .venv/bin/python scripts/accv2026/rebuttal_analyze_antialias.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
SWEEPS = ROOT / "evaluations/accv2026/rebuttal_sweeps/antialias"
OUT = ROOT / "evaluations/accv2026/rebuttal"
KS = [1, 2, 4, 8, 16]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not SWEEPS.exists():
        print(f"no antialias results yet under {SWEEPS}")
        return

    rows = []
    for f in sorted(SWEEPS.glob("*/summary.csv")):
        d = pd.read_csv(f)
        model, dataset = d.model.iloc[0], d.dataset.iloc[0]
        acc = {str(r.config): 100 * r.top1 for r in d.itertuples()}
        for k in KS:
            p, b = acc.get(f"k{k}_point"), acc.get(f"k{k}_box")
            if p is None or b is None:
                continue
            rows.append(dict(model=model, dataset=dataset, k=k,
                             point=round(p, 2), box=round(b, 2),
                             delta=round(b - p, 2)))
    if not rows:
        print("no paired point/box configurations found")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "antialias.csv", index=False)

    print("=" * 78)
    print("Anti-aliasing intervention: box (low-passed) minus point (%)")
    print("=" * 78)
    piv = df.pivot_table(index=["dataset"], columns="k", values="delta",
                         aggfunc="mean")
    print(piv.round(2).to_string())

    print("\nBy architecture:")
    print(df.pivot_table(index="model", columns="k", values="delta",
                         aggfunc="mean").round(2).to_string())

    print("\n" + "=" * 78)
    print("Verdict per k (paired over model-dataset pairs)")
    print("=" * 78)
    for k in KS:
        sub = df[df.k == k]
        if len(sub) < 5:
            continue
        mean = sub.delta.mean()
        try:
            stat, p = wilcoxon(sub.box, sub.point)
        except ValueError:
            p = float("nan")
        n_pos = int((sub.delta > 0).sum())
        if mean > 0.5 and p < 0.05:
            verdict = "ALIASING: low-pass recovers accuracy despite the OOD handicap"
        elif abs(mean) <= 0.5:
            verdict = "no gain (ambiguous: could be no aliasing, or OOD offset)"
        else:
            verdict = "loss (ambiguous: blur is out-of-distribution for these models)"
        print(f"  k={k:<3d} mean delta={mean:+6.2f}pp  "
              f"positive in {n_pos}/{len(sub)}  p={p:.4f}   {verdict}")

    print("\n  Only a GAIN is informative here. The models were trained on sharp")
    print("  frames, so box-filtered input is out-of-distribution and costs")
    print("  accuracy independently of aliasing. A loss therefore cannot be")
    print("  reported as evidence against aliasing; it is simply uninformative.")
    print("  A clean test requires fine-tuning on filtered input as well.")
    print(f"\nWrote {OUT / 'antialias.csv'}")


if __name__ == "__main__":
    main()
