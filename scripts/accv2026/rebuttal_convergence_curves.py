#!/usr/bin/env python3
"""Convergence evidence for the low-resolution fine-tuning runs (reviewer XdvJ).

XdvJ's request:

    "The authors report fine-tuning all models for 10 additional epochs at the
     target resolution, but it is unclear whether this budget is sufficient for
     convergence across all architecture-dataset-resolution combinations. I
     believe that the authors should provide convergence evidence, e.g.
     training/validation curves, particularly for the 48px and 96px settings.
     My concern is not that 10 epochs are necessarily insufficient in general,
     but rather that the paper does not establish that the reported performance
     at severe off-native resolutions is not limited by the optimization budget."

The training scripts print one line per epoch to stdout, so the Slurm logs hold
the curves already. This reconstructs them per (model, dataset, resolution) and
answers the question the reviewer actually asked: was the run still improving
when the budget ran out?

For each run we report:
  best_epoch          epoch of peak validation accuracy
  gain_last_3         val_acc(best) - val_acc(best-3), how much was still being
                      won near the end
  still_improving     best epoch is 9 or 10, i.e. the ceiling was never reached
                      inside the budget

A run that peaks at epoch 10 with a meaningful late gain is one where 10 epochs
was plausibly not enough, and that is exactly what XdvJ is asking us to check.

Usage:  .venv/bin/python scripts/accv2026/rebuttal_convergence_curves.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "evaluations/accv2026/logs"
OUT = ROOT / "evaluations/accv2026/rebuttal"

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+):\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)\s+val_acc=([\d.]+)")
# The retrain scripts announce the run before the epoch lines.
RES_RE = re.compile(r"Spatial resolution:\s*(\d+)px")
MODEL_RE = re.compile(r"\b(r3d_18|mc3_18|r2plus1d_18|slowfast_r50|timesformer|vivit|videomae|videomamba)\b")
DATASET_RE = re.compile(r"\b(ucf101|ssv2|somethingv2|hmdb51|diving48|autsl|driveact|epic_kitchens|finegym)\b")
SAVE_RE = re.compile(r"accv2026_([a-z0-9_]+?)_([a-z0-9_]+?)_(\d+)px_e\d+")


def parse(path: Path) -> list[dict]:
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return []
    epochs = EPOCH_RE.findall(text)
    if not epochs:
        return []

    res = model = dataset = None
    m = SAVE_RE.search(text)
    if m:
        model, dataset, res = m.group(1), m.group(2), int(m.group(3))
    if res is None:
        r = RES_RE.search(text)
        res = int(r.group(1)) if r else None
    if model is None:
        mm = MODEL_RE.search(text)
        model = mm.group(1) if mm else None
    if dataset is None:
        dd = DATASET_RE.search(text)
        dataset = dd.group(1) if dd else None
    if res is None or model is None:
        return []

    curve = [(int(e), float(va)) for e, _, _, _, va in epochs]
    curve.sort()
    return [dict(log=path.name, model=model,
                 dataset="ssv2" if dataset == "somethingv2" else dataset,
                 resolution=res, n_epochs=len(curve),
                 curve=curve)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for f in sorted(LOGS.glob("*.out")):
        runs.extend(parse(f))
    if not runs:
        print("no training curves found in the Slurm logs")
        return

    rows = []
    for r in runs:
        curve = r["curve"]
        accs = {e: a for e, a in curve}
        best_epoch = max(accs, key=lambda e: accs[e])
        best = accs[best_epoch]
        prior = [accs[e] for e in accs if e <= best_epoch - 3]
        rows.append(dict(
            model=r["model"], dataset=r["dataset"], resolution=r["resolution"],
            n_epochs=r["n_epochs"], best_epoch=best_epoch,
            best_val_acc=round(100 * best, 2),
            gain_last_3=round(100 * (best - max(prior)), 2) if prior else float("nan"),
            still_improving=best_epoch >= max(r["n_epochs"] - 1, 1) and r["n_epochs"] >= 5,
            log=r["log"],
        ))
    df = pd.DataFrame(rows).drop_duplicates(
        subset=["model", "dataset", "resolution", "best_val_acc"])
    df.to_csv(OUT / "convergence_curves.csv", index=False)

    print(f"parsed {len(df)} fine-tuning runs across "
          f"{df.resolution.nunique()} resolutions\n")

    low = df[df.resolution.isin([48, 96])]
    print("=" * 78)
    print("Low-resolution runs (48px / 96px) — the ones XdvJ asked about")
    print("=" * 78)
    if low.empty:
        print("  none found")
    else:
        show = low.sort_values(["resolution", "model", "dataset"])
        print(show[["model", "dataset", "resolution", "n_epochs", "best_epoch",
                    "best_val_acc", "gain_last_3", "still_improving"]]
              .to_string(index=False))
        print(f"\n  still improving at the end: "
              f"{int(low.still_improving.sum())}/{len(low)} runs")
        print(f"  mean accuracy still being gained in the last 3 epochs: "
              f"{low.gain_last_3.mean():.2f} pp")

    print("\n" + "=" * 78)
    print("By resolution — is the 10-epoch budget binding?")
    print("=" * 78)
    summary = df.groupby("resolution").agg(
        runs=("best_epoch", "size"),
        mean_best_epoch=("best_epoch", "mean"),
        pct_still_improving=("still_improving", lambda s: 100 * s.mean()),
        mean_gain_last_3=("gain_last_3", "mean"),
    )
    print(summary.round(2).to_string())
    print(f"\nWrote {OUT / 'convergence_curves.csv'}")


if __name__ == "__main__":
    main()
