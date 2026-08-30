#!/usr/bin/env python3
"""Does multi-stride augmentation fix the cliff? (reviewer XdvJ)

XdvJ's doubt, and the reason they hold at weak accept:

    "the observed 'aliasing cliffs' likely reflect inference fragility under
     standard training rather than fundamental architectural limits"

The ablation trains two augmented arms and compares them against the published
checkpoint, all three evaluated under the *published* sampler (`covstride_pad`),
because that is the protocol whose cliff is in question. Evaluating the arms
without padding would remove the cliff by construction and answer nothing.

  baseline  the published fine-tuned checkpoint. No run needed: covstride_pad
            reproduces select_frame_indices exactly, so the published sweep in
            coverage_stride_sweep/ already is this condition.
  paper     trained with random (coverage, stride) drawn through the published
            sampler, padding included.
  fixed     trained with the same grid but uniform resampling, no frozen tail.

Prediction from the padding mechanism: `paper` should NOT recover the cliff. At
stride 16 on AUTSL a model receives ~4 distinct frames and 12 copies of the last
one, and no training recipe extracts 16 frames of evidence from 4. Recovery
would mean the padding diagnosis is incomplete.

Usage:  .venv/bin/python scripts/accv2026/rebuttal_analyze_tra.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "evaluations/accv2026/coverage_stride_sweep"
ARMS_DIR = ROOT / "evaluations/accv2026/rebuttal_sweeps/covstride_pad"
OUT = ROOT / "evaluations/accv2026/rebuttal"

STRIDES = [1, 2, 4, 8, 16]
ARMS = ["paper", "fixed"]


def baseline_curve(model: str, dataset: str) -> dict[int, float] | None:
    """Published stride curve at coverage 100%."""
    for suffix in ("", "_trainres224", "_res224"):
        f = PUBLISHED / f"{model}_{dataset}{suffix}" / "sweep_summary.csv"
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df = df[df.coverage == 100]
        try:
            return {s: 100 * float(df[df.stride == s].top1.iloc[0]) for s in STRIDES}
        except Exception:
            return None
    return None


def arm_curve(model: str, dataset: str, arm: str) -> dict[int, float] | None:
    f = ARMS_DIR / f"{model}_{dataset}_tra_{arm}" / "summary.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    out = {}
    for s in STRIDES:
        row = df[df.config == f"cov100_s{s}"]
        if row.empty:
            return None
        out[s] = 100 * float(row.top1.iloc[0])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = sorted({(p.name.split("_tra_")[0].rsplit("_", 0)[0], )
                    for p in ARMS_DIR.glob("*_tra_*")}) if ARMS_DIR.exists() else []
    # Recover (model, dataset) from directory names of the form
    # "<model>_<dataset>_tra_<arm>", where model itself may contain underscores.
    known_models = ["r2plus1d_18", "slowfast_r50", "timesformer", "videomamba",
                    "videomae", "vivit", "r3d_18", "mc3_18"]
    found = set()
    if ARMS_DIR.exists():
        for p in ARMS_DIR.glob("*_tra_*"):
            stem = p.name.split("_tra_")[0]
            for m in known_models:
                if stem.startswith(m + "_"):
                    found.add((m, stem[len(m) + 1:]))
                    break

    if not found:
        print(f"no TRA evaluations yet under {ARMS_DIR}")
        return

    rows = []
    for model, dataset in sorted(found):
        base = baseline_curve(model, dataset)
        if base is None:
            print(f"[skip] no published baseline for {model}/{dataset}")
            continue
        entry = {"model": model, "dataset": dataset}
        entry.update({f"base_s{s}": round(base[s], 2) for s in STRIDES})
        entry["base_drop"] = round(base[1] - base[16], 2)
        for arm in ARMS:
            cur = arm_curve(model, dataset, arm)
            if cur is None:
                continue
            entry.update({f"{arm}_s{s}": round(cur[s], 2) for s in STRIDES})
            entry[f"{arm}_drop"] = round(cur[1] - cur[16], 2)
            # Fraction of the baseline cliff that the arm recovers at stride 16.
            gap = base[1] - base[16]
            entry[f"{arm}_recovered_pp"] = round(cur[16] - base[16], 2)
            entry[f"{arm}_recovered_pct"] = (
                round(100 * (cur[16] - base[16]) / gap, 1) if gap > 0 else float("nan"))
        rows.append(entry)

    if not rows:
        print("no comparable pairs")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tra_ablation.csv", index=False)

    for r in df.itertuples():
        print("=" * 78)
        print(f"{r.model} / {r.dataset}  —  evaluated under the published sampler")
        print("=" * 78)
        print(f"{'arm':<10s}" + "".join(f"{'s'+str(s):>9s}" for s in STRIDES)
              + f"{'drop':>9s}{'recovered':>11s}")
        print(f"{'baseline':<10s}"
              + "".join(f"{getattr(r, f'base_s{s}'):9.1f}" for s in STRIDES)
              + f"{r.base_drop:9.1f}{'—':>11s}")
        for arm in ARMS:
            if not hasattr(r, f"{arm}_drop"):
                continue
            rec = getattr(r, f"{arm}_recovered_pct")
            print(f"{arm:<10s}"
                  + "".join(f"{getattr(r, f'{arm}_s{s}'):9.1f}" for s in STRIDES)
                  + f"{getattr(r, f'{arm}_drop'):9.1f}"
                  + f"{rec:10.1f}%")
        print()
        print("  'recovered' = share of the baseline stride-1 -> stride-16 cliff")
        print("  regained at stride 16. Near zero means the cliff is not a")
        print("  training-inference mismatch, because training on the degraded")
        print("  distribution does not repair it.")
        print()

    print(f"Wrote {OUT / 'tra_ablation.csv'}")


if __name__ == "__main__":
    main()
