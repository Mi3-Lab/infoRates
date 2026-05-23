#!/usr/bin/env python3
"""Create small class-balanced manifests for fast ACCV pilot runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def balanced(df: pd.DataFrame, samples_per_class: int, seed: int) -> pd.DataFrame:
    parts = []
    for _, group in df.groupby("label_id", sort=False):
        n = min(len(group), samples_per_class)
        parts.append(group.sample(n, random_state=seed))
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-exists", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if args.split != "all" and "split" in df.columns:
        df = df[df["split"].astype(str) == args.split].copy()
    if args.require_exists and "exists" in df.columns:
        df = df[df["exists"].astype(bool)].copy()
    out = balanced(df, args.samples_per_class, args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} rows to {path}")
    print(f"Classes: {out['label_id'].nunique() if 'label_id' in out else 'n/a'}")


if __name__ == "__main__":
    main()
