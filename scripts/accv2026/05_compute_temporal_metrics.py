#!/usr/bin/env python3
"""Compute temporal robustness metrics from a fixed-budget summary CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from info_rates.metrics.temporal_robustness import compute_temporal_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    metrics = compute_temporal_metrics(summary)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out, index=False)
    print(f"Wrote metrics: {out}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
