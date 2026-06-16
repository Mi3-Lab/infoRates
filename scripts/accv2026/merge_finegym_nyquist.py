#!/usr/bin/env python3
"""Merge FineGym flow cutoff results into the Nyquist resolution analysis.

Expected input from the machine with FineGym videos:
  evaluations/accv2026/e3_spectral/finegym_cutoff_freq.csv

Existing local input:
  evaluations/accv2026/e3_spectral/finegym_stride_sensitivity.csv
  evaluations/accv2026/e3_spectral/nyquist_resolution_validation.csv

Output:
  evaluations/accv2026/e3_spectral/nyquist_resolution_validation_with_finegym.csv

The script uses only the Python standard library so it can run in the paper
environment even when pandas/scipy are unavailable.
"""
from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "evaluations/accv2026/e3_spectral"
BASE_CSV = BASE / "nyquist_resolution_validation.csv"
FINEGYM_CUTOFF = BASE / "finegym_cutoff_freq.csv"
FINEGYM_STRIDE = BASE / "finegym_stride_sensitivity.csv"
OUT_CSV = BASE / "nyquist_resolution_validation_with_finegym.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[order[k]] = avg_rank
        i = j
    return out


def pearson(x: list[float], y: list[float]) -> float:
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (den_x * den_y)


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(ranks(x), ranks(y))


def permutation_pvalue(x: list[float], y: list[float], observed: float, n_perm: int = 100_000) -> float:
    rng = random.Random(0)
    yr = ranks(y)
    xr = ranks(x)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(yr)
        if abs(pearson(xr, yr)) >= abs(observed):
            count += 1
    return (count + 1) / (n_perm + 1)


def main() -> None:
    missing = [p for p in [BASE_CSV, FINEGYM_CUTOFF, FINEGYM_STRIDE] if not p.exists()]
    if missing:
        for p in missing:
            print(f"Missing required file: {p}")
        raise SystemExit(1)

    rows = read_csv(BASE_CSV)
    rows = [r for r in rows if r["dataset"].lower() != "finegym"]

    cutoff_by_res = {int(r["resolution"]): r for r in read_csv(FINEGYM_CUTOFF)}
    stride_by_res = {int(r["resolution"]): float(r["stride_sensitivity"])
                     for r in read_csv(FINEGYM_STRIDE)}

    for res in sorted(set(cutoff_by_res) & set(stride_by_res)):
        c = cutoff_by_res[res]
        rows.append({
            "dataset": "finegym",
            "resolution": res,
            "n_videos": int(c["n_videos"]),
            "cutoff_freq": float(c["cutoff_freq"]),
            "stride_sensitivity": stride_by_res[res],
        })

    rows = sorted(rows, key=lambda r: (str(r["dataset"]), int(r["resolution"])))
    fieldnames = ["dataset", "resolution", "n_videos", "cutoff_freq", "stride_sensitivity"]
    write_csv(OUT_CSV, rows, fieldnames)

    x = [float(r["cutoff_freq"]) for r in rows]
    y = [float(r["stride_sensitivity"]) for r in rows]
    rho = spearman(x, y)
    p = permutation_pvalue(x, y, rho)

    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_dataset[str(r["dataset"])].append(r)

    print(f"Saved: {OUT_CSV}")
    print(f"Spearman rho={rho:.3f}, permutation p={p:.5f}, n={len(rows)}")
    print("\nLaTeX rows for Table S12:")
    means = []
    for dataset, ds_rows in by_dataset.items():
        mean_fc = sum(float(r["cutoff_freq"]) for r in ds_rows) / len(ds_rows)
        mean_stride = sum(float(r["stride_sensitivity"]) for r in ds_rows) / len(ds_rows)
        means.append((mean_stride, dataset, mean_fc))
    for mean_stride, dataset, mean_fc in sorted(means, reverse=True):
        label = {
            "autsl": "AUTSL",
            "ssv2": "SSv2",
            "driveact": "DriveAct",
            "hmdb51": "HMDB-51",
            "diving48": "Diving-48",
            "epic_kitchens": "EPIC-Kitchens",
            "ucf101": "UCF-101",
            "finegym": "FineGym",
        }.get(dataset, dataset)
        print(f"{label:<13} & {mean_fc:0.3f} & {mean_stride:5.2f} \\\\")


if __name__ == "__main__":
    main()
