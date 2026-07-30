#!/usr/bin/env python3
"""Filter AUTSL split CSVs to videos that can be decoded.

The Kaggle mirror contains a small number of mp4 files with invalid containers
(`moov atom not found`). This script keeps the raw files in place, writes an
audit CSV, and replaces train/val split CSVs with readable-only rows while
preserving backups.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import pandas as pd


def is_readable(path: str) -> tuple[str, bool, int, str]:
    p = Path(path)
    if not p.exists():
        return path, False, 0, "missing"
    if p.stat().st_size <= 0:
        return path, False, 0, "empty"

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        cap.release()
        return path, False, 0, "open_failed"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, _ = cap.read()
    cap.release()
    if not ok:
        return path, False, frame_count, "read_failed"
    return path, True, frame_count, "ok"


def clean_split(split_csv: Path, audit_csv: Path, workers: int) -> tuple[int, int]:
    df = pd.read_csv(split_csv)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(is_readable, df["video_path"].astype(str).tolist()))

    audit = pd.DataFrame(rows, columns=["video_path", "readable", "frame_count", "reason"])
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_csv, index=False)

    keep = audit["readable"].astype(bool).to_numpy()
    clean = df.loc[keep].copy()

    backup = split_csv.with_suffix(split_csv.suffix + ".bak")
    if not backup.exists():
        split_csv.rename(backup)
    clean.to_csv(split_csv, index=False)
    return len(df), len(clean)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/AUTSL_data")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = Path(args.data_root)
    audit_dir = root / "splits" / "audit"

    for split in ("train", "val"):
        src = root / "splits" / f"{split}.csv"
        audit = audit_dir / f"{split}_readability.csv"
        total, kept = clean_split(src, audit, args.workers)
        print(f"{split}: kept {kept}/{total}, removed {total - kept}")
        if kept == 0:
            raise SystemExit(f"All {split} rows were filtered; aborting.")


if __name__ == "__main__":
    main()
