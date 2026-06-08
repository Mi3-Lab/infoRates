#!/usr/bin/env python3
"""Audit FLAME and UCFCrime datasets and create manifests."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def audit_flame() -> pd.DataFrame:
    """Audit FLAME dataset from PNG frames."""
    flame_path = ROOT / "data/FLAME_data/raw_archives"
    flame_base = None

    # Find the actual FLAME folder (has long name)
    for item in flame_path.iterdir():
        if item.is_dir() and "FlameVision" in item.name:
            flame_base = item / "FlameVision" / "Classification"
            break

    if not flame_base or not flame_base.exists():
        print("[WARN] FLAME dataset structure not found")
        return pd.DataFrame()

    rows = []
    for split_dir in flame_base.iterdir():
        if not split_dir.is_dir():
            continue
        split = split_dir.name  # test, train, valid

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            label = class_dir.name  # fire, nofire, etc.

            for img_file in class_dir.glob("*.png"):
                # Extract video ID from filename
                # Format: "name (XXXX).png" → video_id = "name"
                match = re.match(r"(.+?)\s*\(\d+\)", img_file.stem)
                video_id = match.group(1) if match else img_file.stem

                rows.append({
                    "video_path": str(img_file),
                    "video_id": video_id,
                    "label": label,
                    "split": split,
                    "dataset": "flame",
                })

    df = pd.DataFrame(rows)
    print(f"[FLAME] Found {len(df)} frames across {df['split'].nunique()} splits")
    print(f"        Classes: {sorted(df['label'].unique())}")
    return df


def audit_ufc_crime() -> pd.DataFrame:
    """Audit UCF-Crime dataset from PNG frames."""
    crime_path = ROOT / "data/UCFCrime_data/raw_archives"

    if not crime_path.exists():
        print("[WARN] UCF-Crime dataset path not found")
        return pd.DataFrame()

    rows = []
    for split_dir in crime_path.iterdir():
        if not split_dir.is_dir() or split_dir.name not in ["Train", "Test"]:
            continue
        split = "training" if split_dir.name == "Train" else "validation"

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            label = class_dir.name

            for img_file in class_dir.glob("*.png"):
                # Extract video ID: "Abuse043_x264_1870" → "Abuse043"
                match = re.match(r"([a-zA-Z0-9]+)_x264_\d+", img_file.stem)
                video_id = match.group(1) if match else img_file.stem

                rows.append({
                    "video_path": str(img_file),
                    "video_id": video_id,
                    "label": label,
                    "split": split,
                    "dataset": "ufc_crime",
                })

    df = pd.DataFrame(rows)
    print(f"[UCF-Crime] Found {len(df)} frames across {df['split'].nunique()} splits")
    print(f"            Classes: {sorted(df['label'].unique())}")
    return df


def create_balanced_manifest(df: pd.DataFrame, dataset_name: str, samples_per_class: int = 20) -> pd.DataFrame:
    """Create class-balanced manifest (20 samples per class per split)."""
    if df.empty:
        return df

    manifests = []
    for split in df["split"].unique():
        split_df = df[df["split"] == split].copy()
        parts = []

        for label in split_df["label"].unique():
            label_df = split_df[split_df["label"] == label]
            n_samples = min(len(label_df), samples_per_class)
            sampled = label_df.sample(n=n_samples, random_state=42)
            parts.append(sampled)

        if parts:
            manifests.append(pd.concat(parts, ignore_index=True))

    result = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame()

    if not result.empty:
        # Add label_id mapping
        label_map = {label: i for i, label in enumerate(sorted(result["label"].unique()))}
        result["label_id"] = result["label"].map(label_map)
        result["exists"] = result["video_path"].map(lambda p: Path(p).exists())

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["flame", "ufc_crime", "all"], default="all")
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else (ROOT / "evaluations/accv2026/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if args.dataset in ["flame", "all"]:
        print("\n=== Auditing FLAME ===")
        df_flame = audit_flame()
        if not df_flame.empty:
            manifest_flame = create_balanced_manifest(df_flame, "flame", args.samples_per_class)
            out_path = out_dir / "flame_val_20_per_class.csv"
            manifest_flame.to_csv(out_path, index=False)
            print(f"✓ Wrote manifest: {out_path}")
            print(f"  {len(manifest_flame)} frames, {manifest_flame['label'].nunique()} classes")
            results["flame"] = manifest_flame

    if args.dataset in ["ufc_crime", "all"]:
        print("\n=== Auditing UCF-Crime ===")
        df_crime = audit_ufc_crime()
        if not df_crime.empty:
            manifest_crime = create_balanced_manifest(df_crime, "ufc_crime", args.samples_per_class)
            out_path = out_dir / "ufc_crime_val_20_per_class.csv"
            manifest_crime.to_csv(out_path, index=False)
            print(f"✓ Wrote manifest: {out_path}")
            print(f"  {len(manifest_crime)} frames, {manifest_crime['label'].nunique()} classes")
            results["ufc_crime"] = manifest_crime

    print(f"\nDone. Manifests saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
