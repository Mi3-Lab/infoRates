#!/usr/bin/env python3
"""Extract Ego4D FHO-LTA action segments and build a training manifest.

After running `ego4d --datasets clip_256ss --benchmarks FHO -y`, this script:
  1. Reads FHO LTA train/val annotations
  2. Extracts short action segments (3–8 s) from the downloaded clip_256ss videos
  3. Stores them at {data_root}/action_clips/{verb_slug}/{clip_uid}_a{idx}.mp4
  4. Writes {data_root}/splits/ego4d_manifest.csv

Usage:
    python preprocess_ego4d.py [--data-root /scratch/.../Ego4D_data] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


ANN_DIR = "v2/annotations"
CLIPS_DIR = "v2/clips"     # ego4d 'clips' dataset (pre-cut benchmark clips)
OUTPUT_CLIPS_DIR = "action_clips"
SPLITS_DIR = "splits"


def slug(verb: str) -> str:
    """Convert verb string to a clean directory name.

    'put_(place,_leave,_drop)' → 'put'
    """
    # Take only the first word (before any parenthesis or comma)
    base = re.split(r"[\s_\(,]", verb.strip())[0]
    return base.lower()


def extract_segment(args):
    """Worker: extract one action segment with ffmpeg. Returns (dst_path, ok, msg)."""
    clip_path, dst_path, t_start, t_end = args
    if dst_path.exists():
        return str(dst_path), True, "exists"
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(t_end - t_start, 1.0)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t_start:.3f}",
        "-i", str(clip_path),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",           # no audio
        "-loglevel", "error",
        str(dst_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return str(dst_path), False, result.stderr.decode()[:200]
    return str(dst_path), True, "ok"


def process_split(split_name: str, clips_list: list, clips_dir: Path,
                  out_dir: Path, workers: int) -> list[dict]:
    """Extract segments for one split, return manifest rows."""
    tasks = []
    for entry in clips_list:
        clip_uid = entry["clip_uid"]
        verb = entry["verb"]
        action_idx = entry["action_idx"]
        t_start = entry["action_clip_start_sec"]
        t_end = entry["action_clip_end_sec"]

        clip_path = clips_dir / f"{clip_uid}.mp4"
        if not clip_path.exists():
            continue  # clip not downloaded

        verb_slug = slug(verb)
        dst_name = f"{clip_uid}_a{action_idx:04d}.mp4"
        dst_path = out_dir / verb_slug / dst_name

        tasks.append((clip_path, dst_path, t_start, t_end, verb_slug, entry))

    print(f"  [{split_name}] {len(tasks)} segments to extract "
          f"(from {len(set(e[5]['clip_uid'] for e in tasks))} clips)")

    manifest_rows = []
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {
            exe.submit(extract_segment, (cp, dp, ts, te)): (dp, vs, e)
            for cp, dp, ts, te, vs, e in tasks
        }
        for i, fut in enumerate(as_completed(futures), 1):
            dst_path, verb_slug, entry = futures[fut]
            _, ok, msg = fut.result()
            if ok:
                manifest_rows.append({
                    "video_path": str(dst_path),
                    "label": verb_slug,
                    "split": split_name,
                    "clip_uid": entry["clip_uid"],
                    "verb_full": entry["verb"],
                    "action_idx": entry["action_idx"],
                })
            else:
                failed += 1
                if failed <= 10:
                    print(f"    [WARN] {dst_path.name}: {msg}")
            if i % 1000 == 0:
                print(f"    [{split_name}] {i}/{len(tasks)} done, {failed} failed")

    print(f"  [{split_name}] Done: {len(manifest_rows)} ok, {failed} failed")
    return manifest_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/scratch/wesleyferreiramaia/infoRates/data/Ego4D_data")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--min-examples-per-class", type=int, default=10,
                        help="Drop verb classes with fewer than this many total examples")
    parser.add_argument("--top-n-verbs", type=int, default=50,
                        help="Only use the top N most common verb classes (0=all)")
    args = parser.parse_args()

    root = Path(args.data_root)
    ann_dir = root / ANN_DIR
    clips_dir = root / CLIPS_DIR
    out_dir = root / OUTPUT_CLIPS_DIR
    splits_dir = root / SPLITS_DIR
    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data root: {root}")
    print(f"Clips dir: {clips_dir}")
    available = len(list(clips_dir.glob("*.mp4"))) if clips_dir.exists() else 0
    print(f"Downloaded clips: {available}")

    with open(ann_dir / "fho_lta_train.json") as f:
        train_ann = json.load(f)
    with open(ann_dir / "fho_lta_val.json") as f:
        val_ann = json.load(f)

    train_clips = train_ann["clips"]
    val_clips = val_ann["clips"]
    print(f"Train entries: {len(train_clips)}, Val entries: {len(val_clips)}")

    # Deduplicate: same clip_uid + action_idx can appear multiple times in LTA
    # (the interval window slides, same action observed from different contexts)
    def dedup(entries):
        seen = set()
        out = []
        for e in entries:
            key = (e["clip_uid"], e["action_idx"])
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out

    train_clips = dedup(train_clips)
    val_clips = dedup(val_clips)
    print(f"After dedup — Train: {len(train_clips)}, Val: {len(val_clips)}")

    # Filter to top-N verbs if requested
    if args.top_n_verbs > 0:
        verb_counts = Counter(c["verb"] for c in train_clips)
        top_verbs = {v for v, _ in verb_counts.most_common(args.top_n_verbs)}
        train_clips = [c for c in train_clips if c["verb"] in top_verbs]
        val_clips = [c for c in val_clips if c["verb"] in top_verbs]
        print(f"Top-{args.top_n_verbs} verbs filter: Train={len(train_clips)}, Val={len(val_clips)}")

    # Remap action_idx to be unique per clip_uid within split (some clips appear
    # in both train and val, their action indices are relative to the clip)
    # Re-assign local index to avoid filename collision if same clip in both splits
    for split_clips in (train_clips, val_clips):
        uid_counter: dict[str, int] = {}
        for e in split_clips:
            uid = e["clip_uid"]
            uid_counter[uid] = uid_counter.get(uid, -1) + 1
            e["action_idx"] = uid_counter[uid]

    all_rows: list[dict] = []

    print("\n=== Extracting train segments ===")
    all_rows.extend(process_split("train", train_clips, clips_dir, out_dir, args.workers))

    print("\n=== Extracting val segments ===")
    all_rows.extend(process_split("val", val_clips, clips_dir, out_dir, args.workers))

    df = pd.DataFrame(all_rows)
    print(f"\nTotal rows: {len(df)}")

    # Filter classes with too few examples
    class_counts = Counter(df["label"])
    keep_classes = {c for c, n in class_counts.items() if n >= args.min_examples_per_class}
    df_filtered = df[df["label"].isin(keep_classes)].copy()
    dropped = len(class_counts) - len(keep_classes)
    print(f"Classes: {len(class_counts)} total, dropped {dropped} with <{args.min_examples_per_class} examples")
    print(f"Kept {len(keep_classes)} classes, {len(df_filtered)} samples")

    # Write manifest
    manifest_path = splits_dir / "ego4d_manifest.csv"
    df_filtered.to_csv(manifest_path, index=False)
    print(f"\nManifest written: {manifest_path}")
    print(df_filtered["split"].value_counts().to_string())
    print(f"\nTop 10 classes:\n{df_filtered['label'].value_counts().head(10).to_string()}")


if __name__ == "__main__":
    main()
