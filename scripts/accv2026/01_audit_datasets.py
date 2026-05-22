#!/usr/bin/env python3
"""Audit datasets and build ACCV 2026 manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from info_rates.data.audit import (  # noqa: E402
    add_label_ids,
    attach_video_metadata,
    class_balanced_subset,
    iter_videos,
    load_diving48_pose_manifest,
    load_something_label_map,
    load_something_split,
    summarize_manifest,
    write_json,
    write_manifest,
)


def concat_nonempty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def audit_something(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_something_label_map(args.data_root)
    all_splits = []
    for split in args.splits:
        df = load_something_split(args.data_root, split)
        df = add_label_ids(df, label_map)
        all_splits.append(df)

    manifest = pd.concat(all_splits, ignore_index=True)
    manifest["exists"] = manifest["video_path"].map(lambda p: Path(p).exists())

    missing = manifest[~manifest["exists"]].copy()
    existing = manifest[manifest["exists"]].copy()

    if args.probe:
        probe_limit = None if args.probe_limit <= 0 else args.probe_limit
        probed = attach_video_metadata(existing, probe_limit=probe_limit)
        if probe_limit is not None and probe_limit < len(existing):
            remainder = existing.iloc[probe_limit:].copy()
            for col, value in {
                "readable": None,
                "fps": None,
                "num_frames": None,
                "duration": None,
                "width": None,
                "height": None,
                "error": "not_probed",
            }.items():
                remainder[col] = value
            manifest_out = concat_nonempty([probed, remainder, missing])
        else:
            for col, value in {
                "readable": False,
                "fps": 0.0,
                "num_frames": 0,
                "duration": 0.0,
                "width": 0,
                "height": 0,
                "error": "missing",
            }.items():
                missing[col] = value
            manifest_out = concat_nonempty([probed, missing])
    else:
        manifest_out = manifest

    manifest_path = out_dir / "somethingv2_manifest.csv"
    missing_path = out_dir / "somethingv2_missing.csv"
    summary_path = out_dir / "somethingv2_audit_summary.json"

    write_manifest(manifest_path, manifest_out)
    write_manifest(missing_path, missing)
    write_json(summary_path, summarize_manifest(manifest_out))

    if args.samples_per_class > 0:
        subset = class_balanced_subset(
            manifest_out,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
            require_readable=args.probe,
        )
        write_manifest(out_dir / f"somethingv2_subset_{args.samples_per_class}_per_class.csv", subset)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote missing-file report: {missing_path}")
    print(f"Wrote summary: {summary_path}")


def audit_video_tree(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in iter_videos(args.video_root):
        label = path.parent.name
        records.append(
            {
                "dataset": args.dataset,
                "split": args.split,
                "video_id": path.stem,
                "video_path": str(path),
                "label": label,
            }
        )
    manifest = pd.DataFrame(records)
    manifest["label_id"] = manifest["label"].astype("category").cat.codes if len(manifest) else []
    manifest["exists"] = True

    if args.probe and len(manifest):
        probe_limit = None if args.probe_limit <= 0 else args.probe_limit
        probed = attach_video_metadata(manifest, probe_limit=probe_limit)
        if probe_limit is not None and probe_limit < len(manifest):
            remainder = manifest.iloc[probe_limit:].copy()
            for col, value in {
                "readable": None,
                "fps": None,
                "num_frames": None,
                "duration": None,
                "width": None,
                "height": None,
                "error": "not_probed",
            }.items():
                remainder[col] = value
            remainder = remainder.reindex(columns=probed.columns)
            manifest = concat_nonempty([probed, remainder])
        else:
            manifest = probed

    manifest_path = out_dir / f"{args.dataset}_manifest.csv"
    summary_path = out_dir / f"{args.dataset}_audit_summary.json"
    write_manifest(manifest_path, manifest)
    write_json(summary_path, summarize_manifest(manifest))

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary: {summary_path}")


def audit_diving48_pose(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_diving48_pose_manifest(args.annotation_pkl, args.video_root)
    manifest["exists"] = manifest["video_path"].map(lambda p: Path(p).exists())
    if args.probe and len(manifest):
        probe_limit = None if args.probe_limit <= 0 else args.probe_limit
        probed = attach_video_metadata(manifest[manifest["exists"]], probe_limit=probe_limit)
        missing = manifest[~manifest["exists"]].copy()
        for col, value in {
            "readable": False,
            "fps": 0.0,
            "num_frames": 0,
            "duration": 0.0,
            "width": 0,
            "height": 0,
            "error": "missing",
        }.items():
            missing[col] = value
        if probe_limit is not None and probe_limit < int(manifest["exists"].sum()):
            remainder = manifest[manifest["exists"]].iloc[probe_limit:].copy()
            for col, value in {
                "readable": None,
                "fps": None,
                "num_frames": None,
                "duration": None,
                "width": None,
                "height": None,
                "error": "not_probed",
            }.items():
                remainder[col] = value
            remainder = remainder.reindex(columns=probed.columns)
            manifest = concat_nonempty([probed, remainder, missing.reindex(columns=probed.columns)])
        else:
            manifest = concat_nonempty([probed, missing.reindex(columns=probed.columns)])

    manifest_path = out_dir / "diving48_manifest.csv"
    summary_path = out_dir / "diving48_audit_summary.json"
    write_manifest(manifest_path, manifest)
    write_json(summary_path, summarize_manifest(manifest))

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sth = sub.add_parser("something", help="Audit Something-Something V2")
    sth.add_argument("--data-root", default="data/Something_data")
    sth.add_argument("--output-dir", default="evaluations/accv2026/manifests")
    sth.add_argument("--splits", nargs="+", default=["train", "validation"])
    sth.add_argument("--probe", action="store_true", help="Probe readable/fps/frame metadata")
    sth.add_argument("--probe-limit", type=int, default=0, help="0 means probe all existing files")
    sth.add_argument("--samples-per-class", type=int, default=0)
    sth.add_argument("--seed", type=int, default=42)
    sth.set_defaults(func=audit_something)

    tree = sub.add_parser("video-tree", help="Audit a class-folder video tree")
    tree.add_argument("--dataset", required=True)
    tree.add_argument("--video-root", required=True)
    tree.add_argument("--split", default="unknown")
    tree.add_argument("--output-dir", default="evaluations/accv2026/manifests")
    tree.add_argument("--probe", action="store_true")
    tree.add_argument("--probe-limit", type=int, default=0)
    tree.set_defaults(func=audit_video_tree)

    diving = sub.add_parser("diving48-pose", help="Audit Diving48 using OpenMMLab/PYSKL pkl annotations")
    diving.add_argument("--annotation-pkl", default="data/Diving48_data/annotations/diving48_hrnet.pkl")
    diving.add_argument("--video-root", default="data/Diving48_data/videos")
    diving.add_argument("--output-dir", default="evaluations/accv2026/manifests")
    diving.add_argument("--probe", action="store_true")
    diving.add_argument("--probe-limit", type=int, default=0)
    diving.set_defaults(func=audit_diving48_pose)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
