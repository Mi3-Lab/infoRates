#!/usr/bin/env python3
"""Prepare dataset directories and download user-provided dataset files."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


DATASET_DIRS = {
    "somethingv2": [
        "data/Something_data/raw_archives",
        "data/Something_data/videos",
        "data/Something_data/labels",
    ],
    "diving48": [
        "data/Diving48_data/raw_archives",
        "data/Diving48_data/videos",
        "data/Diving48_data/annotations",
    ],
    "finegym": [
        "data/FineGym_data/raw_archives",
        "data/FineGym_data/videos",
        "data/FineGym_data/annotations",
        "data/FineGym_data/features",
    ],
    "accv": [
        "evaluations/accv2026/manifests",
        "evaluations/accv2026/metrics",
    ],
    "flame": [
        "data/FLAME_data/raw_archives",
        "data/FLAME_data/videos",
    ],
    "ego4d": [
        "data/Ego4D_data/annotations",
        "data/Ego4D_data/clips",
    ],
    "ucf_crime": [
        "data/UCFCrime_data/raw_archives",
        "data/UCFCrime_data/videos",
    ],
}


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("+", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def init_dirs(args: argparse.Namespace) -> None:
    datasets = args.datasets or list(DATASET_DIRS)
    for dataset in datasets:
        if dataset not in DATASET_DIRS:
            raise SystemExit(f"Unknown dataset group: {dataset}")
        for relative in DATASET_DIRS[dataset]:
            path = ROOT / relative
            print(f"mkdir -p {path}")
            if not args.dry_run:
                path.mkdir(parents=True, exist_ok=True)


def check_tools(_: argparse.Namespace) -> None:
    tools = ["curl", "wget", "tar", "unzip", "ffmpeg"]
    for tool in tools:
        found = shutil.which(tool)
        status = found if found else "MISSING"
        print(f"{tool}: {status}")


def download_file(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("curl"):
        cmd = [
            "curl",
            "-L",
            "--fail",
            "--continue-at",
            "-",
            "--output",
            str(output),
            args.url,
        ]
    elif shutil.which("wget"):
        cmd = ["wget", "-c", "-O", str(output), args.url]
    else:
        raise SystemExit("Neither curl nor wget is available")
    run(cmd, dry_run=args.dry_run)


def download_manifest(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise SystemExit("Download manifest must be a JSON list")

    for item in items:
        url = item["url"]
        output = item["output"]
        download_file(argparse.Namespace(url=url, output=output, dry_run=args.dry_run))


def extract_archive(args: argparse.Namespace) -> None:
    archive = Path(args.archive)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffixes = "".join(archive.suffixes)
    if suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz") or suffixes.endswith(".tar"):
        cmd = ["tar", "-xf", str(archive), "-C", str(output_dir)]
    elif suffixes.endswith(".zip"):
        cmd = ["unzip", "-n", str(archive), "-d", str(output_dir)]
    else:
        raise SystemExit(f"Unsupported archive type: {archive}")
    run(cmd, dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-dirs", help="Create dataset/output directories")
    init.add_argument("--datasets", nargs="+", choices=sorted(DATASET_DIRS), default=None)
    init.add_argument("--dry-run", action="store_true")
    init.set_defaults(func=init_dirs)

    tools = sub.add_parser("check-tools", help="Check local command-line tools")
    tools.set_defaults(func=check_tools)

    one = sub.add_parser("download-file", help="Download one user-provided URL with resume support")
    one.add_argument("--url", required=True)
    one.add_argument("--output", required=True)
    one.add_argument("--dry-run", action="store_true")
    one.set_defaults(func=download_file)

    many = sub.add_parser("download-manifest", help="Download files listed in a JSON manifest")
    many.add_argument("--manifest", required=True)
    many.add_argument("--dry-run", action="store_true")
    many.set_defaults(func=download_manifest)

    extract = sub.add_parser("extract", help="Extract .zip/.tar/.tar.gz archives")
    extract.add_argument("--archive", required=True)
    extract.add_argument("--output-dir", required=True)
    extract.add_argument("--dry-run", action="store_true")
    extract.set_defaults(func=extract_archive)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
