#!/usr/bin/env python3
"""Probe which FineGym source videos are still on YouTube.

The FineGym annotations and videos are gone from both /data and /scratch, but
evaluations/accv2026/manifests/finegym_full.csv survived, and every clip id
encodes its full download spec:

    {yt_id}_E_{event_start}_{event_end}_A_{action_start}_{action_end}

So the download can be reconstructed from the manifest alone. This script only
answers the go/no-go question first: how many of the 130 source videos are
still reachable, and how much of the clip set they still cover. Run it before
committing hours of downloading.

Usage:  .venv/bin/python scripts/accv2026/finegym_probe_availability.py [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "evaluations/accv2026/manifests/finegym_full.csv"
OUT = ROOT / "evaluations/accv2026/rebuttal/finegym_availability.csv"
YTDLP = ROOT / ".venv/bin/yt-dlp"

CLIP_RE = re.compile(
    r"^(?P<yt>.+?)_E_(?P<ev_s>\d+)_(?P<ev_e>\d+)_A_(?P<act_s>\d+)_(?P<act_e>\d+)$")


def parse_manifest() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST)
    df["clip_id"] = df.video_path.map(lambda p: Path(p).stem)
    spec = df.clip_id.str.extract(CLIP_RE)
    missing = spec.yt.isna().sum()
    if missing:
        print(f"[WARN] {missing} clip ids did not parse and are ignored")
    return pd.concat([df, spec], axis=1).dropna(subset=["yt"])


def probe(yt_id: str) -> dict:
    """Ask yt-dlp for metadata only; no media is fetched."""
    cmd = [str(YTDLP), "--skip-download", "--dump-json",
           "--no-warnings", "--socket-timeout", "20",
           f"https://www.youtube.com/watch?v={yt_id}"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=90)
    except subprocess.TimeoutExpired:
        return dict(yt=yt_id, available=False, reason="timeout", duration=None)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", "replace").strip().splitlines()
        reason = err[-1][:160] if err else f"exit {r.returncode}"
        return dict(yt=yt_id, available=False, reason=reason, duration=None)
    try:
        meta = json.loads(r.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return dict(yt=yt_id, available=False, reason="bad json", duration=None)
    return dict(yt=yt_id, available=True, reason="", duration=meta.get("duration"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = parse_manifest()
    ids = sorted(df.yt.unique())
    print(f"manifest: {len(df)} clips, {df.groupby(['yt','ev_s','ev_e']).ngroups} events, "
          f"{len(ids)} source videos")
    print(f"probing with {args.workers} workers (metadata only)...\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(probe, ids))

    res = pd.DataFrame(results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    alive = set(res[res.available].yt)
    n_ok, n_bad = len(alive), len(res) - len(alive)
    clips_ok = int(df.yt.isin(alive).sum())
    val = df[df.split == "validation"]
    val_ok = int(val.yt.isin(alive).sum())

    print(f"videos available : {n_ok}/{len(res)}  ({100 * n_ok / len(res):.1f}%)")
    print(f"clips recoverable: {clips_ok}/{len(df)}  ({100 * clips_ok / len(df):.1f}%)")
    print(f"  of which validation: {val_ok}/{len(val)}  ({100 * val_ok / max(len(val),1):.1f}%)")

    classes_before = df.label.nunique()
    classes_after = df[df.yt.isin(alive)].label.nunique()
    print(f"classes still covered: {classes_after}/{classes_before}")

    if n_bad:
        print(f"\ntop failure reasons ({n_bad} videos):")
        for reason, cnt in res[~res.available].reason.value_counts().head(5).items():
            print(f"  {cnt:3d}x  {reason[:100]}")

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
