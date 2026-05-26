#!/usr/bin/env python3
"""Download EPIC-Kitchens pre-extracted clips from lightly-ai/epic-kitchens-100-clips."""
import os, sys, requests, concurrent.futures
from pathlib import Path

BASE = Path("/data/wesleyferreiramaia/infoRates/data/EPIC_data/clips")
BASE.mkdir(parents=True, exist_ok=True)

HF_BASE = "https://huggingface.co/datasets/lightly-ai/epic-kitchens-100-clips/resolve/main"

from huggingface_hub import list_repo_files
print("Listing all clip files...")
all_files = list(list_repo_files("lightly-ai/epic-kitchens-100-clips", repo_type="dataset"))
mp4_files = [f for f in all_files if f.endswith(".mp4")]
print(f"Found {len(mp4_files)} MP4 clips")

# Also download annotation files
ann_files = [f for f in all_files if "annotation" in f.lower() and not f.endswith(".mp4")]
print(f"Found {len(ann_files)} annotation files")

def download_file(rel_path):
    dest = BASE.parent / rel_path  # data/EPIC_data/clips/P01/P01_102_0.mp4
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return f"skip {rel_path}"
    url = f"{HF_BASE}/{rel_path}"
    try:
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        return f"ok {rel_path}"
    except Exception as e:
        return f"err {rel_path}: {e}"

# Download annotations first
print("Downloading annotation files...")
for f in ann_files:
    result = download_file(f)
    print(f"  {result}")

# Download clips in parallel
print(f"Downloading {len(mp4_files)} clips with 16 workers...")
done = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    futures = {pool.submit(download_file, f): f for f in mp4_files}
    for fut in concurrent.futures.as_completed(futures):
        result = fut.result()
        done += 1
        if done % 1000 == 0:
            print(f"  {done}/{len(mp4_files)} done")
        if result.startswith("err"):
            print(f"  ERROR: {result}")

n_clips = len(list(BASE.glob("**/*.mp4")))
print(f"\nDone! {n_clips} clips in {BASE}")
