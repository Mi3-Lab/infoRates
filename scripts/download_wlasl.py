#!/usr/bin/env python3
"""Download WLASL from HuggingFace (aipieces/WLASL)."""
import os, zipfile, requests, sys
from pathlib import Path

BASE = Path("/data/wesleyferreiramaia/infoRates/data/WLASL_data")
BASE.mkdir(parents=True, exist_ok=True)
VIDEOS = BASE / "raw_videos"
VIDEOS.mkdir(exist_ok=True)

HF_BASE = "https://huggingface.co/datasets/aipieces/WLASL/resolve/main"

# Download annotation JSON
json_path = BASE / "WLASL_v0.3.json"
if not json_path.exists():
    print("Downloading WLASL_v0.3.json...")
    r = requests.get(f"{HF_BASE}/WLASL_v0.3.json", timeout=60, stream=True)
    with open(json_path, "wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
    print(f"  -> {json_path.stat().st_size / 1024:.0f} KB")

# Download 14 MP4 zip shards
for i in range(1, 15):
    shard = f"shard_{i:03d}_014.zip"
    url = f"{HF_BASE}/start_kit/raw_videos_mp4/{shard}"
    zip_path = BASE / shard
    if not zip_path.exists():
        print(f"Downloading {shard}...")
        r = requests.get(url, timeout=300, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        print(f"  -> {zip_path.stat().st_size / 1024**2:.0f} MB")
    
    # Extract
    print(f"Extracting {shard}...")
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if member.endswith(".mp4"):
                dest = VIDEOS / Path(member).name
                if not dest.exists():
                    with z.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())

n_videos = len(list(VIDEOS.glob("*.mp4")))
print(f"\nDone! {n_videos} MP4 files in {VIDEOS}")
