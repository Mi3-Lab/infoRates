#!/usr/bin/env python3
"""
Create full manifests for UCFCrime (video-based) and note FLAME is not usable.

UCFCrime: uses MP4 video clips reconstructed from sequential PNG frames.
FLAME: skipped — image classification dataset without temporal sequences.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
out_dir = ROOT / "evaluations/accv2026/manifests"
out_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# UCF-Crime: use reconstructed MP4 videos
# ============================================================
print("Creating UCFCrime manifest from MP4 videos...")
videos_path = ROOT / "data/UCFCrime_data/videos"

rows = []
for split_dir in videos_path.glob("*/"):
    split = "training" if split_dir.name == "Train" else "validation"
    for class_dir in split_dir.glob("*/"):
        label = class_dir.name
        for mp4 in class_dir.glob("*.mp4"):
            rows.append({
                "video_path": str(mp4),
                "video_id": mp4.stem,
                "label": label,
                "split": split,
                "dataset": "ufc_crime",
            })

df = pd.DataFrame(rows)
df["label_id"] = df.groupby("label").ngroup()
df["exists"] = df["video_path"].apply(lambda p: Path(p).exists())

out_path = out_dir / "ufc_crime_full.csv"
df.to_csv(out_path, index=False)
print(f"✅ Saved: {out_path}")
print(f"   Total: {len(df)} videos")
print(f"   Training: {len(df[df['split']=='training'])}")
print(f"   Validation: {len(df[df['split']=='validation'])}")
print(f"   Classes: {df['label'].nunique()}")
print(f"   Missing: {(~df['exists']).sum()}")

# ============================================================
# FLAME: not usable for video temporal aliasing research
# ============================================================
print("\n⚠️  FLAME skipped — it is an aerial image classification dataset.")
print("   Individual photos with no temporal sequences.")
print("   Not suitable for temporal aliasing research.")
