"""E3 — Spectral Analysis: optical flow magnitude → temporal frequency per action class.

For each dataset, estimates the dominant temporal frequency of each action class
by computing optical flow magnitude statistics across sampled frames, then
correlates with aliasing sensitivity (from E1).

Since computing full optical flow on all videos is expensive, we:
1. Sample N=20 videos per class (already in the manifest)
2. Compute mean inter-frame optical flow magnitude
3. Use this as proxy for temporal frequency demand
4. Correlate with class-level aliasing sensitivity from E5 taxonomy

Outputs:
  evaluations/accv2026/e3_spectral/{dataset}_flow_stats.csv
  evaluations/accv2026/e3_spectral/flow_aliasing_correlation.csv
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from scipy import stats

MANIFESTS = Path("evaluations/accv2026/manifests")
OUT       = Path("evaluations/accv2026/e3_spectral")
TAXONOMY  = Path("evaluations/accv2026/e5_taxonomy")
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "ssv2":          "somethingv2_val_20_per_class.csv",
    "hmdb51":        "hmdb51_val_20_per_class.csv",
    "diving48":      "diving48_val_20_per_class.csv",
    "ucf101":        "ucf101_val_20_per_class.csv",
    "autsl":         "autsl_val_20_per_class.csv",
    "driveact":      "driveact_val_20_per_class.csv",
    "epic_kitchens": "epic_kitchens_val_20_per_class.csv",
}

MAX_VIDEOS_PER_CLASS = 5   # sample 5 videos per class (fast estimate)
MAX_FRAMES           = 16  # sample 16 frames per video
FRAME_SIZE           = (112, 112)  # resize for speed

def compute_flow_magnitude(video_path: str, n_frames: int = 16):
    """Estimate mean inter-frame optical flow magnitude (Farneback)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return None

    # Sample frames evenly
    indices = np.linspace(0, max(0, total - 1), n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, FRAME_SIZE)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
    cap.release()

    if len(frames) < 2:
        return None

    # Compute flow between consecutive pairs
    magnitudes = []
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i+1], None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
        magnitudes.append(mag)

    return float(np.mean(magnitudes)) if magnitudes else None


corr_rows = []

for dataset, manifest_file in DATASETS.items():
    manifest_path = MANIFESTS / manifest_file
    if not manifest_path.exists():
        print(f"  {dataset}: manifest not found, skipping")
        continue

    taxonomy_path = TAXONOMY / f"{dataset}_class_taxonomy.csv"
    if not taxonomy_path.exists():
        print(f"  {dataset}: taxonomy not found, run E5 first")
        continue

    manifest = pd.read_csv(manifest_path)
    # Filter to val split and existing files
    if "split" in manifest.columns:
        manifest = manifest[manifest["split"].isin(["val", "validation", "test"])].copy()
    manifest = manifest[manifest["exists"] == True].copy() if "exists" in manifest.columns else manifest

    taxonomy = pd.read_csv(taxonomy_path)

    print(f"  {dataset}: {len(manifest)} videos, {manifest['label_id'].nunique()} classes")

    flow_rows = []
    classes = sorted(manifest["label_id"].unique())

    for label_id in classes:
        class_videos = manifest[manifest["label_id"] == label_id]["video_path"].tolist()
        # Sample at most MAX_VIDEOS_PER_CLASS
        sample = class_videos[:MAX_VIDEOS_PER_CLASS]

        mags = []
        for vp in sample:
            mag = compute_flow_magnitude(str(vp), MAX_FRAMES)
            if mag is not None:
                mags.append(mag)

        if mags:
            flow_rows.append({
                "dataset":        dataset,
                "label_id":       label_id,
                "n_videos":       len(mags),
                "mean_flow_mag":  np.mean(mags),
                "std_flow_mag":   np.std(mags),
                "max_flow_mag":   np.max(mags),
            })

    if not flow_rows:
        print(f"    → no flow computed")
        continue

    flow_df = pd.DataFrame(flow_rows)
    flow_df.to_csv(OUT / f"{dataset}_flow_stats.csv", index=False)

    # Merge with taxonomy
    merged = flow_df.merge(taxonomy[["label_id","mean_abs_drop","mean_rel_drop","sensitivity"]], on="label_id", how="inner")
    if len(merged) < 5:
        print(f"    → insufficient merged data ({len(merged)} classes)")
        continue

    # Pearson correlation: flow magnitude vs aliasing sensitivity
    r_abs, p_abs = stats.pearsonr(merged["mean_flow_mag"], merged["mean_abs_drop"])
    r_rel, p_rel = stats.pearsonr(merged["mean_flow_mag"], merged["mean_rel_drop"])

    print(f"    Flow vs aliasing: r={r_abs:.3f} (abs drop), r={r_rel:.3f} (rel drop), n={len(merged)}")

    # Mean flow per sensitivity tier
    tier_means = merged.groupby("sensitivity")["mean_flow_mag"].mean()

    corr_rows.append({
        "dataset": dataset,
        "n_classes": len(merged),
        "pearson_r_abs":  r_abs,
        "pearson_p_abs":  p_abs,
        "pearson_r_rel":  r_rel,
        "pearson_p_rel":  p_rel,
        "significant":    p_abs < 0.05,
        "flow_high_tier": tier_means.get("High", np.nan),
        "flow_mod_tier":  tier_means.get("Moderate", np.nan),
        "flow_low_tier":  tier_means.get("Low", np.nan),
    })

if corr_rows:
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT / "flow_aliasing_correlation.csv", index=False)

    print("\n" + "="*70)
    print("E3 — Flow magnitude vs aliasing sensitivity (Pearson r)")
    print("Nyquist prediction: higher flow → more aliasing (positive correlation)")
    print("="*70)
    print(f"\n{'Dataset':<15} {'r(abs)':>8} {'p':>8} {'r(rel)':>8} {'p':>8} {'Sig?':>5}")
    print("-"*55)
    for _, row in corr_df.iterrows():
        sig = "✅" if row["significant"] else "  "
        print(f"{row['dataset']:<15} {row['pearson_r_abs']:>8.3f} {row['pearson_p_abs']:>8.4f} "
              f"{row['pearson_r_rel']:>8.3f} {row['pearson_p_rel']:>8.4f} {sig}")
else:
    print("\nNo correlation data computed — check video paths")
