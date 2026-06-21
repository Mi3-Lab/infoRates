"""E3-v2 — Spectral Analysis: temporal burstiness of optical flow vs aliasing.

Extends e3_spectral_analysis.py (which only uses mean flow magnitude) with
temporal features that capture *when* motion happens within a clip, not just
how much there is overall:

  temporal_std  – std of inter-frame flow magnitudes within a clip (averaged
                  across sampled videos per class).  High std = motion
                  concentrated in bursts rather than sustained throughout.

  temporal_cv   – temporal_std / mean_flow (coefficient of variation).
                  Normalises burstiness by base motion level.

  peak_ratio    – max(frame_flows) / mean(frame_flows) per video, averaged.
                  How prominent is the peak motion relative to the baseline.

Hypothesis (Opção A):
  "Discrete-event" classes have BURST motion: one brief salient moment, quiet
  otherwise → high temporal_std / cv / peak_ratio.
  These classes should alias MORE (positive correlation with aliasing).
  If confirmed, this validates the Nyquist analogy at the CLASS level even
  when mean flow is inversely correlated (as observed for FineGym).

Outputs (evaluations/accv2026/e3_spectral/):
  {dataset}_flow_stats_v2.csv          per-class temporal flow stats
  flow_aliasing_correlation_v2.csv     all-dataset correlation table
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from scipy import stats

ROOT      = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "evaluations/accv2026/manifests"
OUT       = ROOT / "evaluations/accv2026/e3_spectral"
TAXONOMY  = ROOT / "evaluations/accv2026/e5_taxonomy"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "finegym":       "finegym_val_20_per_class.csv",
    "autsl":         "autsl_val_20_per_class.csv",
    "ssv2":          "somethingv2_val_20_per_class.csv",
    "driveact":      "driveact_val_20_per_class.csv",
    "diving48":      "diving48_val_20_per_class.csv",
    "hmdb51":        "hmdb51_val_20_per_class.csv",
    "epic_kitchens": "epic_kitchens_val_20_per_class.csv",
    "ucf101":        "ucf101_val_20_per_class.csv",
}

MAX_VIDEOS_PER_CLASS = 5
N_FRAMES             = 32   # more frames → finer temporal resolution
FRAME_SIZE           = (112, 112)


def compute_flow_stats(video_path: str, n_frames: int = 32) -> dict | None:
    """Return temporal flow statistics for a single video.

    Returns a dict with:
      mean_flow     – mean inter-frame flow magnitude
      temporal_std  – std of inter-frame magnitudes (within-clip burstiness)
      temporal_cv   – temporal_std / mean_flow  (normalised burstiness)
      peak_ratio    – max / mean  (peak prominence)
    Returns None if the video cannot be decoded.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return None

    indices = np.linspace(0, max(0, total - 1), n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, FRAME_SIZE)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
    cap.release()

    if len(frames) < 2:
        return None

    magnitudes = []
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i + 1], None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
        magnitudes.append(mag)

    if not magnitudes:
        return None

    mags   = np.array(magnitudes)
    mean   = float(mags.mean())
    t_std  = float(mags.std())
    t_cv   = float(t_std / mean) if mean > 1e-6 else 0.0
    peak_r = float(mags.max() / mean) if mean > 1e-6 else 1.0

    return {
        "mean_flow":    mean,
        "temporal_std": t_std,
        "temporal_cv":  t_cv,
        "peak_ratio":   peak_r,
    }


def correlate(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


corr_rows = []

for dataset, manifest_file in DATASETS.items():
    manifest_path = MANIFESTS / manifest_file
    taxonomy_path = TAXONOMY / f"{dataset}_class_taxonomy.csv"

    if not manifest_path.exists():
        print(f"  {dataset}: manifest not found – skipping")
        continue
    if not taxonomy_path.exists():
        print(f"  {dataset}: taxonomy not found – skipping")
        continue

    manifest = pd.read_csv(manifest_path)
    if "split" in manifest.columns:
        manifest = manifest[manifest["split"].isin(["val", "validation", "test"])].copy()
    if "exists" in manifest.columns:
        manifest = manifest[manifest["exists"] == True].copy()

    taxonomy = pd.read_csv(taxonomy_path)
    print(f"\n{'='*60}")
    print(f"  {dataset}: {len(manifest)} videos, "
          f"{manifest['label_id'].nunique()} classes")

    flow_rows = []
    classes   = sorted(manifest["label_id"].unique())

    for label_id in classes:
        paths  = manifest[manifest["label_id"] == label_id]["video_path"].tolist()
        sample = paths[:MAX_VIDEOS_PER_CLASS]

        video_stats = []
        for vp in sample:
            s = compute_flow_stats(str(vp), N_FRAMES)
            if s is not None:
                video_stats.append(s)

        if not video_stats:
            continue

        flow_rows.append({
            "dataset":      dataset,
            "label_id":     label_id,
            "n_videos":     len(video_stats),
            "mean_flow":    np.mean([s["mean_flow"]    for s in video_stats]),
            "temporal_std": np.mean([s["temporal_std"] for s in video_stats]),
            "temporal_cv":  np.mean([s["temporal_cv"]  for s in video_stats]),
            "peak_ratio":   np.mean([s["peak_ratio"]   for s in video_stats]),
        })

    if not flow_rows:
        print("    → no flow computed")
        continue

    flow_df = pd.DataFrame(flow_rows)
    out_csv = OUT / f"{dataset}_flow_stats_v2.csv"
    flow_df.to_csv(out_csv, index=False)
    print(f"    → saved {out_csv.name} ({len(flow_df)} classes)")

    merged = flow_df.merge(
        taxonomy[["label_id", "mean_abs_drop", "mean_rel_drop"]],
        on="label_id", how="inner"
    )
    if len(merged) < 5:
        print(f"    → insufficient merged data ({len(merged)} classes)")
        continue

    n = len(merged)
    y = merged["mean_abs_drop"].values

    r_mean,  p_mean  = correlate(merged["mean_flow"].values,    y)
    r_tstd,  p_tstd  = correlate(merged["temporal_std"].values, y)
    r_tcv,   p_tcv   = correlate(merged["temporal_cv"].values,  y)
    r_peak,  p_peak  = correlate(merged["peak_ratio"].values,   y)

    print(f"    n={n:3d} classes  (aliasing = mean_abs_drop stride=1→16)")
    print(f"    mean_flow     r={r_mean:+.3f}  p={p_mean:.4f}"
          f"  {'✅' if p_mean<0.05 else '  '}")
    print(f"    temporal_std  r={r_tstd:+.3f}  p={p_tstd:.4f}"
          f"  {'✅' if p_tstd<0.05 else '  '}")
    print(f"    temporal_cv   r={r_tcv:+.3f}  p={p_tcv:.4f}"
          f"  {'✅' if p_tcv<0.05 else '  '}")
    print(f"    peak_ratio    r={r_peak:+.3f}  p={p_peak:.4f}"
          f"  {'✅' if p_peak<0.05 else '  '}")

    corr_rows.append({
        "dataset":          dataset,
        "n_classes":        n,
        "r_mean_flow":      r_mean,  "p_mean_flow":  p_mean,
        "r_temporal_std":   r_tstd,  "p_temporal_std": p_tstd,
        "r_temporal_cv":    r_tcv,   "p_temporal_cv":  p_tcv,
        "r_peak_ratio":     r_peak,  "p_peak_ratio":   p_peak,
        "sig_mean":         p_mean  < 0.05,
        "sig_tstd":         p_tstd  < 0.05,
        "sig_tcv":          p_tcv   < 0.05,
        "sig_peak":         p_peak  < 0.05,
    })

if corr_rows:
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT / "flow_aliasing_correlation_v2.csv", index=False)

    print("\n" + "=" * 70)
    print("E3-v2 — Temporal burstiness vs aliasing  (Pearson r, per-class)")
    print("Hypothesis: higher burstiness → more aliasing  (positive r)")
    print("=" * 70)
    hdr = f"{'Dataset':<15} {'n':>4}  {'mean_flow':>9}  {'temp_std':>9}  "
    hdr += f"{'temp_cv':>9}  {'peak_r':>9}"
    print(hdr)
    print("-" * 70)
    for _, row in corr_df.iterrows():
        def fmt(r, sig): return f"{r:+.3f}{'*' if sig else ' '}"
        print(f"{row['dataset']:<15} {int(row['n_classes']):>4}  "
              f"{fmt(row['r_mean_flow'],  row['sig_mean']):>10}  "
              f"{fmt(row['r_temporal_std'],row['sig_tstd']):>10}  "
              f"{fmt(row['r_temporal_cv'], row['sig_tcv']):>10}  "
              f"{fmt(row['r_peak_ratio'],  row['sig_peak']):>10}")
    print("  * p < 0.05")
else:
    print("\nNo correlation data – check video paths.")
