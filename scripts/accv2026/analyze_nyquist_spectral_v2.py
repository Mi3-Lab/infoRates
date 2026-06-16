#!/usr/bin/env python3
"""Direct Nyquist validation v2: resolution as an additional axis.

v1 (analyze_nyquist_spectral.py) correlated one flow-cutoff-frequency point
per dataset (n=7) against TDS (n=7) -- underpowered (p=0.34).

v2 uses resolution as a second axis: for each (dataset, resolution) pair we
have (a) a flow-based spectral cutoff frequency computed on frames resized
to that resolution, and (b) an empirical stride-sensitivity computed from
the EXISTING coverage x stride x resolution sweep
(coverage_stride_sweep/{model}_{dataset}_trainres{R}/sweep_summary.csv,
8 models x 7 datasets x 5 resolutions, no new training/inference needed for
the accuracy side). This gives up to 7 datasets x 5 resolutions = 35 points
instead of 7, with real statistical power.

Hypothesis: if resolution downsampling acts like a spatial low-pass filter,
the temporal signal extracted from low-res frames should look smoother
(lower estimated cutoff frequency) and the model's empirical
stride-sensitivity at that resolution should track it.

Run from /scratch/wesleyferreiramaia/infoRates (manifests use relative paths).
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from scipy.signal import periodogram
from scipy.stats import spearmanr

ROOT = Path("/data/wesleyferreiramaia/infoRates")
MANIFESTS = ROOT / "evaluations/accv2026/manifests"
SWEEP_ROOT = ROOT / "evaluations/accv2026/coverage_stride_sweep"
OUT = ROOT / "evaluations/accv2026/e3_spectral"

DATASETS = {
    "ucf101":       "ucf101_val_20_per_class.csv",
    "ssv2":         "somethingv2_val_20_per_class.csv",
    "hmdb51":       "hmdb51_val_20_per_class.csv",
    "diving48":     "diving48_val_20_per_class.csv",
    "autsl":        "autsl_val_20_per_class.csv",
    "driveact":     "driveact_val_20_per_class.csv",
    "epic_kitchens":"epic_kitchens_val_20_per_class.csv",
}
MODEL_KEYS = ["r3d_18", "mc3_18", "r2plus1d_18", "slowfast_r50",
              "timesformer", "vivit", "videomae", "videomamba"]
RESOLUTIONS = [48, 96, 112, 160, 224]

N_FRAMES_CONSEC = 64
MAX_CLASSES = 12
N_VIDEOS_PER_CLASS = 2


def flow_series(video_path: str, frame_size: tuple, n_frames: int = N_FRAMES_CONSEC) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 10:
        cap.release()
        return None
    n = min(n_frames, total)
    frames = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) < 10:
        return None
    mags = []
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()))
    return np.array(mags)


def spectral_cutoff(signal: np.ndarray, energy_frac: float = 0.90) -> float:
    sig = signal - signal.mean()
    if np.allclose(sig, 0):
        return 0.0
    freqs, psd = periodogram(sig, fs=1.0)
    if psd.sum() <= 0:
        return 0.0
    cum = np.cumsum(psd) / psd.sum()
    idx = min(int(np.searchsorted(cum, energy_frac)), len(freqs) - 1)
    return float(freqs[idx])


def empirical_stride_sensitivity(model_pool: list[str], dataset: str, res: int) -> float | None:
    """Mean accuracy drop stride=1->16 at coverage=100, averaged over available models."""
    drops = []
    for m in model_pool:
        f = SWEEP_ROOT / f"{m}_{dataset}_trainres{res}" / "sweep_summary.csv"
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df["acc"] = df["top1"] * 100
        sub = df[df.coverage == 100]
        s1 = sub[sub.stride == 1]["acc"]
        s16 = sub[sub.stride == 16]["acc"]
        if s1.empty or s16.empty or s1.values[0] < 5:
            continue
        drops.append(s1.values[0] - s16.values[0])
    return float(np.mean(drops)) if drops else None


def main():
    rng = np.random.default_rng(0)
    rows = []

    for ds, manifest_name in DATASETS.items():
        df_m = pd.read_csv(MANIFESTS / manifest_name)
        if "exists" in df_m.columns:
            df_m = df_m[df_m["exists"] == True]
        classes = sorted(df_m.label_id.unique())
        if len(classes) > MAX_CLASSES:
            classes = list(rng.choice(classes, size=MAX_CLASSES, replace=False))

        sample_rows = []
        for c in classes:
            sub = df_m[df_m.label_id == c]
            sample_rows.append(sub.sample(n=min(N_VIDEOS_PER_CLASS, len(sub)), random_state=0))
        sample_df = pd.concat(sample_rows)
        video_paths = sample_df["video_path"].tolist()

        for res in RESOLUTIONS:
            cutoffs = []
            for vp in video_paths:
                sig = flow_series(vp, frame_size=(res, res))
                if sig is None or len(sig) < 10:
                    continue
                fc = spectral_cutoff(sig)
                if fc > 0:
                    cutoffs.append(fc)
            if not cutoffs:
                print(f"[{ds}@{res}px] no usable videos")
                continue
            mean_fc = float(np.mean(cutoffs))

            tds_res = empirical_stride_sensitivity(MODEL_KEYS, ds, res)
            if tds_res is None:
                print(f"[{ds}@{res}px] f_c={mean_fc:.4f} but no sweep data, skipping")
                continue

            rows.append({
                "dataset": ds, "resolution": res, "n_videos": len(cutoffs),
                "cutoff_freq": round(mean_fc, 4),
                "stride_sensitivity": round(tds_res, 2),
            })
            print(f"[{ds}@{res}px] n={len(cutoffs)}  f_c={mean_fc:.4f} cyc/frame  "
                  f"stride_sensitivity={tds_res:.2f}pp")

    result = pd.DataFrame(rows)
    print()
    print(f"Total (dataset, resolution) points: {len(result)}")
    print(result.to_string(index=False))

    if len(result) >= 8:
        rho, p = spearmanr(result.cutoff_freq, result.stride_sensitivity)
        print()
        print(f"POOLED Spearman(cutoff_freq, stride_sensitivity) across all "
              f"(dataset,resolution) points: rho={rho:+.3f}  p={p:.4f}  n={len(result)}")

        print()
        print("Per-dataset within-resolution trend (does f_c track sensitivity as res varies?):")
        for ds in result.dataset.unique():
            sub = result[result.dataset == ds]
            if len(sub) >= 3:
                r, pv = spearmanr(sub.cutoff_freq, sub.stride_sensitivity)
                print(f"  {ds:15s} rho={r:+.3f}  p={pv:.4f}  n={len(sub)}")

        print()
        print("Per-resolution across-dataset trend (does f_c rank datasets consistently per res?):")
        for res in RESOLUTIONS:
            sub = result[result.resolution == res]
            if len(sub) >= 4:
                r, pv = spearmanr(sub.cutoff_freq, sub.stride_sensitivity)
                print(f"  {res}px  rho={r:+.3f}  p={pv:.4f}  n={len(sub)}")

    out_csv = OUT / "nyquist_resolution_validation.csv"
    result.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
