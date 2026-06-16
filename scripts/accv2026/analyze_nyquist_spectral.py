#!/usr/bin/env python3
"""Direct Nyquist validation: dominant temporal frequency vs. empirical TDS.

Reviewer weakness: the Nyquist-Shannon framing is conceptual; the paper never
measures actual action-frequency spectra or alias frequencies, only accuracy
degradation under subsampling.

This script computes a literal Nyquist test using EXISTING decoded video
frames (no retraining, no GPU):
  1. For a sample of videos per dataset, read up to 64 CONSECUTIVE frames at
     native frame order/rate (not resampled across the whole clip, unlike the
     existing E3 mean-flow-magnitude proxy).
  2. Compute inter-frame optical-flow magnitude -> a temporal signal sampled
     at 1 sample/original-frame, matching the units our coverage/stride sweep
     uses (stride=k means "keep every k-th original frame").
  3. Estimate the signal's spectral cutoff frequency f_c (cycles/frame, 90%
     cumulative power) via a periodogram.
  4. Nyquist implies aliasing-free reconstruction requires sampling at
     >= 2*f_c, i.e. max safe stride ~= 1 / (2*f_c) original frames.
  5. Correlate the theoretical max-safe-stride (and f_c) against the
     empirical TDS ranking across datasets (Spearman).

Run from /scratch/wesleyferreiramaia/infoRates (video paths in manifests are
relative to that directory).
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

# TDS (full 8-architecture pool), from main.tex Table 2 / dashboard compute_tds()
TDS = {
    "finegym": 55.92, "autsl": 53.02, "ssv2": 27.29, "driveact": 21.48,
    "diving48": 19.16, "hmdb51": 16.47, "epic_kitchens": 9.77, "ucf101": 4.86,
}

N_FRAMES_CONSEC = 64
MAX_CLASSES = 15
N_VIDEOS_PER_CLASS = 2
FRAME_SIZE = (112, 112)


def flow_series(video_path: str, n_frames: int = N_FRAMES_CONSEC) -> np.ndarray | None:
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
        frame = cv2.resize(frame, FRAME_SIZE)
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
    """Frequency (cycles/frame) below which energy_frac of total power lies."""
    sig = signal - signal.mean()
    if np.allclose(sig, 0):
        return 0.0
    freqs, psd = periodogram(sig, fs=1.0)  # fs=1 sample/original-frame
    if psd.sum() <= 0:
        return 0.0
    cum = np.cumsum(psd) / psd.sum()
    idx = min(int(np.searchsorted(cum, energy_frac)), len(freqs) - 1)
    return float(freqs[idx])


def main():
    rng = np.random.default_rng(0)
    rows = []
    for ds, manifest_name in DATASETS.items():
        df = pd.read_csv(MANIFESTS / manifest_name)
        df = df[df.get("exists", True) == True] if "exists" in df.columns else df
        classes = sorted(df.label_id.unique())
        if len(classes) > MAX_CLASSES:
            classes = list(rng.choice(classes, size=MAX_CLASSES, replace=False))

        cutoffs = []
        n_ok, n_tried = 0, 0
        for c in classes:
            sub = df[df.label_id == c]
            sample = sub.sample(n=min(N_VIDEOS_PER_CLASS, len(sub)), random_state=0)
            for _, row in sample.iterrows():
                n_tried += 1
                sig = flow_series(row["video_path"])
                if sig is None or len(sig) < 10:
                    continue
                fc = spectral_cutoff(sig)
                if fc > 0:
                    cutoffs.append(fc)
                    n_ok += 1

        if not cutoffs:
            print(f"[{ds}] no usable videos, skipping")
            continue

        mean_fc = float(np.mean(cutoffs))
        median_fc = float(np.median(cutoffs))
        safe_stride = 1.0 / (2.0 * mean_fc) if mean_fc > 0 else float("inf")
        rows.append({
            "dataset": ds, "n_videos": n_ok, "n_tried": n_tried,
            "cutoff_freq_mean": round(mean_fc, 4),
            "cutoff_freq_median": round(median_fc, 4),
            "nyquist_safe_stride": round(safe_stride, 2),
            "tds": TDS[ds],
        })
        print(f"[{ds}] n={n_ok}/{n_tried}  f_c(mean)={mean_fc:.4f} cyc/frame  "
              f"safe_stride~{safe_stride:.2f}  TDS={TDS[ds]}")

    result = pd.DataFrame(rows).sort_values("tds", ascending=False).reset_index(drop=True)
    print()
    print(result.to_string(index=False))

    if len(result) >= 4:
        rho_fc, p_fc = spearmanr(result.tds, result.cutoff_freq_mean)
        rho_ss, p_ss = spearmanr(result.tds, result.nyquist_safe_stride)
        print()
        print(f"Spearman(TDS, cutoff_freq):        rho={rho_fc:+.3f}  p={p_fc:.4f}")
        print(f"Spearman(TDS, nyquist_safe_stride): rho={rho_ss:+.3f}  p={p_ss:.4f}")
        print("(Nyquist framing predicts: higher TDS <-> higher cutoff freq <-> SMALLER safe stride)")

    out_csv = OUT / "nyquist_nominal_validation.csv"
    result.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
