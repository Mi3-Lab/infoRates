#!/usr/bin/env python3
"""Knapsack Allocator using Confidence as Difficulty Estimator — ACCV 2026.

Key insight from 11_spectral_router.py: cinematic features (FDE, DCT, HOG)
have r≈0.09 with optimal budget on SSV2. But the model's own confidence at
a cheap budget IS a semantic signal.

This script replaces the spectral regressor with a confidence-based value
function:
  V(video, budget) ≈ confidence@budget_low  (proxy for expected accuracy)

Then the knapsack uses this to allocate frames across a batch.

Comparison table produced:
  - Fixed-4f, Fixed-8f, Fixed-16f, Fixed-32f
  - Cascade 4f→16f (confidence threshold)
  - Knapsack (confidence-based V function)
  - Oracle knapsack (ground-truth V)

Runs on any samples CSV with confidence column.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BUDGETS = [4, 8, 16, 32]


# ─── Value function ───────────────────────────────────────────────────────────

def _build_pivot(df: pd.DataFrame, probe_budget: int):
    """Build per-video pivot dicts for confidence and accuracy."""
    pivot_conf: dict[str, dict[int, float]] = {}
    pivot_acc:  dict[str, dict[int, float]] = {}
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        b   = int(row["budget"])
        pivot_conf.setdefault(vid, {})[b] = float(row["confidence"])
        pivot_acc.setdefault(vid,  {})[b] = float(row["correct_top1"])
    valid = [v for v in pivot_conf
             if probe_budget in pivot_conf[v]
             and all(b in pivot_acc[v] for b in BUDGETS)]
    return valid, pivot_conf, pivot_acc


def build_value_matrix_learned(
    df: pd.DataFrame,
    probe_budget: int = 4,
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Learn V[i,b] = accuracy@budget from confidence@probe via MLP (80/20 split).

    Features: [conf@probe, conf@probe^2, log(conf@probe + 1e-6)]
    Target:   accuracy at each budget (4-dim regression, one MLP per budget)

    Returns (V_pred_test, test_video_ids, V_true_test).
    """
    df = df[~df["skipped"].astype(bool)].copy()
    df["video_id"] = df["video_id"].astype(str)

    valid, pivot_conf, pivot_acc = _build_pivot(df, probe_budget)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(valid))
    n_train = int(len(valid) * train_frac)
    train_ids = [valid[i] for i in idx[:n_train]]
    test_ids  = [valid[i] for i in idx[n_train:]]

    def featurize(vids):
        X = []
        for v in vids:
            c = pivot_conf[v][probe_budget]
            X.append([c, c ** 2, np.log(c + 1e-6)])
        return np.array(X, dtype=np.float32)

    X_train = featurize(train_ids)
    X_test  = featurize(test_ids)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    n_test = len(test_ids)
    V_true = np.zeros((n_test, len(BUDGETS)))
    V_pred = np.zeros((n_test, len(BUDGETS)))

    for j, b in enumerate(BUDGETS):
        y_train = np.array([pivot_acc[v][b] for v in train_ids], dtype=np.float32)
        mlp = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=500,
            random_state=seed,
        )
        mlp.fit(X_train, y_train)
        V_pred[:, j] = mlp.predict(X_test)
        V_true[:, j] = [pivot_acc[v][b] for v in test_ids]

    return V_pred, test_ids, V_true


def build_value_matrix_proxy(
    df: pd.DataFrame,
    probe_budget: int = 4,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Simple confidence-scaling proxy (baseline value function)."""
    df = df[~df["skipped"].astype(bool)].copy()
    df["video_id"] = df["video_id"].astype(str)

    valid, pivot_conf, pivot_acc = _build_pivot(df, probe_budget)
    n = len(valid)
    V_true = np.zeros((n, len(BUDGETS)))
    V_pred = np.zeros((n, len(BUDGETS)))

    for i, vid in enumerate(valid):
        c = pivot_conf[vid][probe_budget]
        for j, b in enumerate(BUDGETS):
            V_true[i, j] = pivot_acc[vid][b]
            V_pred[i, j] = c * np.sqrt(b / max(BUDGETS))

    return V_pred, valid, V_true


# ─── Knapsack ─────────────────────────────────────────────────────────────────

def knapsack_allocate(V: np.ndarray, total_budget: int) -> np.ndarray:
    """Greedy knapsack: given value matrix V[i,b] maximise sum(V) s.t. sum(frames)<=B.

    Start all videos at BUDGETS[0]; greedily upgrade based on marginal V gain
    per extra frame.
    """
    n = V.shape[0]
    budget_idx = np.zeros(n, dtype=int)
    budget_arr = np.array(BUDGETS)
    used = n * BUDGETS[0]
    remaining = total_budget - used

    while remaining > 0:
        best_gain = -np.inf
        best_i    = -1
        for i in range(n):
            ci = budget_idx[i]
            if ci >= len(BUDGETS) - 1:
                continue
            ni = ci + 1
            delta_f = budget_arr[ni] - budget_arr[ci]
            if delta_f > remaining:
                continue
            gain = (V[i, ni] - V[i, ci]) / delta_f
            if gain > best_gain:
                best_gain = gain
                best_i    = i
        if best_i == -1:
            break
        nxt = budget_idx[best_i] + 1
        remaining -= budget_arr[nxt] - budget_arr[budget_idx[best_i]]
        budget_idx[best_i] = nxt

    return budget_arr[budget_idx]


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_all_methods(
    df: pd.DataFrame,
    avg_budget_targets: list[int] = [8, 10, 12, 14, 16],
    probe_budget: int = 4,
) -> pd.DataFrame:
    """Compare fixed baselines, cascade, knapsack (confidence), oracle knapsack."""
    df = df[~df["skipped"].astype(bool)].copy()
    df["video_id"]     = df["video_id"].astype(str)
    df["correct_top1"] = df["correct_top1"].astype(bool)
    df["confidence"]   = df["confidence"].astype(float)

    # Fixed baselines
    fixed = {}
    for b in BUDGETS:
        sub = df[df["budget"] == b]
        if len(sub):
            fixed[b] = float(sub["correct_top1"].mean())

    # Build value matrices — proxy (full set) and learned (test split)
    V_proxy,   ids_proxy,   V_true_proxy  = build_value_matrix_proxy(df, probe_budget)
    V_learned, ids_learned, V_true_learned = build_value_matrix_learned(df, probe_budget)
    n_proxy   = len(ids_proxy)
    n_learned = len(ids_learned)

    rows = []

    # Fixed baselines (full set)
    for b, acc in fixed.items():
        rows.append({"method": f"Fixed-{b}f", "avg_frames": b,
                     "accuracy": acc, "type": "fixed"})

    for avg_k in avg_budget_targets:
        nearest = min(BUDGETS, key=lambda b: abs(b - avg_k))

        # Proxy knapsack (simple scaling, full set)
        alloc_p  = knapsack_allocate(V_proxy, n_proxy * avg_k)
        accs_p   = [V_true_proxy[i, BUDGETS.index(int(alloc_p[i]))] for i in range(n_proxy)]
        rows.append({
            "method":    f"Knapsack-proxy@avg{avg_k}f",
            "avg_frames": float(alloc_p.mean()),
            "accuracy":   float(np.mean(accs_p)),
            "type":       "knapsack_proxy",
            "vs_fixed":   float(np.mean(accs_p)) - fixed.get(nearest, np.nan),
            "budget_dist": str({str(b): int((alloc_p == b).sum()) for b in BUDGETS}),
        })

        # Learned knapsack (MLP regressor, test split only)
        alloc_l  = knapsack_allocate(V_learned, n_learned * avg_k)
        accs_l   = [V_true_learned[i, BUDGETS.index(int(alloc_l[i]))] for i in range(n_learned)]
        rows.append({
            "method":    f"Knapsack-learned@avg{avg_k}f",
            "avg_frames": float(alloc_l.mean()),
            "accuracy":   float(np.mean(accs_l)),
            "type":       "knapsack_learned",
            "vs_fixed":   float(np.mean(accs_l)) - fixed.get(nearest, np.nan),
            "budget_dist": str({str(b): int((alloc_l == b).sum()) for b in BUDGETS}),
        })

        # Oracle knapsack (test split)
        alloc_o  = knapsack_allocate(V_true_learned, n_learned * avg_k)
        accs_o   = [V_true_learned[i, BUDGETS.index(int(alloc_o[i]))] for i in range(n_learned)]
        rows.append({
            "method":    f"Oracle-knapsack@avg{avg_k}f",
            "avg_frames": float(alloc_o.mean()),
            "accuracy":   float(np.mean(accs_o)),
            "type":       "oracle_knapsack",
            "vs_fixed":   float(np.mean(accs_o)) - fixed.get(nearest, np.nan),
            "budget_dist": str({str(b): int((alloc_o == b).sum()) for b in BUDGETS}),
        })

    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

EVAL_CONFIGS = [
    # SSV2
    ("r3d18_ssv2_full_e10_a100",         "somethingv2_*_samples.csv",  "R3D-18",      "SSV2"),
    ("mc3_18_ssv2_full_e10_a100",         "somethingv2_*_samples.csv",  "MC3-18",      "SSV2"),
    ("r2plus1d_18_ssv2_full_e10_a100",    "somethingv2_*_samples.csv",  "R2plus1D-18", "SSV2"),
    ("timesformer_ssv2_full_e10_h200",    "somethingv2_*_samples.csv",  "TimeSformer", "SSV2"),
    ("vivit_ssv2_full_e10_h200",          "somethingv2_*_samples.csv",  "ViViT",       "SSV2"),
    ("slowfast_r50_ssv2_full_e10_a100",   "somethingv2_*_samples.csv",  "SlowFast",    "SSV2"),
    ("videomae_ssv2_full_e5_h200",        "somethingv2_*_samples.csv",  "VideoMAE",    "SSV2"),
    # UCF101
    ("r2plus1d_18_ucf101_full_e10_a100",  "ucf101_*_samples.csv",       "R2plus1D-18", "UCF101"),
    ("slowfast_r50_ucf101_full_e10_a100", "ucf101_*_samples.csv",       "SlowFast",    "UCF101"),
    ("videomae_ucf101_full_e10_h200",     "ucf101_*_samples.csv",       "VideoMAE",    "UCF101"),
    # HMDB51
    ("r2plus1d_18_hmdb51_full_e10_a100",  "hmdb51_*_samples.csv",       "R2plus1D-18", "HMDB51"),
    ("slowfast_r50_hmdb51_full_e10_a100", "hmdb51_*_samples.csv",       "SlowFast",    "HMDB51"),
    ("videomae_hmdb51_full_e10_h200",     "hmdb51_*_samples.csv",       "VideoMAE",    "HMDB51"),
    # Diving48
    ("r2plus1d_18_diving48_full_e10_a100","diving48_*_samples.csv",     "R2plus1D-18", "Diving48"),
    ("slowfast_r50_diving48_full_e10_a100","diving48_*_samples.csv",    "SlowFast",    "Diving48"),
    ("videomae_diving48_full_e10_h200",   "diving48_*_samples.csv",     "VideoMAE",    "Diving48"),
    # EPIC-Kitchens (added when evals complete)
    ("r2plus1d_18_epic_kitchens_full_e10_a100","epic_kitchens_*_samples.csv","R2plus1D-18","EPIC"),
    ("videomae_epic_kitchens_full_e10_h200",   "epic_kitchens_*_samples.csv","VideoMAE",   "EPIC"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-base",  default="evaluations/accv2026/fixed_budget")
    parser.add_argument("--output-dir", default="evaluations/accv2026/knapsack_confidence")
    parser.add_argument("--datasets",   nargs="+",
                        default=["ssv2", "ucf101", "hmdb51", "diving48"])
    parser.add_argument("--avg-budgets", nargs="+", type=int,
                        default=[8, 10, 12, 14, 16])
    args = parser.parse_args()

    eval_base = ROOT / args.eval_base
    out_base  = ROOT / args.output_dir
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("KNAPSACK + CONFIDENCE ESTIMATOR — ACCV 2026")
    print("=" * 65)

    all_rows = []

    for eval_dir, pattern, model, dataset in EVAL_CONFIGS:
        if not any(ds in eval_dir for ds in args.datasets):
            continue
        matches = list((eval_base / eval_dir).glob(pattern))
        if not matches:
            continue

        df = pd.read_csv(matches[0])
        results = evaluate_all_methods(df, args.avg_budgets)
        results["model"]    = model
        results["dataset"]  = dataset
        results["eval_dir"] = eval_dir

        out_dir = out_base / eval_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_dir / "knapsack_confidence_results.csv", index=False)

        # Print summary for this model/dataset
        print(f"\n  {model} [{dataset}]")
        for _, r in results.iterrows():
            if r["type"] == "fixed":
                print(f"    Fixed-{int(r['avg_frames'])}f:              "
                      f"{r['accuracy']*100:.2f}%")
            else:
                delta = r.get("vs_fixed", np.nan)
                sign  = "+" if not np.isnan(delta) and delta >= 0 else ""
                print(f"    {r['method']:35s}: "
                      f"{r['accuracy']*100:.2f}% @ {r['avg_frames']:.1f}f avg  "
                      f"delta={sign}{delta*100:.2f}%")

        all_rows.append(results)

    if not all_rows:
        print("No results.")
        return

    global_df = pd.concat(all_rows, ignore_index=True)
    global_df.to_csv(out_base / "knapsack_global_results.csv", index=False)

    # Print best gains
    print("\n" + "=" * 65)
    print("KNAPSACK-LEARNED GAINS (per model/dataset, avg_budget=16):")
    print("=" * 65)
    filt = global_df[
        (global_df["type"] == "knapsack_learned") &
        (global_df["method"].str.contains("avg16"))
    ].copy()
    if not filt.empty:
        filt = filt.sort_values("vs_fixed", ascending=False)
        print(filt[["model", "dataset", "accuracy", "avg_frames", "vs_fixed"]
              ].to_string(index=False))

    print("\n" + "=" * 65)
    print("KNAPSACK-PROXY GAINS (per model/dataset, avg_budget=16):")
    print("=" * 65)
    filt_p = global_df[
        (global_df["type"] == "knapsack_proxy") &
        (global_df["method"].str.contains("avg16"))
    ].copy()
    if not filt_p.empty:
        filt_p = filt_p.sort_values("vs_fixed", ascending=False)
        print(filt_p[["model", "dataset", "accuracy", "avg_frames", "vs_fixed"]
              ].to_string(index=False))

    print("\n" + "=" * 65)
    print("ORACLE UPPER BOUND (per model/dataset, avg_budget=16):")
    print("=" * 65)
    filt_or = global_df[
        (global_df["type"] == "oracle_knapsack") &
        (global_df["method"].str.contains("avg16"))
    ].copy()
    if not filt_or.empty:
        filt_or = filt_or.sort_values("vs_fixed", ascending=False)
        print(filt_or[["model", "dataset", "accuracy", "avg_frames", "vs_fixed"]
               ].to_string(index=False))

    print(f"\nAll results saved to {out_base}/")


if __name__ == "__main__":
    main()
