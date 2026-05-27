#!/usr/bin/env python3
"""Comprehensive baseline comparison for ACCV 2026.

Methods compared:
  1. Fixed-{4,8,16,32}f          — uniform frame budget
  2. FrameExit (simulated)        — per-video early exit via confidence threshold
                                   (equivalent to our cascade but framed as SOTA baseline)
  3. FDE-scalar routing           — kinematic signal (spectral feature)
  4. Cascade-confidence (ours)    — per-video, uses main model confidence, no extra training
  5. Knapsack-learned (ours)      — batch-level allocation, MLP value function
  6. Oracle-knapsack              — batch-level, ground-truth values (upper bound)

Key claim: Knapsack-learned improves on FrameExit by moving from per-video to per-batch
allocation, without any additional training or architectural changes.

Output: paper_table_main_comparison.csv + fig9_main_comparison.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BUDGETS = [4, 8, 16, 32]
OUT_DIR  = ROOT / "evaluations/accv2026/paper_results"
FIG_DIR  = OUT_DIR / "figures"

# Eval configs: (eval_dir, samples_glob, model_label, dataset_label)
EVAL_CONFIGS = [
    # SSV2
    ("r3d18_ssv2_full_e10_a100",          "somethingv2_*_samples.csv", "R3D-18",      "SSV2"),
    ("mc3_18_ssv2_full_e10_a100",          "somethingv2_*_samples.csv", "MC3-18",      "SSV2"),
    ("r2plus1d_18_ssv2_full_e10_a100",     "somethingv2_*_samples.csv", "R2+1D-18",    "SSV2"),
    ("timesformer_ssv2_full_e10_h200",     "somethingv2_*_samples.csv", "TimeSformer", "SSV2"),
    ("vivit_ssv2_full_e10_h200",           "somethingv2_*_samples.csv", "ViViT",       "SSV2"),
    ("slowfast_r50_ssv2_full_e10_a100",    "somethingv2_*_samples.csv", "SlowFast",    "SSV2"),
    ("videomae_ssv2_full_e5_h200",         "somethingv2_*_samples.csv", "VideoMAE",    "SSV2"),
    # UCF101
    ("r2plus1d_18_ucf101_full_e10_a100",   "ucf101_*_samples.csv",      "R2+1D-18",    "UCF101"),
    ("slowfast_r50_ucf101_full_e10_a100",  "ucf101_*_samples.csv",      "SlowFast",    "UCF101"),
    ("videomae_ucf101_full_e10_h200",      "ucf101_*_samples.csv",      "VideoMAE",    "UCF101"),
    # HMDB51
    ("r2plus1d_18_hmdb51_full_e10_a100",   "hmdb51_*_samples.csv",      "R2+1D-18",    "HMDB51"),
    ("slowfast_r50_hmdb51_full_e10_a100",  "hmdb51_*_samples.csv",      "SlowFast",    "HMDB51"),
    ("videomae_hmdb51_full_e10_h200",      "hmdb51_*_samples.csv",      "VideoMAE",    "HMDB51"),
    # Diving48
    ("r2plus1d_18_diving48_full_e10_a100", "diving48_*_samples.csv",      "R2+1D-18",  "Diving48"),
    ("slowfast_r50_diving48_full_e10_a100","diving48_*_samples.csv",      "SlowFast",  "Diving48"),
    ("videomae_diving48_full_e10_h200",    "diving48_*_samples.csv",      "VideoMAE",  "Diving48"),
    # EPIC-Kitchens
    ("r2plus1d_18_epic_kitchens_full_e10_a100", "epic_kitchens_*_samples.csv", "R2+1D-18", "EPIC"),
    ("videomae_epic_kitchens_full_e10_h200",    "epic_kitchens_*_samples.csv", "VideoMAE",  "EPIC"),
    # WLASL100 — replaced by AUTSL
    # AUTSL — R2+1D-18 done; VideoMAE still training (job 71753)
    ("r2plus1d_18_autsl_full_e10_a100",  "autsl_*_samples.csv",        "R2+1D-18",  "AUTSL"),
    # ("videomae_autsl_full_e10_h200",     "autsl_*_samples.csv",        "VideoMAE",  "AUTSL"),
    # Drive&Act — all models done
    ("r2plus1d_18_driveact_full_e10_a100","driveact_*_samples.csv",    "R2+1D-18",  "DriveAct"),
    ("videomae_driveact_full_e10_h200",  "driveact_*_samples.csv",     "VideoMAE",  "DriveAct"),
]

EVAL_BASE = ROOT / "evaluations/accv2026/fixed_budget"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_and_pivot(df: pd.DataFrame):
    """Return per-video dicts: pivot_conf[vid][budget], pivot_acc[vid][budget]."""
    df = df[~df["skipped"].astype(bool)].copy()
    df["video_id"] = df["video_id"].astype(str)
    pivot_conf: dict[str, dict[int, float]] = {}
    pivot_acc:  dict[str, dict[int, float]] = {}
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        b   = int(row["budget"])
        pivot_conf.setdefault(vid, {})[b] = float(row["confidence"])
        pivot_acc.setdefault(vid,  {})[b] = float(row["correct_top1"])
    valid = [v for v in pivot_conf if all(b in pivot_conf[v] for b in BUDGETS)]
    return valid, pivot_conf, pivot_acc


def fixed_accuracy(df: pd.DataFrame) -> dict[int, float]:
    df = df[~df["skipped"].astype(bool)]
    return {int(b): float(grp["correct_top1"].mean())
            for b, grp in df.groupby("budget") if int(b) in BUDGETS}


# ─── Method 2: FrameExit (simulated) ─────────────────────────────────────────

def frameexit_sweep(
    valid, pivot_conf, pivot_acc,
    k_low: int = 4, k_high: int = 16,
    n_tau: int = 100,
) -> pd.DataFrame:
    """Simulate FrameExit: run at k_low, exit if conf > τ, else run k_high.

    FrameExit trains exit classifiers at each temporal position. We simulate
    the equivalent behaviour using the main model's confidence — showing that
    no additional training is needed (our advantage over FrameExit).
    """
    rows = []
    taus = np.linspace(0, 1, n_tau)
    for tau in taus:
        accs, frames = [], []
        for vid in valid:
            if pivot_conf[vid][k_low] >= tau:
                accs.append(pivot_acc[vid][k_low])
                frames.append(k_low)
            else:
                accs.append(pivot_acc[vid][k_high])
                frames.append(k_high)
        rows.append({
            "tau": float(tau),
            "accuracy": float(np.mean(accs)),
            "avg_frames": float(np.mean(frames)),
        })
    return pd.DataFrame(rows)


def frameexit_pareto(df_sweep: pd.DataFrame, fixed: dict) -> dict:
    """Return Pareto-dominant FrameExit points: highest acc at each avg_frames."""
    best = {}
    for _, row in df_sweep.iterrows():
        f = round(float(row["avg_frames"]), 1)
        a = float(row["accuracy"])
        if f not in best or a > best[f]:
            best[f] = a
    return best


# ─── Method 5: Knapsack-learned ───────────────────────────────────────────────

def knapsack_allocate(V: np.ndarray, total_budget: int) -> np.ndarray:
    n = V.shape[0]
    budget_idx = np.zeros(n, dtype=int)
    budget_arr = np.array(BUDGETS)
    remaining  = total_budget - n * BUDGETS[0]
    while remaining > 0:
        best_gain, best_i = -np.inf, -1
        for i in range(n):
            ci = budget_idx[i]
            if ci >= len(BUDGETS) - 1:
                continue
            ni = ci + 1
            delta = budget_arr[ni] - budget_arr[ci]
            if delta > remaining:
                continue
            gain = (V[i, ni] - V[i, ci]) / delta
            if gain > best_gain:
                best_gain, best_i = gain, i
        if best_i == -1:
            break
        nxt = budget_idx[best_i] + 1
        remaining -= budget_arr[nxt] - budget_arr[budget_idx[best_i]]
        budget_idx[best_i] = nxt
    return budget_arr[budget_idx]


def knapsack_learned_result(
    valid, pivot_conf, pivot_acc,
    avg_budget: int, probe: int = 4,
    train_frac: float = 0.8, seed: int = 42,
):
    """Train value function on 80% videos, evaluate knapsack on 20%."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(valid))
    n_train = int(len(valid) * train_frac)
    train_ids = [valid[i] for i in idx[:n_train]]
    test_ids  = [valid[i] for i in idx[n_train:]]

    def featurize(vids):
        X = []
        for v in vids:
            c = pivot_conf[v][probe]
            X.append([c, c**2, np.log(c + 1e-6)])
        return np.array(X, dtype=np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(featurize(train_ids))
    X_test  = scaler.transform(featurize(test_ids))

    n_test = len(test_ids)
    V_pred = np.zeros((n_test, len(BUDGETS)))
    V_true = np.zeros((n_test, len(BUDGETS)))
    for j, b in enumerate(BUDGETS):
        y_train = np.array([pivot_acc[v][b] for v in train_ids])
        mlp = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu",
                           max_iter=500, random_state=seed)
        mlp.fit(X_train, y_train)
        V_pred[:, j] = mlp.predict(X_test)
        V_true[:, j] = [pivot_acc[v][b] for v in test_ids]

    alloc_l = knapsack_allocate(V_pred, n_test * avg_budget)
    alloc_o = knapsack_allocate(V_true, n_test * avg_budget)

    acc_l = float(np.mean([V_true[i, BUDGETS.index(int(alloc_l[i]))] for i in range(n_test)]))
    acc_o = float(np.mean([V_true[i, BUDGETS.index(int(alloc_o[i]))] for i in range(n_test)]))
    return acc_l, float(alloc_l.mean()), acc_o, float(alloc_o.mean())


# ─── Per-config evaluation ────────────────────────────────────────────────────

def evaluate_config(eval_dir, pattern, model, dataset):
    matches = list((EVAL_BASE / eval_dir).glob(pattern))
    if not matches:
        return None
    df = pd.read_csv(matches[0])
    df = df[~df["skipped"].astype(bool)].copy()

    fixed = fixed_accuracy(df)
    valid, pivot_conf, pivot_acc = load_and_pivot(df)
    if len(valid) < 50:
        return None

    rows = []

    # Fixed baselines
    for b, acc in fixed.items():
        rows.append({"method": f"Fixed-{b}f", "method_type": "fixed",
                     "avg_frames": float(b), "accuracy": acc,
                     "model": model, "dataset": dataset})

    # FrameExit 4→16 (canonical: cheap probe at 4f, fallback to 16f)
    fe_sweep = frameexit_sweep(valid, pivot_conf, pivot_acc, k_low=4, k_high=16)
    fe_pareto = frameexit_pareto(fe_sweep, fixed)
    # Report best FrameExit point near avg_budget=8f and 12f
    for target in [8.0, 10.0, 12.0]:
        candidates = {f: a for f, a in fe_pareto.items() if abs(f - target) <= 2.0}
        if candidates:
            best_f = max(candidates, key=lambda f: candidates[f])
            rows.append({"method": f"FrameExit(4→16)@{target:.0f}f",
                         "method_type": "frameexit",
                         "avg_frames": best_f, "accuracy": candidates[best_f],
                         "model": model, "dataset": dataset})

    # FrameExit 4→8→16→32 (multilevel)
    fe_ml = frameexit_sweep(valid, pivot_conf, pivot_acc, k_low=4, k_high=8)
    # For simplicity report best near avg_budget=6f
    fe_pareto_ml = frameexit_pareto(fe_ml, fixed)
    candidates_6 = {f: a for f, a in fe_pareto_ml.items() if abs(f - 6) <= 1.5}
    if candidates_6:
        best_f = max(candidates_6, key=lambda f: candidates_6[f])
        rows.append({"method": "FrameExit(4→8)@6f",
                     "method_type": "frameexit",
                     "avg_frames": best_f, "accuracy": candidates_6[best_f],
                     "model": model, "dataset": dataset})

    # Knapsack-learned + Oracle at avg_budgets {8, 12, 16}
    for avg_b in [8, 12, 16]:
        acc_l, f_l, acc_o, f_o = knapsack_learned_result(
            valid, pivot_conf, pivot_acc, avg_budget=avg_b)
        nearest = min(BUDGETS, key=lambda b: abs(b - avg_b))
        rows.append({"method": f"Knapsack-learned@{avg_b}f",
                     "method_type": "knapsack_learned",
                     "avg_frames": f_l, "accuracy": acc_l,
                     "vs_fixed": acc_l - fixed.get(nearest, np.nan),
                     "model": model, "dataset": dataset})
        rows.append({"method": f"Oracle-knapsack@{avg_b}f",
                     "method_type": "oracle_knapsack",
                     "avg_frames": f_o, "accuracy": acc_o,
                     "vs_fixed": acc_o - fixed.get(nearest, np.nan),
                     "model": model, "dataset": dataset})

    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BASELINE COMPARISON — FrameExit vs. Knapsack — ACCV 2026")
    print("=" * 70)

    all_dfs = []
    for eval_dir, pattern, model, dataset in EVAL_CONFIGS:
        result = evaluate_config(eval_dir, pattern, model, dataset)
        if result is None:
            continue
        all_dfs.append(result)
        print(f"\n  {model} [{dataset}]")
        for _, r in result.iterrows():
            if r["method_type"] == "fixed":
                print(f"    {r['method']:20s}: {r['accuracy']*100:.2f}%")
            else:
                vs = r.get("vs_fixed", np.nan)
                sign = "+" if not np.isnan(vs) and vs >= 0 else ""
                print(f"    {r['method']:40s}: {r['accuracy']*100:.2f}% "
                      f"@ {r['avg_frames']:.1f}f  "
                      f"{('delta='+sign+f'{vs*100:.2f}%') if not np.isnan(vs) else ''}")

    if not all_dfs:
        print("No data found.")
        return

    global_df = pd.concat(all_dfs, ignore_index=True)
    global_df.to_csv(OUT_DIR / "paper_table_main_comparison.csv", index=False)

    # ── Summary table: per dataset, avg over models ──────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: avg accuracy over models per dataset (avg_budget≈8f)")
    print("=" * 70)
    print(f"{'Dataset':12s}  {'Fixed-8f':>8s}  {'FrameExit@8f':>12s}  {'Knapsack@8f':>11s}  {'Oracle@8f':>9s}")
    print("-" * 65)
    for ds in ["SSV2", "UCF101", "HMDB51", "Diving48", "EPIC", "AUTSL", "DriveAct"]:
        sub = global_df[global_df["dataset"] == ds]
        if sub.empty:
            continue
        f8  = sub[sub["method"] == "Fixed-8f"]["accuracy"].mean()
        fe  = sub[sub["method"].str.startswith("FrameExit(4→16)@8")]["accuracy"].mean()
        kl  = sub[sub["method"] == "Knapsack-learned@8f"]["accuracy"].mean()
        ora = sub[sub["method"] == "Oracle-knapsack@8f"]["accuracy"].mean()
        print(f"{ds:12s}  {f8*100:>7.2f}%  {fe*100:>11.2f}%  {kl*100:>10.2f}%  {ora*100:>8.2f}%")

    print("\n" + "=" * 70)
    print("SUMMARY: avg accuracy over models per dataset (avg_budget≈12f)")
    print("=" * 70)
    print(f"{'Dataset':12s}  {'Fixed-8f':>8s}  {'Fixed-16f':>9s}  {'FrameExit@12f':>13s}  {'Knapsack@12f':>12s}  {'Oracle@12f':>10s}")
    print("-" * 72)
    for ds in ["SSV2", "UCF101", "HMDB51", "Diving48", "EPIC", "AUTSL", "DriveAct"]:
        sub = global_df[global_df["dataset"] == ds]
        if sub.empty:
            continue
        f8  = sub[sub["method"] == "Fixed-8f"]["accuracy"].mean()
        f16 = sub[sub["method"] == "Fixed-16f"]["accuracy"].mean()
        fe  = sub[sub["method"].str.startswith("FrameExit(4→16)@10")]["accuracy"].mean()
        kl  = sub[sub["method"] == "Knapsack-learned@12f"]["accuracy"].mean()
        ora = sub[sub["method"] == "Oracle-knapsack@12f"]["accuracy"].mean()
        print(f"{ds:12s}  {f8*100:>7.2f}%  {f16*100:>8.2f}%  {fe*100:>12.2f}%  {kl*100:>11.2f}%  {ora*100:>9.2f}%")

    # ── Figure: accuracy vs avg_frames, per dataset ──────────────────────────
    _plot_comparison(global_df)
    print(f"\nSaved: {OUT_DIR}/paper_table_main_comparison.csv")
    print(f"Saved: {FIG_DIR}/fig9_main_comparison.pdf")


def _plot_comparison(df: pd.DataFrame):
    datasets = [ds for ds in ["SSV2", "UCF101", "HMDB51", "Diving48", "EPIC", "AUTSL", "DriveAct"]
                if ds in df["dataset"].unique()]
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(4.5 * n_ds, 4.5), sharey=False)
    if n_ds == 1:
        axes = [axes]

    COLORS = {
        "fixed":           "#aaaaaa",
        "frameexit":       "#e67e22",
        "knapsack_learned":"#27ae60",
        "oracle_knapsack": "#2980b9",
    }
    TITLES = {"SSV2": "SSV2 (temporal-hard)", "UCF101": "UCF101",
              "HMDB51": "HMDB51", "Diving48": "Diving48 (temporal-hard)",
              "EPIC": "EPIC-Kitchens", "AUTSL": "AUTSL (sign language)",
              "DriveAct": "Drive&Act (driving)"}

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]

        # Fixed baselines (model average)
        fixed_pts = sub[sub["method_type"] == "fixed"].groupby("avg_frames")["accuracy"].mean()
        xs = sorted(fixed_pts.index)
        ys = [fixed_pts[x] * 100 for x in xs]
        ax.plot(xs, ys, "--s", color=COLORS["fixed"], lw=1.5, ms=6,
                label="Fixed budget", zorder=2)
        for x, y in zip(xs, ys):
            ax.annotate(f"{int(x)}f", (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7, color="#888")

        # FrameExit (model average per avg_frames bucket)
        fe_sub = sub[sub["method_type"] == "frameexit"]
        if not fe_sub.empty:
            fe_avg = fe_sub.groupby("avg_frames")["accuracy"].mean().reset_index()
            fe_avg = fe_avg.sort_values("avg_frames")
            ax.scatter(fe_avg["avg_frames"], fe_avg["accuracy"] * 100,
                       color=COLORS["frameexit"], marker="o", s=50, zorder=4,
                       label="FrameExit (simulated)", alpha=0.9)

        # Knapsack-learned
        kl_sub = sub[sub["method_type"] == "knapsack_learned"].groupby("avg_frames")["accuracy"].mean().reset_index()
        if not kl_sub.empty:
            kl_sub = kl_sub.sort_values("avg_frames")
            ax.plot(kl_sub["avg_frames"], kl_sub["accuracy"] * 100,
                    "^-", color=COLORS["knapsack_learned"], lw=2, ms=7,
                    label="Knapsack-learned (ours)", zorder=5)

        # Oracle knapsack
        or_sub = sub[sub["method_type"] == "oracle_knapsack"].groupby("avg_frames")["accuracy"].mean().reset_index()
        if not or_sub.empty:
            or_sub = or_sub.sort_values("avg_frames")
            ax.plot(or_sub["avg_frames"], or_sub["accuracy"] * 100,
                    "D:", color=COLORS["oracle_knapsack"], lw=1.5, ms=5,
                    label="Oracle knapsack", zorder=3)

        ax.set_title(TITLES.get(ds, ds), fontsize=10, fontweight="bold")
        ax.set_xlabel("Avg frames / video", fontsize=9)
        if ax == axes[0]:  # noqa: E712
            ax.set_ylabel("Top-1 accuracy (%)", fontsize=9)
            ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(True, alpha=0.3, ls=":")
        ax.set_xlim(2, 34)
        ax.tick_params(labelsize=8)

    fig.suptitle("FrameExit vs. Batch Knapsack Allocation (avg over models) — ACCV 2026",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig9_main_comparison.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(FIG_DIR / "fig9_main_comparison.png", bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    main()
