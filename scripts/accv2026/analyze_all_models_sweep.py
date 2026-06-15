#!/usr/bin/env python3
"""Comparative analysis across all 8 models: CNN vs Transformer vs SSM.

Generates:
- ANOVA across models
- Effect size comparisons
- Sensitivity curves
- Aliasing signatures
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    sweep_dir = ROOT / "evaluations/accv2026/coverage_stride_resolution_sweep"
    global_csv = sweep_dir / "GLOBAL_sweep_summary_all_models.csv"

    if not global_csv.exists():
        print(f"[ERROR] Global summary not found: {global_csv}")
        print("Run sweep_all_models_coverage_stride_resolution.py first")
        sys.exit(1)

    df = pd.read_csv(global_csv)

    print("\n" + "="*90)
    print("COMPARATIVE ANALYSIS: CNN vs TRANSFORMER vs SSM")
    print("="*90)

    # Model types
    CNNS = ["r3d_18", "mc3_18", "r2plus1d_18", "slowfast_r50"]
    TRANSFORMERS = ["timesformer", "vivit", "videomae"]
    SSMS = ["videomamba"]

    # 1. Overall stats by architecture type
    print("\n1️⃣  ARCHITECTURE COMPARISON")
    print("-"*90)

    for arch_name, models in [("CNNs", CNNS), ("Transformers", TRANSFORMERS), ("SSMs", SSMS)]:
        arch_data = df[df["model"].isin(models)]
        if not arch_data.empty:
            max_acc = arch_data["top1"].max() * 100
            mean_acc = arch_data["top1"].mean() * 100
            min_acc = arch_data["top1"].min() * 100
            std_acc = arch_data["top1"].std() * 100

            print(f"\n{arch_name}:")
            print(f"  Max:  {max_acc:6.1f}%")
            print(f"  Mean: {mean_acc:6.1f}%")
            print(f"  Min:  {min_acc:6.1f}%")
            print(f"  Std:  {std_acc:6.1f}%")

    # 2. Per-model stats
    print("\n\n2️⃣  PER-MODEL STATISTICS")
    print("-"*90)

    models = sorted(df["model"].unique())
    model_stats = []

    for model in models:
        model_data = df[df["model"] == model]
        max_acc = model_data["top1"].max() * 100
        mean_acc = model_data["top1"].mean() * 100
        std_acc = model_data["top1"].std() * 100

        # Best config
        best_row = model_data.loc[model_data["top1"].idxmax()]
        best_config = f"res={int(best_row['resolution'])}px cov={int(best_row['coverage'])}% s={int(best_row['stride'])}"

        arch_type = "CNN" if model in CNNS else "Transformer" if model in TRANSFORMERS else "SSM"

        print(f"\n{model:15s} [{arch_type}]")
        print(f"  Max:  {max_acc:6.1f}% @ {best_config}")
        print(f"  Mean: {mean_acc:6.1f}% ± {std_acc:5.1f}%")

        model_stats.append({
            "model": model,
            "arch": arch_type,
            "max_acc": max_acc,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
        })

    # 3. ANOVA: Model effect
    print("\n\n3️⃣  ANOVA: MODEL EFFECT")
    print("-"*90)

    groups_model = [df[df["model"] == m]["top1"].values for m in models]
    f_stat_model, p_val_model = stats.f_oneway(*groups_model)

    print(f"\nF-statistic: {f_stat_model:.2f}")
    print(f"p-value: {p_val_model:.2e}")
    print(f"Significância: {'***' if p_val_model < 0.001 else '**' if p_val_model < 0.01 else 'ns'}")

    # 4. Sensitivity by architecture
    print("\n\n4️⃣  SENSITIVITY ANALYSIS BY ARCHITECTURE")
    print("-"*90)

    for arch_name, models_list in [("CNNs", CNNS), ("Transformers", TRANSFORMERS), ("SSMs", SSMS)]:
        arch_data = df[df["model"].isin(models_list)]

        corr_cov = arch_data["coverage"].corr(arch_data["top1"])
        corr_stride = arch_data["stride"].corr(arch_data["top1"])
        corr_res = arch_data["resolution"].corr(arch_data["top1"])

        print(f"\n{arch_name}:")
        print(f"  Coverage ↔ Accuracy: r={corr_cov:+.4f}")
        print(f"  Stride   ↔ Accuracy: r={corr_stride:+.4f}")
        print(f"  Resolution ↔ Accuracy: r={corr_res:+.4f}")

    # 5. Aliasing robustness: stride sensitivity
    print("\n\n5️⃣  ALIASING ROBUSTNESS (stride degradation @ cov=100%)")
    print("-"*90)

    print("\nDegradation from stride=1 to stride=16 (at coverage=100%):")
    print("\nModel           │ s=1    │ s=2    │ s=4    │ s=8    │ s=16   │ Degradation")
    print("─"*85)

    for model in sorted(models):
        model_100 = df[(df["model"] == model) & (df["coverage"] == 100)]
        accs = []
        for s in [1, 2, 4, 8, 16]:
            s_data = model_100[model_100["stride"] == s]
            if not s_data.empty:
                acc = s_data["top1"].mean() * 100
                accs.append(acc)
            else:
                accs.append(np.nan)

        if len(accs) >= 2:
            deg = ((accs[0] - accs[-1]) / accs[0] * 100) if accs[0] > 0 else 0
            print(f"{model:15s} │ {accs[0]:6.1f}% │ {accs[1]:6.1f}% │ {accs[2]:6.1f}% │ "
                  f"{accs[3]:6.1f}% │ {accs[4]:6.1f}% │ {deg:6.1f}%")

    # 6. Resolution sensitivity
    print("\n\n6️⃣  RESOLUTION SENSITIVITY (mean across all coverage/stride)")
    print("-"*90)

    for res in sorted(df["resolution"].unique()):
        res_data = df[df["resolution"] == res]
        print(f"\nResolution {res}px:")
        for model in models:
            model_res = res_data[res_data["model"] == model]
            if not model_res.empty:
                mean_acc = model_res["top1"].mean() * 100
                max_acc = model_res["top1"].max() * 100
                print(f"  {model:15s}: mean={mean_acc:6.1f}%, max={max_acc:6.1f}%")

    # Save stats table
    stats_df = pd.DataFrame(model_stats)
    stats_df.to_csv(sweep_dir / "model_comparison_stats.csv", index=False)
    print(f"\n✓ Model stats saved to: model_comparison_stats.csv")

    print(f"\n{'='*90}")


if __name__ == "__main__":
    main()
