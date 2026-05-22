"""Temporal robustness metrics for fixed-budget evaluations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_robustness_auc(summary: pd.DataFrame, accuracy_col: str = "top1") -> pd.DataFrame:
    """Compute normalized area under the accuracy-vs-budget curve."""
    rows = []
    group_cols = [c for c in ["dataset", "split", "coverage", "stride"] if c in summary.columns]
    for keys, group in summary.groupby(group_cols, dropna=False):
        group = group.sort_values("budget")
        budgets = group["budget"].to_numpy(dtype=float)
        acc = group[accuracy_col].to_numpy(dtype=float)
        if len(group) < 2 or budgets.max() == budgets.min():
            auc = float(acc.mean()) if len(acc) else 0.0
        else:
            auc = float(np.trapz(acc, budgets) / (budgets.max() - budgets.min()))
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["temporal_robustness_auc"] = auc
        row["min_budget"] = int(budgets.min()) if len(budgets) else 0
        row["max_budget"] = int(budgets.max()) if len(budgets) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def critical_frame_budget(
    summary: pd.DataFrame,
    target_fraction: float = 0.95,
    accuracy_col: str = "top1",
) -> pd.DataFrame:
    """Minimum budget retaining target_fraction of the best/dense accuracy."""
    rows = []
    group_cols = [c for c in ["dataset", "split", "coverage", "stride"] if c in summary.columns]
    for keys, group in summary.groupby(group_cols, dropna=False):
        group = group.sort_values("budget")
        best = float(group[accuracy_col].max())
        target = target_fraction * best
        passing = group[group[accuracy_col] >= target]
        critical = int(passing["budget"].min()) if len(passing) else int(group["budget"].max())
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["best_accuracy"] = best
        row["target_accuracy"] = target
        row["critical_frame_budget"] = critical
        rows.append(row)
    return pd.DataFrame(rows)


def compute_temporal_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    auc = temporal_robustness_auc(summary)
    crit = critical_frame_budget(summary)
    group_cols = [c for c in ["dataset", "split", "coverage", "stride"] if c in summary.columns]
    if not group_cols:
        return pd.concat([auc, crit], axis=1)
    return auc.merge(crit, on=group_cols, how="outer")

