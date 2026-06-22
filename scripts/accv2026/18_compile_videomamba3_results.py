#!/usr/bin/env python3
"""Compile VideoMamba3 training/latency artifacts into paper-ready tables."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evaluations/accv2026/videomamba3"
DEFAULT_DEPTHS = {"videomamba3_tiny": 24, "videomamba3_small": 24, "videomamba3_middle": 32}


def infer_depth(meta: dict) -> int | None:
    depth = meta.get("depth")
    if depth:
        return depth
    return DEFAULT_DEPTHS.get(meta.get("model_name"))


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    view = df.copy()
    for col in cols:
        if col not in view.columns:
            view[col] = ""
    rows = view[cols].fillna("").astype(str).values.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def load_training_rows() -> list[dict]:
    rows = []
    for meta_path in sorted((ROOT / "fine_tuned_models").glob("accv2026_videomamba3*/accv_meta.json")):
        meta = json.loads(meta_path.read_text())
        save_dir = meta_path.parent
        history_path = save_dir.with_name(save_dir.name + "_history.csv")
        hist = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        best = hist.iloc[hist["val_accuracy"].idxmax()].to_dict() if not hist.empty else {}
        rows.append({
            "checkpoint": str(save_dir.relative_to(ROOT)),
            "model_name": meta.get("model_name"),
            "variant": meta.get("mamba3_variant"),
            "num_frames": meta.get("num_frames"),
            "input_size": meta.get("input_size"),
            "depth": infer_depth(meta),
            "num_labels": meta.get("num_labels"),
            "epoch": meta.get("epoch", best.get("epoch")),
            "val_acc": meta.get("val_acc", best.get("val_accuracy")),
            "best_history_val_acc": best.get("val_accuracy"),
            "train_loss": best.get("train_loss"),
            "val_loss": best.get("val_loss"),
            "epoch_seconds": best.get("epoch_seconds"),
            "samples_per_second": best.get("samples_per_second"),
            "ssm_cfg": json.dumps(meta.get("ssm_cfg", {}), sort_keys=True),
        })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_training_rows()
    train_csv = OUT / "training_summary.csv"
    if rows:
        pd.DataFrame(rows).to_csv(train_csv, index=False)
        print(f"Wrote {train_csv}")
    else:
        print("No VideoMamba3 checkpoints found.")

    latency_csv = OUT / "latency_benchmark.csv"
    system_csvs = sorted(OUT.glob("system_validation*.csv"))
    paper_md = OUT / "paper_tables.md"
    with open(paper_md, "w", encoding="utf-8") as f:
        f.write("# VideoMamba3 Experiment Tables\n\n")
        if rows:
            df = pd.DataFrame(rows)
            cols = ["checkpoint", "variant", "num_frames", "input_size", "depth", "val_acc", "epoch_seconds", "samples_per_second"]
            f.write("## Training Summary\n\n")
            f.write(markdown_table(df, cols))
            f.write("\n\n")
        if latency_csv.exists():
            lat = pd.read_csv(latency_csv)
            if "depth" not in lat.columns and "model_size" in lat.columns:
                lat["depth"] = lat["model_size"].map({
                    "tiny": 24,
                    "small": 24,
                    "middle": 32,
                })
            cols = ["variant", "seq_len", "depth", "mean_inference_s", "videos_per_second", "peak_memory_mb", "params"]
            f.write("## Latency Summary\n\n")
            f.write(markdown_table(lat, cols))
            f.write("\n")
        if system_csvs:
            sys_df = pd.concat([pd.read_csv(path).assign(source=path.name) for path in system_csvs], ignore_index=True)
            cols = [
                "source", "status", "variant", "depth", "mode", "batch_size", "num_frames", "input_size",
                "seq_len", "mean_s", "videos_per_second", "tokens_per_second",
                "peak_allocated_mb", "peak_reserved_mb", "torch_compile", "gpu_name",
            ]
            f.write("\n## System Validation\n\n")
            f.write(markdown_table(sys_df, cols))
            f.write("\n")
    print(f"Wrote {paper_md}")


if __name__ == "__main__":
    main()
