import argparse
import math
import os
import yaml
import pandas as pd
import torch
import torch.distributed as dist
import wandb
import random
import numpy as np
from transformers import AutoImageProcessor, AutoModelForVideoClassification

from info_rates.analysis.evaluate import (
    evaluate_fixed_parallel,
    per_class_analysis_fast,
)
from info_rates.analysis.evaluate_fixed import evaluate_fixed_parallel_counts

# Optional: plotting (requires matplotlib, seaborn)
try:
    from info_rates.viz.plots import plot_accuracy_curves, plot_heatmap
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = correct / total
    denom = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    low = max(0.0, (centre - margin) / denom)
    high = min(1.0, (centre + margin) / denom)
    return low, high


def compute_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Pareto-optimal (accuracy, latency) frontier."""
    df_sorted = df.sort_values("avg_time")
    frontier = []
    best_acc = -1.0
    for _, row in df_sorted.iterrows():
        acc = row["accuracy"]
        if acc >= best_acc:
            frontier.append(row)
            best_acc = acc
    return pd.DataFrame(frontier)


def main():
    parser = argparse.ArgumentParser(description="Evaluate temporal sampling effects on fixed-length clips.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model-path", type=str, help="Path to saved model directory")
    parser.add_argument("--manifest", type=str, help="CSV manifest with columns: video_path,label")
    parser.add_argument("--coverages", nargs="*", type=int)
    parser.add_argument("--strides", nargs="*", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--out", type=str)
    parser.add_argument("--wandb-project", default="inforates-ucf101", help="WandB project name")
    parser.add_argument("--wandb-run-name", default=None, help="WandB run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--ddp", action="store_true", help="Enable Distributed Data Parallel evaluation")
    parser.add_argument("--per-class", action="store_true", help="Compute per-class metrics as well")
    parser.add_argument("--per-class-out", type=str, help="Path to save per-class CSV")
    parser.add_argument("--per-class-sample-size", type=int, help="Limit per-class eval to N samples for speed")
    parser.add_argument("--jitter-coverage-pct", type=float, help="Randomly jitter coverage ±pct during eval for robustness checks")
    args = parser.parse_args()

    # Load config and apply defaults
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    model_path = args.model_path or cfg.get("save_path", "models/timesformer_ucf101_ddp")
    manifest = args.manifest or cfg.get("eval_manifest", "data/UCF101_data/manifests/ucf101_50f.csv")
    coverages = args.coverages if args.coverages else cfg.get("eval_coverages", [10, 25, 50, 75, 100])
    strides = args.strides if args.strides else cfg.get("eval_strides", [1, 2, 4, 8, 16])
    sample_size = args.sample_size if args.sample_size is not None else int(cfg.get("eval_sample_size", 200))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("eval_batch_size", 8))
    workers = args.workers if args.workers is not None else int(cfg.get("eval_workers", 8))


    # Derive per-model default output paths when not explicitly provided
    default_results_dir = cfg.get("eval_results_dir", "data/UCF101_data/results")
    model_tag = os.path.basename(os.path.normpath(model_path))
    # Extract model name (timesformer, videomae, vivit) from model_tag
    model_name = "timesformer" if "timesformer" in model_tag.lower() else \
                 "videomae" if "videomae" in model_tag.lower() else \
                 "vivit" if "vivit" in model_tag.lower() else "other"
    model_results_dir = os.path.join(default_results_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    default_out = os.path.join(model_results_dir, f"{model_tag}_temporal_sampling.csv")
    out_path = args.out or cfg.get("eval_out", default_out)

    # Set model-specific frame count for ViViT (32), VideoMAE (16), TimeSformer (8)
    if model_name == "vivit":
        eval_num_frames = 32
    elif model_name == "videomae":
        eval_num_frames = 16
    else:
        eval_num_frames = 8

    wandb_project = args.wandb_project or cfg.get("wandb_project", "inforates-ucf101")
    wandb_run_name = args.wandb_run_name or cfg.get("eval_wandb_run_name")
    ddp = args.ddp or bool(cfg.get("use_ddp", False))
    do_per_class = args.per_class or bool(cfg.get("eval_per_class", False))
    default_per_class_out = os.path.join(model_results_dir, f"{model_tag}_per_class.csv")
    per_class_out = args.per_class_out or cfg.get("eval_per_class_out", default_per_class_out)
    per_class_sample_size = (
        args.per_class_sample_size if args.per_class_sample_size is not None else int(cfg.get("eval_per_class_sample_size", -1))
    )  # -1 means full
    jitter_coverage_pct = args.jitter_coverage_pct if args.jitter_coverage_pct is not None else float(cfg.get("eval_jitter_coverage_pct", 0.0))

    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Setup DDP if requested
    rank = 0
    world_size = 1
    if ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize WandB (only rank 0 when DDP)
    if not args.no_wandb and (not ddp or rank == 0):
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_name": model_name,
                "model_path": model_path,
                "manifest": args.manifest,
                "dataset": os.path.basename(args.manifest).replace('.csv', ''),
                "coverages": coverages,
                "strides": strides,
                "sample_size": sample_size,
                "batch_size": batch_size,
                "workers": workers,
                "ddp": ddp,
                "world_size": world_size,
                "jitter_coverage_pct": jitter_coverage_pct,
            }
        )

    processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
    model = AutoModelForVideoClassification.from_pretrained(model_path).to(device).eval()

    df = pd.read_csv(manifest)

    # Check for existing results to enable resume/checkpoint
    completed_configs = set()
    if os.path.exists(out_path):
        try:
            existing_df = pd.read_csv(out_path)
            if not existing_df.empty and all(c in existing_df.columns for c in ["coverage", "stride", "total"]):
                # Only mark as completed if evaluated on full dataset
                expected_total = len(df) if (sample_size is None or sample_size <= 0) else min(sample_size, len(df))
                for _, row in existing_df.iterrows():
                    # Check if this row represents a full evaluation (allow 5% tolerance for DDP sharding)
                    if row['total'] >= expected_total * 0.95:
                        completed_configs.add((int(row['coverage']), int(row['stride'])))
                
                if not ddp or rank == 0:
                    print(f"Found existing results with {len(existing_df)} rows.")
                    print(f"Dataset size: {len(df)}, Expected per config: {expected_total}")
                    if completed_configs:
                        print(f"Marking {len(completed_configs)} configs as completed (full dataset).")
                        print(f"Skipping: {sorted(completed_configs)}")
                    else:
                        print("All existing results are partial - will re-evaluate all configs.")
                        existing_df = None  # Don't merge partial results
        except Exception as e:
            if not ddp or rank == 0:
                print(f"Warning: Could not load existing results from {out_path}: {e}")

    # Filter out already-completed configs
    pending_configs = [(c, s) for c in coverages for s in strides if (c, s) not in completed_configs]
    existing_df = None  # Ensure this is always defined
    if not pending_configs and not args.per_class:
        if not ddp or rank == 0:
            print("All configurations already completed. Nothing to evaluate.")
        if not args.no_wandb and (not ddp or rank == 0):
            wandb.finish()
        if ddp:
            dist.destroy_process_group()
        return

    # Extract only pending coverages and strides
    pending_coverages = sorted(set(c for c, s in pending_configs)) if pending_configs else []
    pending_strides = sorted(set(s for c, s in pending_configs)) if pending_configs else []

    # If no pending configs but per-class is requested, load existing results
    if not pending_configs and args.per_class:
        if not ddp or rank == 0:
            print(f"No pending configs. Loading existing results for per-class analysis...")
        if existing_df is None and os.path.exists(out_path):
            existing_df = pd.read_csv(out_path)
        if existing_df is not None and not existing_df.empty:
            if not ddp or rank == 0:
                print(f"Loaded {len(existing_df)} configurations for per-class analysis.")
        else:
            if not ddp or rank == 0:
                print("Error: No existing results to use for per-class analysis.")
            if not args.no_wandb and (not ddp or rank == 0):
                wandb.finish()
            if ddp:
                dist.destroy_process_group()
            return

    if not ddp or rank == 0:
        print(f"Evaluating {len(pending_configs)} pending configurations...")
        if pending_coverages:
            print(f"Coverages: {pending_coverages}")
            print(f"Strides: {pending_strides}")

    # Prepare data subset
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        subset = df.sample(sample_size, random_state=42)
    else:
        subset = df

    # Evaluate one config at a time with incremental saving (skip if no pending configs)
    if pending_configs and ddp and world_size > 1:
        # Shard rows across ranks
        my_subset = subset.iloc[rank::world_size]
        # Process each configuration individually
        for stride in pending_strides:
            for coverage in pending_coverages:
                if (coverage, stride) in completed_configs:
                    continue
                # Evaluate single config (do NOT pass num_frames to evaluate_fixed_parallel_counts)
                local_counts = evaluate_fixed_parallel_counts(
                    df=my_subset,
                    processor=processor,
                    model=model,
                    coverages=[coverage],
                    strides=[stride],
                    sample_size=len(my_subset),
                    batch_size=batch_size,
                    num_workers=workers,
                    jitter_coverage_pct=jitter_coverage_pct,
                    num_frames=eval_num_frames,
                )
                # Aggregate across ranks
                if not local_counts.empty:
                    correct_sum = int((local_counts["accuracy"] * local_counts["n_samples"]).sum())
                    total_sum = int(local_counts["n_samples"].sum())
                    total_time_sum = 0.0  # Not available per class
                else:
                    correct_sum = 0
                    total_sum = 0
                    total_time_sum = 0.0
                tensor = torch.tensor([correct_sum, total_sum, total_time_sum], dtype=torch.float32, device=device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                correct_sum = int(tensor[0].item())
                total_sum = int(tensor[1].item())
                total_time_sum = float(tensor[2].item())
                acc = (correct_sum / total_sum) if total_sum > 0 else 0.0
                avg_time = (total_time_sum / total_sum) if total_sum > 0 else 0.0
                new_row = pd.DataFrame([{
                    "coverage": coverage,
                    "stride": stride,
                    "accuracy": acc,
                    "avg_time": avg_time,
                    "correct": correct_sum,
                    "total": total_sum,
                }])
                # Append to existing results and save immediately (rank 0 only)
                if rank == 0:
                    if existing_df is not None:
                        existing_df = pd.concat([existing_df, new_row], ignore_index=True)
                    else:
                        existing_df = new_row
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    existing_df.to_csv(out_path, index=False, float_format='%.6f')
                    print(f"✓ Saved: stride={stride} cov={coverage}% -> acc={acc:.4f}")
        df_results = existing_df if existing_df is not None else pd.DataFrame()
    elif pending_configs:
        # Non-DDP: evaluate one config at a time
        all_rows = []
        for stride in pending_strides:
            for coverage in pending_coverages:
                if (coverage, stride) in completed_configs:
                    continue
                result = evaluate_fixed_parallel(
                    df=subset,
                    processor=processor,
                    model=model,
                    coverages=[coverage],
                    strides=[stride],
                    sample_size=sample_size,
                    batch_size=batch_size,
                    num_workers=workers,
                    jitter_coverage_pct=jitter_coverage_pct,
                    rank=rank,
                    num_frames=eval_num_frames,
                )
                # Append and save immediately
                all_rows.append(result.iloc[0])
                if existing_df is not None:
                    existing_df = pd.concat([existing_df, result], ignore_index=True)
                else:
                    existing_df = result
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                existing_df.to_csv(out_path, index=False, float_format='%.6f')
                print(f"✓ Saved: stride={stride} cov={coverage}% -> acc={result.iloc[0]['accuracy']:.4f}")
        df_results = existing_df if existing_df is not None else pd.DataFrame()
    else:
        # No evaluation needed, use loaded results for per-class
        df_results = existing_df if existing_df is not None else pd.DataFrame()

    # Final save (redundant but safe)
    if not ddp or rank == 0:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_results.to_csv(out_path, index=False, float_format='%.6f')

    # Gather results to rank 0 and log to WandB (for DDP)
    if ddp and world_size > 1:
        # Gather df_results from all ranks to rank 0
        import pickle
        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, pickle.dumps(df_results))
        if rank == 0:
            import pandas as pd
            df_results = pd.concat([pickle.loads(x) for x in gathered_results], ignore_index=True)
    if not args.no_wandb and (not ddp or rank == 0):
        wandb.log({"results_table": wandb.Table(dataframe=df_results)})
        # Log Pareto frontier
        frontier = compute_pareto(df_results)
        if not frontier.empty:
            wandb.log({"pareto_frontier": wandb.Table(dataframe=frontier)})
        # Log summary metrics
        best_acc = df_results["accuracy"].max()
        best_config = df_results.loc[df_results["accuracy"].idxmax()]
        wandb.summary["best_accuracy"] = best_acc
        wandb.summary["best_coverage"] = best_config["coverage"]
        wandb.summary["best_stride"] = best_config["stride"]
        # Log plots if available
        if PLOTTING_AVAILABLE:
            import glob
            model_results_dir = os.path.dirname(out_path)
            for img_path in glob.glob(os.path.join(model_results_dir, '*.png')):
                wandb.log({"plot": wandb.Image(img_path)})

    if not ddp or rank == 0:
        if PLOTTING_AVAILABLE:
            # Get model-specific results directory
            model_results_dir = os.path.dirname(out_path)
            plot_accuracy_curves(df_results, output_dir=model_results_dir)
            plot_heatmap(df_results, output_dir=model_results_dir)
        else:
            print("Plotting skipped (matplotlib/seaborn not installed)")

    # Optional per-class analysis - use both GPUs by splitting configs
    if do_per_class:
        subset_size = len(df) if per_class_sample_size <= 0 else min(per_class_sample_size, len(df))
        # Get model's expected frame count from config
        model_num_frames = model.config.num_frames if hasattr(model.config, 'num_frames') else 16
        
        # Split configs across ranks for parallel processing
        all_configs = [(c, s) for c in coverages for s in strides]
        if ddp:
            # Each rank processes a subset of configs
            my_configs = [(c, s) for i, (c, s) in enumerate(all_configs) if i % world_size == rank]
            my_coverages = sorted(set(c for c, s in my_configs))
            my_strides = sorted(set(s for c, s in my_configs))
            if rank == 0:
                print(f"Rank {rank}: processing {len(my_configs)}/{len(all_configs)} configs")
        else:
            my_coverages = coverages
            my_strides = strides
        
        # Each rank processes its subset
        # Use per_class_out as checkpoint path
        df_per_class = per_class_analysis_fast(
            df=df,
            processor=processor,
            model=model,
            coverages=my_coverages,
            strides=my_strides,
            sample_size=subset_size,
            batch_size=max(32, batch_size),  # Use larger batches for per-class (faster)
            num_workers=workers,
            rank=rank,
            num_frames=model_num_frames,  # Pass correct frame count
            checkpoint_path=per_class_out,
        )
        
        # Gather results from all ranks
        if ddp:
            # Gather all dataframes to rank 0
            gathered_dfs = [None] * world_size
            dist.all_gather_object(gathered_dfs, df_per_class)
            if rank == 0:
                # Combine all results
                df_per_class = pd.concat(gathered_dfs, ignore_index=True)
        
        # Save and log only on rank 0
        if not ddp or rank == 0:
            os.makedirs(os.path.dirname(per_class_out), exist_ok=True)
            df_per_class.to_csv(per_class_out, index=False, float_format='%.6f')
            print(f"✅ Saved per-class results: {len(df_per_class)} rows")

        if not args.no_wandb and (not ddp or rank == 0):
            wandb.log({"per_class_table": wandb.Table(dataframe=df_per_class)})

            # Summaries: best per class and top aliasing drop (100% vs 25% coverage mean across strides)
            best_per_class = (
                df_per_class.sort_values(["class", "accuracy"], ascending=[True, False])
                .groupby("class", as_index=False)
                .first()[["class", "coverage", "stride", "accuracy", "n_samples"]]
            )
            wandb.log({"per_class_best": wandb.Table(dataframe=best_per_class)})

            # Aliasing sensitivity: drop from 100% to 25%
            pivot_drop = (
                df_per_class[df_per_class["coverage"].isin([25, 100])]
                .groupby(["class", "coverage"], as_index=False)["accuracy"].mean()
                .pivot(index="class", columns="coverage", values="accuracy")
            )
            if 25 in pivot_drop.columns and 100 in pivot_drop.columns:
                drop_df = pivot_drop.reset_index().rename(columns={100: "acc_100", 25: "acc_25"})
                drop_df["drop_100_minus_25"] = drop_df["acc_100"] - drop_df["acc_25"]
                drop_df = drop_df.sort_values("drop_100_minus_25", ascending=False).head(20)
                wandb.log({"per_class_aliasing_drop": wandb.Table(dataframe=drop_df)})
    
    if not args.no_wandb and (not ddp or rank == 0):
        wandb.finish()

    # Cleanup DDP
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
