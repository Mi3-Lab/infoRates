import argparse
import os
import yaml
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from transformers import AutoImageProcessor, AutoModelForVideoClassification

from info_rates.analysis.evaluate import evaluate_fixed_parallel, evaluate_fixed_parallel_counts
from info_rates.viz.plots import plot_accuracy_curves, plot_heatmap


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
    args = parser.parse_args()

    # Load config and apply defaults
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    model_path = args.model_path or cfg.get("save_path", "models/timesformer_ucf101_ddp")
    manifest = args.manifest or cfg.get("eval_manifest", "UCF101_data/manifests/ucf101_50f.csv")
    coverages = args.coverages if args.coverages else cfg.get("eval_coverages", [10, 25, 50, 75, 100])
    strides = args.strides if args.strides else cfg.get("eval_strides", [1, 2, 4, 8, 16])
    sample_size = args.sample_size if args.sample_size is not None else int(cfg.get("eval_sample_size", 200))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("eval_batch_size", 8))
    workers = args.workers if args.workers is not None else int(cfg.get("eval_workers", 8))
    out_path = args.out or cfg.get("eval_out", "UCF101_data/results/ucf101_50f_finetuned.csv")
    wandb_project = args.wandb_project or cfg.get("wandb_project", "inforates-ucf101")
    wandb_run_name = args.wandb_run_name or cfg.get("eval_wandb_run_name")
    ddp = args.ddp or bool(cfg.get("use_ddp", False))

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
                "model_path": model_path,
                "coverages": coverages,
                "strides": strides,
                "sample_size": sample_size,
                "batch_size": batch_size,
                "workers": workers,
                "ddp": ddp,
                "world_size": world_size,
            }
        )

    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForVideoClassification.from_pretrained(model_path).to(device).eval()

    df = pd.read_csv(manifest)

    if ddp and world_size > 1:
        # Ensure same subset across ranks; sample_size<=0 means full dataset
        if sample_size is not None and sample_size > 0 and sample_size < len(df):
            subset = df.sample(sample_size, random_state=42)
        else:
            subset = df
        # Shard rows across ranks
        my_subset = subset.iloc[rank::world_size]
        local_counts = evaluate_fixed_parallel_counts(
            df=my_subset,
            processor=processor,
            model=model,
            coverages=coverages,
            strides=strides,
            sample_size=len(my_subset),
            batch_size=batch_size,
            num_workers=workers,
        )

        # Aggregate counts across ranks
        # Build a stable ordering of (coverage, stride)
        combos = [(cov, st) for st in strides for cov in coverages]
        agg_rows = []
        for cov, st in combos:
            # Find local row
            row = next((r for r in local_counts if r["coverage"] == cov and r["stride"] == st), {
                "coverage": cov, "stride": st, "correct": 0, "total": 0, "total_time": 0.0
            })
            tensor = torch.tensor([row["correct"], row["total"], row["total_time"]], dtype=torch.float32, device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            correct_sum = int(tensor[0].item())
            total_sum = int(tensor[1].item())
            total_time_sum = float(tensor[2].item())
            acc = (correct_sum / total_sum) if total_sum > 0 else 0.0
            avg_time = (total_time_sum / total_sum) if total_sum > 0 else 0.0
            agg_rows.append({
                "coverage": cov,
                "stride": st,
                "accuracy": acc,
                "avg_time": avg_time,
            })

        df_results = pd.DataFrame(agg_rows)
    else:
        df_results = evaluate_fixed_parallel(
            df=df,
            processor=processor,
            model=model,
            coverages=coverages,
            strides=strides,
            sample_size=sample_size,
            batch_size=batch_size,
            num_workers=workers,
        )

    # Save and log only rank 0
    if not ddp or rank == 0:
        # Ensure parent directory exists before saving
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_results.to_csv(out_path, index=False)

    # Log results to WandB (only rank 0 when DDP)
    if not args.no_wandb and (not ddp or rank == 0):
        wandb.log({"results_table": wandb.Table(dataframe=df_results)})
        
        # Log summary metrics
        best_acc = df_results["accuracy"].max()
        best_config = df_results.loc[df_results["accuracy"].idxmax()]
        wandb.summary["best_accuracy"] = best_acc
        wandb.summary["best_coverage"] = best_config["coverage"]
        wandb.summary["best_stride"] = best_config["stride"]

    if not ddp or rank == 0:
        plot_accuracy_curves(df_results)
        plot_heatmap(df_results)
    
    if not args.no_wandb and (not ddp or rank == 0):
        wandb.finish()

    # Cleanup DDP
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
