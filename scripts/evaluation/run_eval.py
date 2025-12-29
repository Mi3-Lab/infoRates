# Ensure project root and src/ are on sys.path for direct script execution
import os
import sys
# __file__ is scripts/evaluation/run_eval.py; we want the repository root (parent of scripts)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_PROJECT_ROOT, _SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import math
import os
import yaml
import pandas as pd
import torch
import torch.distributed as dist
try:
    import wandb
except ImportError:
    wandb = None
import random
import numpy as np
from transformers import AutoImageProcessor, AutoModelForVideoClassification

from info_rates.analysis.evaluate import (
    evaluate_fixed_parallel,
    per_class_analysis_fast,
)
from info_rates.analysis.evaluate_fixed import evaluate_fixed_parallel_counts
from info_rates.sampling.temporal import norm_label
from scripts.data_processing.dataset_handler import DatasetHandler

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
    parser.add_argument("--dataset", type=str, choices=["ucf101", "kinetics400", "hmdb51", "something"], help="Dataset to evaluate on (overrides config.yaml)")
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
    parser.add_argument("--label-map", type=str, help="CSV file mapping textual label -> integer model id (text,id)")
    parser.add_argument("--topk", type=int, default=1, help="Consider top-k predictions when matching textual labels")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.5, help="Token overlap threshold for fuzzy textual matching (0-1)")
    args = parser.parse_args()

    # Load config and apply defaults
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # Get dataset configuration (use config.yaml dataset as default, command-line overrides)
    dataset_name = args.dataset or cfg.get("dataset", "ucf101")
    dataset_config = DatasetHandler.get_dataset_config(dataset_name)
    dataset_defaults = DatasetHandler.get_model_defaults(dataset_name)

    # Set dataset-specific defaults
    model_name = cfg.get("model_name", "timesformer")
    model_path = args.model_path or cfg.get("save_path", f"fine_tuned_models/fine_tuned_{model_name}_{dataset_name}")
    manifest = args.manifest or DatasetHandler.get_default_manifest(dataset_name)
    coverages = args.coverages if args.coverages else cfg.get("eval_coverages", [10, 25, 50, 75, 100])
    strides = args.strides if args.strides else cfg.get("eval_strides", [1, 2, 4, 8, 16])
    sample_size = args.sample_size if args.sample_size is not None else int(cfg.get("eval_sample_size", 200))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("eval_batch_size", 8))
    workers = args.workers if args.workers is not None else int(cfg.get("eval_workers", 8))


    # Derive per-model default output paths when not explicitly provided
    default_results_dir = dataset_config["results_dir"]
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

    wandb_project = args.wandb_project or dataset_config["wandb_project"]
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


    # Special handling for HMDB51: load Kinetics-400 pretrained model, but re-init head for 51 classes
    if dataset_name == "hmdb51" and (not os.path.exists(model_path) or model_path == f"fine_tuned_models/fine_tuned_{model_name}_{dataset_name}"):
        # Use Kinetics-400 model as base, but set num_labels=51
        from transformers import AutoConfig
        base_model_id = "facebook/timesformer-base-finetuned-k400"
        num_labels = 51
        processor = AutoImageProcessor.from_pretrained(base_model_id, use_fast=True)
        config = AutoConfig.from_pretrained(base_model_id)
        # Build HMDB51 label mapping (0-50)
        id2label = {i: str(i) for i in range(num_labels)}
        label2id = {str(i): i for i in range(num_labels)}
        config.id2label = id2label
        config.label2id = label2id
        config.num_labels = num_labels
        config.num_labels = num_labels  # Set number of labels in config
        model = AutoModelForVideoClassification.from_pretrained(
            base_model_id,
            config=config,
            ignore_mismatched_sizes=True,
        ).to(device).eval()
        print(f"[INFO] Loaded Kinetics-400 pretrained model with randomly initialized 51-class head for HMDB51 evaluation.")
    else:
        processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
        model = AutoModelForVideoClassification.from_pretrained(model_path).to(device).eval()

    # Load manifest: prefer explicit --manifest if provided, otherwise load or build default
    if args.manifest:
        manifest_path = args.manifest
        if os.path.exists(manifest_path):
            # Support manifests with or without header; attempt best-effort parsing
            try:
                df = pd.read_csv(manifest_path, header=None, names=["video_path","label"], dtype={0:str,1:object})
            except Exception:
                df = pd.read_csv(manifest_path)
        else:
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    else:
        df, manifest_path = DatasetHandler.load_or_build_manifest(dataset_name)
    print(f"Loaded dataset: {dataset_name} with {len(df)} samples (manifest: {manifest_path})")

    # Load optional label map (textual_label -> integer id)
    label_map = None
    if args.label_map:
        if os.path.exists(args.label_map):
            try:
                # Try reading CSV with header first to detect common column names
                df_map = pd.read_csv(args.label_map)
                if 'k400_id' in df_map.columns and 'sth_text' in df_map.columns:
                    label_map = {}
                    # Build template-token mapping for fuzzy matching with templates like 'something'
                    templates = []
                    patterns = []
                    import re
                    for _, r in df_map.iterrows():
                        try:
                            if pd.isna(r.get('k400_id')):
                                continue
                            sth = str(r['sth_text'])
                            kid = int(r['k400_id'])
                            label_map[norm_label(sth)] = kid
                            toks = set(re.findall(r"\w+", sth.lower()))
                            toks -= {"a","the","an","something"}
                            templates.append((toks, kid))
                            # build a regex pattern where 'something' matches any token sequence
                            pat = re.escape(sth.lower())
                            pat = pat.replace(re.escape('something'), '(.+?)')
                            patterns.append((re.compile(pat, flags=re.I), kid, sth))
                        except Exception:
                            continue
                    label_map['__templates'] = templates
                    label_map['__patterns'] = patterns
                elif 'text' in df_map.columns and 'id' in df_map.columns:
                    label_map = {norm_label(str(r['text'])): int(r['id']) for _, r in df_map.iterrows()}
                else:
                    # Fallback: assume two-column CSV text,id
                    try:
                        df_map = pd.read_csv(args.label_map, header=None, names=["text","id"])
                        label_map = {norm_label(str(r['text'])): int(r['id']) for _, r in df_map.iterrows()}
                    except Exception:
                        raise RuntimeError("Unrecognized label map format. Expected columns like (sth_text,k400_id) or (text,id)")
                print(f"Loaded label map: {len(label_map)} entries from {args.label_map}")
            except Exception as e:
                raise RuntimeError(f"Failed to parse label map {args.label_map}: {e}")
        else:
            raise FileNotFoundError(f"Label map file not found: {args.label_map}")
    topk = args.topk
    fuzzy_threshold = args.fuzzy_threshold

    # Initialize WandB (only rank 0 when DDP) - after manifest is loaded
    if wandb is not None and not args.no_wandb and (not ddp or rank == 0):
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_name": model_name,
                "model_path": model_path,
                "manifest": manifest_path,
                "dataset": dataset_name,
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

    print(f"Loaded dataset: {dataset_name} with {len(df)} samples")

    # Check for existing results to enable resume/checkpoint
    completed_configs = set()
    existing_df = None  # Initialize
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
            existing_df = None
    
    # Calculate pending configurations
    all_configs = set((c, s) for c in coverages for s in strides)
    pending_configs = all_configs - completed_configs
    
    # Prioritize per-class analysis when requested
    if args.per_class:
        if not ddp or rank == 0:
            print("Per-class analysis requested. Loading existing results...")
        if existing_df is None and os.path.exists(out_path):
            existing_df = pd.read_csv(out_path)
        if existing_df is not None and not existing_df.empty:
            if not ddp or rank == 0:
                print(f"Loaded {len(existing_df)} configurations for per-class analysis.")
        else:
            if not ddp or rank == 0:
                print("Error: No existing results to use for per-class analysis.")
            if wandb is not None and not args.no_wandb and (not ddp or rank == 0):
                wandb.finish()
            if ddp:
                dist.destroy_process_group()
            return
    elif not pending_configs:
        if not ddp or rank == 0:
            print("All configurations already completed. Nothing to evaluate.")
        if wandb is not None and not args.no_wandb and (not ddp or rank == 0):
            wandb.finish()
        if ddp:
            dist.destroy_process_group()
        return

    # Extract only pending coverages and strides (only needed for main evaluation)
    if not args.per_class:
        pending_coverages = sorted(set(c for c, s in pending_configs)) if pending_configs else []
        pending_strides = sorted(set(s for c, s in pending_configs)) if pending_configs else []
    else:
        pending_coverages = []
        pending_strides = []

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
                    # print(f"[DEBUG] Skipping completed stride={stride} cov={coverage}%", flush=True)
                    continue
                # print(f"[DEBUG] Rank {rank} START stride={stride} cov={coverage}% (calling evaluate_fixed_parallel_counts)", flush=True)
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
                    label_map=label_map,
                    topk=topk,
                    fuzzy_threshold=fuzzy_threshold,
                )
                # print(f"[DEBUG] Rank {rank} END stride={stride} cov={coverage}% (returned from evaluate_fixed_parallel_counts)", flush=True)
                # print(f"[DEBUG] local_counts head:\n{local_counts.head() if hasattr(local_counts, 'head') else local_counts}", flush=True)
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
                    print(f"[RESULT] stride={stride} cov={coverage}% -> acc={acc:.4f}", flush=True)
                # print(f"[DEBUG] Rank {rank} COMPLETE stride={stride} cov={coverage}%", flush=True)
        df_results = existing_df if existing_df is not None else pd.DataFrame()
    elif pending_configs and ddp and world_size > 1:
        # DDP main evaluation: shard data across ranks
        my_subset = subset.iloc[rank::world_size]
        all_rows = []
        processed_configs = set(completed_configs)
        failed_configs = set()
        for stride in pending_strides:
            for coverage in pending_coverages:
                if (coverage, stride) in processed_configs:
                    # print(f"[DEBUG] Skipping already processed stride={stride} cov={coverage}%", flush=True)
                    continue
                # print(f"[DEBUG] Processing stride={stride} cov={coverage}%", flush=True)
                try:
                    result = evaluate_fixed_parallel(
                        df=my_subset,
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
                        label_map=label_map,
                        topk=topk,
                        fuzzy_threshold=fuzzy_threshold,
                    )
                    # print(f"[DEBUG] Type of result: {type(result)}", flush=True)
                    # print(f"[DEBUG] Content of result: {result}", flush=True)
                    # Handle both DataFrame and list return types
                    valid = False
                    local_correct = local_total = local_time_sum = 0
                    if hasattr(result, 'iloc') and 'accuracy' in result.columns:
                        row = result.iloc[0]
                        local_correct = int(row['correct'])
                        local_total = int(row['total'])
                        local_time_sum = row['avg_time'] * local_total  # Convert avg_time back to total time
                        valid = True
                    elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and 'accuracy' in result[0]:
                        row = result[0]
                        local_correct = int(row['correct'])
                        local_total = int(row['total'])
                        local_time_sum = row['avg_time'] * local_total  # Convert avg_time back to total time
                        valid = True
                    else:
                        print(f"[WARNING] Invalid or empty result for stride={stride} cov={coverage}%, skipping save.", flush=True)
                        valid = False

                    if valid:
                        # Aggregate results across ranks
                        tensor = torch.tensor([local_correct, local_total, local_time_sum], dtype=torch.float32, device=device)
                        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                        global_correct = int(tensor[0].item())
                        global_total = int(tensor[1].item())
                        global_time_sum = float(tensor[2].item())
                        global_acc = (global_correct / global_total) if global_total > 0 else 0.0
                        global_avg_time = (global_time_sum / global_total) if global_total > 0 else 0.0
                        result_df = pd.DataFrame([{
                            "coverage": coverage,
                            "stride": stride,
                            "accuracy": global_acc,
                            "avg_time": global_avg_time,
                            "correct": global_correct,
                            "total": global_total,
                        }])
                        acc_val = global_acc
                    else:
                        acc_val = 'N/A'

                    # Save results (only rank 0 in DDP mode)
                    if rank == 0:
                        if existing_df is not None:
                            existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                        else:
                            existing_df = result_df
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        existing_df.to_csv(out_path, index=False, float_format='%.6f')
                        print(f"[RESULT] stride={stride} cov={coverage}% -> acc={acc_val:.4f}", flush=True)
                    processed_configs.add((coverage, stride))
                except Exception as e:
                    print(f"[ERROR] Exception for stride={stride} cov={coverage}%: {e}", flush=True)
                    failed_configs.add((coverage, stride))
                # print(f"[DEBUG] Processed configs so far: {processed_configs}", flush=True)
        df_results = existing_df if existing_df is not None else pd.DataFrame()
    elif pending_configs:
        # Non-DDP: evaluate one config at a time
        all_rows = []
        processed_configs = set(completed_configs)
        failed_configs = set()
        for stride in pending_strides:
            for coverage in pending_coverages:
                if (coverage, stride) in processed_configs:
                    # print(f"[DEBUG] Skipping already processed stride={stride} cov={coverage}%", flush=True)
                    continue
                # print(f"[DEBUG] Processing stride={stride} cov={coverage}%", flush=True)
                try:
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
                        label_map=label_map,
                        topk=topk,
                        fuzzy_threshold=fuzzy_threshold,
                    )
                    # print(f"[DEBUG] Type of result: {type(result)}", flush=True)
                    # print(f"[DEBUG] Content of result: {result}", flush=True)
                    # Handle both DataFrame and list return types
                    valid = False
                    local_correct = local_total = local_time_sum = 0
                    if hasattr(result, 'iloc') and 'accuracy' in result.columns:
                        row = result.iloc[0]
                        local_correct = int(row['correct'])
                        local_total = int(row['total'])
                        local_time_sum = row['avg_time'] * local_total  # Convert avg_time back to total time
                        valid = True
                    elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and 'accuracy' in result[0]:
                        row = result[0]
                        local_correct = int(row['correct'])
                        local_total = int(row['total'])
                        local_time_sum = row['avg_time'] * local_total  # Convert avg_time back to total time
                        valid = True
                    else:
                        print(f"[WARNING] Invalid or empty result for stride={stride} cov={coverage}%, skipping save.", flush=True)
                        valid = False

                    if valid:
                        if ddp:
                            # Aggregate results across ranks
                            tensor = torch.tensor([local_correct, local_total, local_time_sum], dtype=torch.float32, device=device)
                            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                            global_correct = int(tensor[0].item())
                            global_total = int(tensor[1].item())
                            global_time_sum = float(tensor[2].item())
                            global_acc = (global_correct / global_total) if global_total > 0 else 0.0
                            global_avg_time = (global_time_sum / global_total) if global_total > 0 else 0.0
                            result_df = pd.DataFrame([{
                                "coverage": coverage,
                                "stride": stride,
                                "accuracy": global_acc,
                                "avg_time": global_avg_time,
                                "correct": global_correct,
                                "total": global_total,
                            }])
                            acc_val = global_acc
                        else:
                            # Non-DDP: use local results directly
                            if hasattr(result, 'iloc'):
                                result_df = result
                                acc_val = result.iloc[0]['accuracy']
                            else:
                                result_df = pd.DataFrame(result)
                                acc_val = result[0]['accuracy']

                        # Save results (only rank 0 in DDP mode)
                        if not ddp or rank == 0:
                            if existing_df is not None:
                                existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                            else:
                                existing_df = result_df
                            os.makedirs(os.path.dirname(out_path), exist_ok=True)
                            existing_df.to_csv(out_path, index=False, float_format='%.6f')
                            # print(f"[DEBUG] Saved stride={stride} cov={coverage}% -> acc={acc_val}", flush=True)
                    processed_configs.add((coverage, stride))
                except Exception as e:
                    print(f"[ERROR] Exception for stride={stride} cov={coverage}%: {e}", flush=True)
                    failed_configs.add((coverage, stride))
                # print(f"[DEBUG] Processed configs so far: {processed_configs}", flush=True)
                # print(f"[DEBUG] Failed configs so far: {failed_configs}", flush=True)
        print(f"[SUMMARY] All processed configs: {processed_configs}", flush=True)
        print(f"[SUMMARY] All failed configs: {failed_configs}", flush=True)
        df_results = existing_df if existing_df is not None else pd.DataFrame()
    else:
        # No evaluation needed, use loaded results for per-class
        df_results = existing_df if existing_df is not None else pd.DataFrame()

    # Final save (redundant but safe)
    if not ddp or rank == 0:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_results.to_csv(out_path, index=False, float_format='%.6f')

    # Gather results to rank 0 and log to WandB (for DDP) - only if we actually evaluated new configs
    if ddp and world_size > 1 and pending_configs:
        # Gather df_results from all ranks to rank 0
        import pickle
        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, pickle.dumps(df_results))
        if rank == 0:
            df_results = pd.concat([pickle.loads(x) for x in gathered_results], ignore_index=True)
    if wandb is not None and not args.no_wandb and (not ddp or rank == 0):
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
        if not ddp or rank == 0:
            print("Starting per-class analysis...")
        
        # Clear GPU memory before per-class analysis
        if device == "cuda":
            torch.cuda.empty_cache()
        
        subset_size = len(df) if per_class_sample_size <= 0 else min(per_class_sample_size, len(df))
        # Get model's expected frame count from config
        model_num_frames = model.config.num_frames if hasattr(model.config, 'num_frames') else 16
        
        # Prepare data subset for per-class analysis
        # For per-class analysis, ALL ranks need ALL data to compute global class statistics
        per_class_subset = df  # Use full dataset on all ranks
        
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
            df=per_class_subset,
            processor=processor,
            model=model,
            coverages=my_coverages,
            strides=my_strides,
            sample_size=subset_size,
            batch_size=batch_size,  # Use the specified batch size
            num_workers=workers,
            rank=rank,
            num_frames=model_num_frames,  # Pass correct frame count
            checkpoint_path=per_class_out,
            label_map=label_map,
            topk=topk,
            fuzzy_threshold=fuzzy_threshold,
        )
        
        # Gather results from all ranks
        if ddp:
            # All ranks process all data, so they all have the same complete results
            # Only rank 0 needs to save
            gathered_dfs = [None] * world_size
            dist.all_gather_object(gathered_dfs, df_per_class)
            if rank == 0:
                # Use the results from rank 0 (all ranks have identical results)
                df_per_class = gathered_dfs[0]
        
        # Save and log only on rank 0
        if not ddp or rank == 0:
            os.makedirs(os.path.dirname(per_class_out), exist_ok=True)
            df_per_class.to_csv(per_class_out, index=False, float_format='%.6f')
            print(f"✅ Saved per-class results: {len(df_per_class)} rows")

        if wandb is not None and not args.no_wandb and (not ddp or rank == 0):
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
    
    if wandb is not None and not args.no_wandb and (not ddp or rank == 0):
        wandb.finish()

    # Cleanup DDP
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
