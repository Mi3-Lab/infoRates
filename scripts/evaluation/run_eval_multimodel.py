#!/usr/bin/env python3
"""
Multi-Model Evaluation Script

Evaluates temporal sampling effects across multiple video models.
Supports: TimeSformer, VideoMAE, ViViT

Usage:
    python scripts/run_eval_multimodel.py --model timesformer
    python scripts/run_eval_multimodel.py --model videomae
    python scripts/run_eval_multimodel.py --model vivit
    python scripts/run_eval_multimodel.py --model all  # Run all models sequentially
"""

import os
import sys
import argparse
import gc
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from decord import VideoReader, cpu
import yaml

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))
from model_factory import ModelFactory

# ============================================================
# CONFIGURATION
# ============================================================

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ============================================================
# DATA LOADING
# ============================================================

def extract_and_prepare(video_path, model_name, target_coverage, stride, resize=224):
    """Extract and subsample frames from video."""
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        frames = vr.get_batch(np.arange(len(vr))).asnumpy()
        
        # Apply temporal subsampling
        n_total = len(frames)
        n_keep = max(1, int(n_total * target_coverage / 100))
        frames = frames[:n_keep:stride]
        
        # Standardize frame count based on model
        model_info = ModelFactory.get_model_info(model_name)
        target_frames = model_info["default_frames"]
        
        if len(frames) >= target_frames:
            idx = np.linspace(0, len(frames) - 1, target_frames).astype(int)
            frames = [cv2.resize(frames[j], (resize, resize)) for j in idx]
        else:
            # Pad with repeated frames if not enough
            frames = [cv2.resize(f, (resize, resize)) for f in frames]
            while len(frames) < target_frames:
                frames.append(frames[-1])  # Repeat last frame
        
        return frames, True
    except Exception as e:
        print(f"⚠️ Error processing {video_path}: {e}")
        return None, False

# ============================================================
# EVALUATION FUNCTION
# ============================================================

@torch.inference_mode()
def evaluate_model_temporal_sampling(
    model_name,
    df_test,
    processor,
    model,
    coverages=[10, 25, 50, 75, 100],
    strides=[1, 2, 4, 8, 16],
    sample_size=None,
    batch_size=8,
    num_workers=8,
    device="cuda",
    rank=0,
):
    """
    Evaluate model across temporal sampling configurations.
    
    Args:
        model_name: Model identifier ('timesformer', 'videomae', 'vivit')
        df_test: DataFrame with test videos
        processor: Image processor for the model
        model: Loaded model
        coverages: List of coverage percentages
        strides: List of stride values
        sample_size: Number of samples to evaluate (None = all)
        batch_size: Batch size for inference
        num_workers: Number of workers for parallel extraction
        device: Device to run inference on
        rank: Rank for distributed evaluation
    
    Returns:
        DataFrame with results (coverage, stride, accuracy)
    """
    
    subset = df_test.sample(sample_size, random_state=42) if sample_size else df_test
    results = []
    
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name.upper()}")
    print(f"Dataset: {len(subset)} videos")
    print(f"Configurations: {len(coverages)} × {len(strides)} = {len(coverages)*len(strides)}")
    print(f"{'='*70}\n")
    
    for stride in strides:
        for cov in coverages:
            correct = total = 0
            t0 = time.time()
            
            # Parallel frame extraction
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = [
                    ex.submit(extract_and_prepare, row["video_path"], model_name, cov, stride)
                    for _, row in subset.iterrows()
                ]
                
                batch_frames, batch_labels = [], []
                
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"{model_name} | cov={cov:3d}% | stride={stride}",
                    disable=(rank != 0),
                ):
                    frames, success = fut.result()
                    if not success or frames is None:
                        continue
                    
                    # Get label from original dataframe
                    idx = list(subset.iterrows()).__len__() - len(list(as_completed(futures))) - 1
                    label = subset.iloc[idx % len(subset)]["label"]
                    
                    batch_frames.append(frames)
                    batch_labels.append(label)
                    
                    # Batch inference
                    if len(batch_frames) == batch_size:
                        with torch.amp.autocast('cuda', dtype=torch.float16):
                            inputs = processor(batch_frames, return_tensors="pt").to(device)
                            logits = model(**inputs).logits
                        
                        preds = [
                            model.config.id2label[int(i)]
                            for i in logits.argmax(-1).cpu().numpy()
                        ]
                        
                        for p, l in zip(preds, batch_labels):
                            if str(p).lower() == str(l).lower():
                                correct += 1
                        
                        total += len(batch_labels)
                        batch_frames, batch_labels = [], []
                
                # Handle leftover batch
                if batch_frames:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        inputs = processor(batch_frames, return_tensors="pt").to(device)
                        logits = model(**inputs).logits
                    
                    preds = [
                        model.config.id2label[int(i)]
                        for i in logits.argmax(-1).cpu().numpy()
                    ]
                    
                    for p, l in zip(preds, batch_labels):
                        if str(p).lower() == str(l).lower():
                            correct += 1
                    
                    total += len(batch_labels)
            
            acc = correct / max(1, total)
            elapsed = time.time() - t0
            
            results.append({
                "model": model_name,
                "coverage": cov,
                "stride": stride,
                "accuracy": acc,
                "n_samples": total,
                "elapsed_time": elapsed,
            })
            
            # Memory cleanup
            del batch_frames, batch_labels
            torch.cuda.empty_cache()
    
    return pd.DataFrame(results)

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple models on temporal sampling"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="timesformer",
        choices=["timesformer", "videomae", "vivit", "all"],
        help="Model to evaluate ('all' runs all models sequentially)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of test samples (None = all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/UCF101_data/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to Weights & Biases",
    )
    
    args = parser.parse_args()
    
    # Setup
    config = load_config()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    manifest_path = config.get(
        "test_manifest",
        "data/UCF101_data/manifests/ucf101_50f.csv"
    )
    df_test = pd.read_csv(manifest_path)
    print(f"✓ Loaded {len(df_test)} test videos")
    
    # Determine which models to evaluate
    models_to_eval = (
        ["timesformer", "videomae", "vivit"]
        if args.model == "all"
        else [args.model]
    )
    
    # Evaluate each model
    all_results = []
    
    for model_name in models_to_eval:
        try:
            # Load model and processor
            model, info = ModelFactory.load_model(
                model_name,
                num_labels=101,
                device=args.device,
            )
            processor = ModelFactory.load_processor(model_name)
            
            # Evaluate
            results_df = evaluate_model_temporal_sampling(
                model_name,
                df_test,
                processor,
                model,
                coverages=config.get("eval_coverages", [10, 25, 50, 75, 100]),
                strides=config.get("eval_strides", [1, 2, 4, 8, 16]),
                sample_size=args.sample_size or len(df_test),
                batch_size=args.batch_size,
                device=args.device,
            )
            
            all_results.append(results_df)
            
            # Save per-model results
            output_file = output_dir / f"results_{model_name}.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\n✓ Results saved: {output_file}")
            
            # Cleanup
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ Error evaluating {model_name}: {e}")
            continue
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_file = output_dir / "results_multimodel.csv"
        combined_results.to_csv(combined_file, index=False)
        print(f"\n✓ Combined results: {combined_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY: BEST ACCURACY PER MODEL")
        print("="*70)
        for model_name in models_to_eval:
            model_data = combined_results[combined_results["model"] == model_name]
            best = model_data.loc[model_data["accuracy"].idxmax()]
            print(
                f"{model_name:15s}: {best['accuracy']:.4f} "
                f"(cov={int(best['coverage']):3d}%, stride={int(best['stride'])})"
            )
        print("="*70)
    else:
        print("✗ No results generated")

if __name__ == "__main__":
    main()
