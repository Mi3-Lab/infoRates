"""
Fine-tune video models with Temporal Robustness Augmentation (TRA).

This script fine-tunes video action recognition models (TimeSformer, VideoMAE, ViViT)
with TRA to improve temporal sampling robustness. Compares baseline (no TRA) vs TRA
training and evaluates robustness across coverage×stride grid.

Usage:
    # Baseline (no TRA)
    python scripts/train_with_tra.py --model timesformer --epochs 5 --tra-mode baseline
    
    # With TRA
    python scripts/train_with_tra.py --model timesformer --epochs 5 --tra-mode tra --p-augment 0.5
    
    # With DDP (multi-GPU)
    torchrun --nproc_per_node=2 scripts/train_with_tra.py --model timesformer --ddp --tra-mode tra
"""

import os
import sys

# Add project paths
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in [_PROJECT_ROOT, _SRC_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
from pathlib import Path
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# Import TRA components
from info_rates.training.temporal_augmentation import (
    TemporalRobustnessAugmentation,
    TRADataset,
    create_tra_dataloaders,
    get_tra_stats,
)


def setup_ddp() -> int:
    """Initialize DDP. Returns local_rank."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    if local_rank == 0:
        print(f"[DDP] Initialized with backend={backend}, world_size={world_size}")
    return local_rank


def cleanup_ddp():
    """Cleanup DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def clear_memory():
    """Force memory cleanup."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_ucf101_split(split: str = "train") -> Tuple[List[str], List[int], List[str]]:
    """
    Load UCF101 split from CSV manifests (50f preprocessed datasets).
    
    Args:
        split: "train" (50f for training), "val" (50f_testset for validation), or "test" (50f_testset for robustness eval)
    
    Returns:
        Tuple of (video_paths, labels, class_names)
    """
    import pandas as pd
    
    # Map split name to CSV file - USE 50f DATASETS (padronizados com 50 frames)
    csv_files = {
        "train": "data/UCF101_data/manifests/ucf101_50f_trainlist01.csv",  # Treino (split oficial)
        "val": "data/UCF101_data/manifests/ucf101_50f_testlist01.csv",  # Teste (split oficial)
        "test": "data/UCF101_data/manifests/ucf101_50f_testlist01.csv",  # Mesmo dataset para eval robustez
    }
    
    manifest_path = csv_files.get(split)
    if manifest_path is None:
        raise ValueError(f"Unknown split: {split}. Choose from: train, val, test")
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    # Load CSV
    df = pd.read_csv(manifest_path)
    
    # Filter only existing videos
    df = df[df['video_path'].apply(os.path.exists)].copy()
    
    # Get unique class names
    class_names = sorted(df['label'].unique().tolist())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    # Extract paths and labels
    video_paths = df['video_path'].tolist()
    labels = [class_to_idx[label] for label in df['label'].tolist()]
    
    print(f"✅ Loaded {len(video_paths)} videos from {split} split ({manifest_path})")
    print(f"   Dataset: {'UCF101_50f' if '50f.csv' in manifest_path else 'UCF101_50f_testset'}")
    print(f"   Classes: {len(class_names)}")
    
    return video_paths, labels, class_names


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    scheduler,
    device: str,
    epoch: int,
    total_epochs: int,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    show_progress: bool = True,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", disable=not show_progress)
    
    for batch_idx, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward with mixed precision
        with torch.amp.autocast('cuda'):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
        
        # Backward
        scaler.scale(loss).backward()
        total_loss += loss.item()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        # Update progress
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
    
    return total_loss / len(train_loader)


def validate(
    model,
    val_loader,
    device: str,
    epoch: int,
    total_epochs: int,
    show_progress: bool = True,
) -> Tuple[float, float]:
    """Validate model. Returns (accuracy, loss)."""
    model.eval()
    correct = total = 0
    val_loss = 0.0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", disable=not show_progress)
    
    with torch.no_grad():
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast('cuda'):
                outputs = model(**batch)
                logits = outputs.logits
                loss = outputs.loss
            
            val_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])
            
            # Update progress
            acc = correct / total
            pbar.set_postfix({"acc": f"{acc:.4f}"})
    
    accuracy = correct / total
    avg_loss = val_loss / len(val_loader)
    
    return accuracy, avg_loss


def evaluate_robustness(
    model,
    video_paths: List[str],
    labels: List[int],
    processor,
    device: str,
    coverage_values: List[int] = None,
    stride_values: List[int] = None,
    batch_size: int = 8,
    num_frames: int = 8,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model robustness across coverage×stride grid.
    
    Returns:
        Dictionary mapping (coverage, stride) -> accuracy
    """
    coverage_values = coverage_values or [25, 50, 75, 100]
    stride_values = stride_values or [1, 2, 4, 8, 16]
    
    results = {}
    
    for coverage in coverage_values:
        for stride in stride_values:
            # Create test dataset with fixed coverage/stride
            tra = TemporalRobustnessAugmentation(
                coverage_range=[coverage],
                stride_range=[stride],
                mode="train",  # Use fixed values (single-item ranges)
                p_augment=1.0,  # Always apply the specified coverage/stride
            )
            
            dataset = TRADataset(
                video_paths=video_paths,
                labels=labels,
                processor=processor,
                num_frames=num_frames,
                tra=tra,
            )
            
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
            )
            
            # Evaluate
            model.eval()
            correct = total = 0
            
            desc = f"Eval cov={coverage}%, stride={stride}"
            pbar = tqdm(loader, desc=desc, disable=not show_progress, leave=False)
            
            with torch.no_grad():
                for batch in pbar:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    with torch.amp.autocast('cuda'):
                        outputs = model(**batch)
                        logits = outputs.logits
                    
                    preds = logits.argmax(dim=-1)
                    correct += (preds == batch["labels"]).sum().item()
                    total += len(batch["labels"])
            
            accuracy = correct / total
            results[f"cov{coverage}_stride{stride}"] = accuracy
            
            if show_progress:
                print(f"  {desc}: {accuracy:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune with Temporal Robustness Augmentation")
    
    # Model settings
    parser.add_argument("--model", type=str, default="timesformer", choices=["timesformer", "videomae", "vivit"])
    parser.add_argument("--model-id", type=str, default=None, help="Hugging Face model ID (optional)")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    
    # TRA settings
    parser.add_argument("--tra-mode", type=str, default="tra", choices=["baseline", "tra"])
    parser.add_argument("--p-augment", type=float, default=0.5, help="TRA augmentation probability")
    parser.add_argument("--coverage-range", type=int, nargs="+", default=[25, 50, 75, 100])
    parser.add_argument("--stride-range", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    
    # Data settings
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # System settings
    parser.add_argument("--ddp", action="store_true", help="Use Distributed Data Parallel")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    # Output settings
    parser.add_argument("--save-dir", type=str, default="fine_tuned_models/tra_experiments")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--eval-robustness", action="store_true", help="Evaluate robustness grid")
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    
    # DDP setup
    if args.ddp:
        local_rank = setup_ddp()
        is_main_process = (local_rank == 0)
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    else:
        local_rank = 0
        is_main_process = True
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    show_progress = is_main_process
    
    # Model ID defaults
    model_ids = {
        "timesformer": "facebook/timesformer-base-finetuned-k400",
        "videomae": "MCG-NJU/videomae-base-finetuned-kinetics",
        "vivit": "google/vivit-b-16x2-kinetics400",
    }
    model_id = args.model_id or model_ids[args.model]
    
    if is_main_process:
        print(f"\n{'='*70}")
        print(f"Fine-tuning {args.model.upper()} with TRA")
        print(f"{'='*70}")
        print(f"Model ID: {model_id}")
        print(f"TRA Mode: {args.tra_mode}")
        if args.tra_mode == "tra":
            print(f"Augmentation Probability: {args.p_augment}")
            print(f"Coverage Range: {args.coverage_range}")
            print(f"Stride Range: {args.stride_range}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Learning Rate: {args.lr}")
        print(f"Device: {device}")
        print(f"{'='*70}\n")
    
    # Load data
    if is_main_process:
        print("Loading UCF101 data (50f preprocessed datasets)...")
    
    train_paths, train_labels, class_names = load_ucf101_split("train")
    val_paths, val_labels, _ = load_ucf101_split("val")
    num_classes = len(class_names)

    # Guard against train/val leakage due to overlapping clips
    train_bases = {Path(p).name for p in train_paths}
    val_bases = {Path(p).name for p in val_paths}
    overlap = train_bases.intersection(val_bases)
    if overlap:
        raise RuntimeError(
            f"Train/val overlap detected: {len(overlap)} clips share the same filename. "
            "Check that your 50f manifests were built with the official split lists."
        )
    
    if is_main_process:
        print(f"\n📊 Dataset Summary:")
        print(f"   Training: {len(train_paths)} videos (UCF101_50f - 50 frames each)")
        print(f"   Validation: {len(val_paths)} videos (UCF101_50f_testset - 50 frames each)")
        print(f"   Classes: {num_classes}")
        print(f"   ⚠️  Robustness eval will use SAME testset (independent from training)\n")
    
    # Load processor and model
    if is_main_process:
        print(f"Loading model: {model_id}")
    
    from transformers import AutoImageProcessor, AutoModelForVideoClassification
    
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForVideoClassification.from_pretrained(
        model_id,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)
    
    # Wrap with DDP if needed
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.startswith("cuda") else None,
        )
    
    # Create dataloaders with/without TRA
    if args.tra_mode == "tra":
        # With TRA
        train_loader, val_loader = create_tra_dataloaders(
            train_paths=train_paths,
            train_labels=train_labels,
            val_paths=val_paths,
            val_labels=val_labels,
            processor=processor,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            size=args.size,
            coverage_range=args.coverage_range,
            stride_range=args.stride_range,
            p_augment=args.p_augment,
            num_workers=args.num_workers,
        )
        
        # Print TRA statistics
        if is_main_process:
            tra_temp = TemporalRobustnessAugmentation(
                coverage_range=args.coverage_range,
                stride_range=args.stride_range,
                mode="train",
                p_augment=args.p_augment,
            )
            stats = get_tra_stats(tra_temp, n_samples=1000)
            print("\n=== TRA Statistics (1000 samples) ===")
            print(f"Mean Coverage: {stats['mean_coverage']:.1f}%")
            print(f"Std Coverage: {stats['std_coverage']:.1f}")
            print(f"Mean Stride: {stats['mean_stride']:.2f}")
            print(f"Std Stride: {stats['std_stride']:.2f}")
            print(f"Augmentation Rate: {stats['augmentation_rate']:.2%}\n")
    else:
        # Baseline: no TRA (coverage=100%, stride=1 always)
        train_loader, val_loader = create_tra_dataloaders(
            train_paths=train_paths,
            train_labels=train_labels,
            val_paths=val_paths,
            val_labels=val_labels,
            processor=processor,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            size=args.size,
            coverage_range=[100],  # No coverage reduction
            stride_range=[1],      # No stride
            p_augment=0.0,         # No augmentation
            num_workers=args.num_workers,
        )
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')
    
    num_training_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=num_training_steps,
    )
    
    # WandB init
    if args.wandb and is_main_process:
        wandb.init(
            project="infoRates",
            name=f"{args.model}_tra{args.tra_mode}",
            config=vars(args),
        )
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        if args.ddp and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            show_progress=show_progress,
        )
        
        # Validate
        val_acc, val_loss = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            show_progress=show_progress,
        )
        
        if is_main_process:
            print(f"✅ Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            
            # Log to WandB
            if args.wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                })
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                save_path = Path(args.save_dir) / args.tra_mode / args.model
                save_path.mkdir(parents=True, exist_ok=True)
                
                model_to_save = model.module if args.ddp else model
                model_to_save.save_pretrained(save_path)
                processor.save_pretrained(save_path)
                
                print(f"💾 Saved best model to {save_path}")
    
    if is_main_process:
        print(f"\n🎉 Training complete! Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch+1}")
    
    # Robustness evaluation - LOAD SEPARATE TESTSET (não usar val_paths!)
    if args.eval_robustness and is_main_process:
        print("\n=== Evaluating Robustness Across Coverage×Stride Grid ===")
        print("⚠️  Loading SEPARATE test dataset for robustness evaluation (no data leakage)")
        
        # Load dedicated test set (50f_testset) - INDEPENDENT from validation
        test_paths, test_labels, _ = load_ucf101_split("test")
        print(f"✅ Using {len(test_paths)} test videos for robustness evaluation")
        
        model_to_eval = model.module if args.ddp else model
        
        robustness_results = evaluate_robustness(
            model=model_to_eval,
            video_paths=test_paths,  # ← CHANGE: use test_paths instead of val_paths
            labels=test_labels,       # ← CHANGE: use test_labels instead of val_labels
            processor=processor,
            device=device,
            coverage_values=args.coverage_range,
            stride_values=args.stride_range,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            show_progress=True,
        )
        
        # Save results
        results_path = Path(args.save_dir) / args.tra_mode / f"robustness_{args.model}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(robustness_results, f, indent=2)
        
        print(f"\n💾 Saved robustness results to {results_path}")
        
        # Log to WandB
        if args.wandb:
            wandb.log({"robustness": robustness_results})
    
    # Cleanup
    if args.wandb and is_main_process:
        wandb.finish()
    
    if args.ddp:
        cleanup_ddp()
    
    clear_memory()


if __name__ == "__main__":
    main()
