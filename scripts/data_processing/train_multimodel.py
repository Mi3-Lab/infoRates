"""
Unified fine-tuning script for multiple video models (TimeSformer, VideoMAE, ViViT).

Supports:
- DDP (Distributed Data Parallel) for multi-GPU training
- Mixed precision (fp16) for memory efficiency
- Gradient accumulation to handle larger batches
- Memory cleanup and optimization
- W&B logging for experiment tracking

Usage:
    # Fine-tune single model
    python scripts/train_multimodel.py --model videomae --epochs 5
    
    # Fine-tune all models sequentially
    python scripts/train_multimodel.py --model all --epochs 5
    
    # With DDP (multi-GPU)
    torchrun --nproc_per_node=2 scripts/train_multimodel.py --model vivit --ddp --epochs 5
"""

# Ensure project root and src/ are on sys.path for direct script execution
import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_PROJECT_ROOT, _SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
import yaml
import gc
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from info_rates.data.ucf101 import list_classes, train_val_test_split
from info_rates.models.timesformer import UCFDataset, build_dataloaders, save_model
from scripts.model_factory import ModelFactory


def setup_ddp() -> int:
    """Initialize DDP environment. Returns local_rank."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    if local_rank == 0:
        print(f"[DDP] Initialized with world_size={world_size}, local_rank={local_rank}")
    return local_rank


def cleanup_ddp():
    """Cleanup DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def clear_memory():
    """Force memory cleanup to prevent leaks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fine_tune_model(
    train_dl,
    val_dl,
    num_classes: int,
    model_name: str,
    model_id: str,
    epochs: int = 2,
    lr: float = 1e-5,
    device: str = "cuda",
    use_wandb: bool = True,
    use_ddp: bool = False,
    local_rank: int = 0,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    cleanup_interval: int = 0,
) -> torch.nn.Module:
    """
    Fine-tune a video classification model with DDP support and memory optimization.
    
    Args:
        train_dl: Training DataLoader
        val_dl: Validation DataLoader
        num_classes: Number of action classes
        model_name: Model name (timesformer|videomae|vivit)
        model_id: Hugging Face model ID
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to use (cuda or cpu)
        use_wandb: Whether to use Weights & Biases
        use_ddp: Whether to use Distributed Data Parallel
        local_rank: Local rank for DDP
        gradient_accumulation_steps: Gradient accumulation for larger effective batch
        max_grad_norm: Max gradient norm for clipping
    
    Returns:
        Trained model (unwrapped from DDP if needed)
    """
    
    show_progress = (local_rank == 0)
    is_main_process = (local_rank == 0)
    
    # Set device for DDP
    if use_ddp:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    
    # Load model using ModelFactory
    model_info = ModelFactory.get_model_info(model_name)
    model, info = ModelFactory.load_model(
        model_name,
        num_labels=num_classes,
        device=device
    )
    
    if show_progress:
        print(f"\n{'='*70}")
        print(f"Fine-tuning {model_name.upper()}")
        print(f"{'='*70}")
        print(f"Model ID: {model_id}")
        print(f"Num Classes: {num_classes}")
        print(f"Learning Rate: {lr}")
        print(f"Epochs: {epochs}")
        print(f"Device: {device}")
        print(f"DDP: {use_ddp}")
        print(f"Gradient Accumulation: {gradient_accumulation_steps}")
        print(f"{'='*70}\n")
    
    # Wrap model with DDP if needed
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=True,
        )
    
    # Use mixed precision for memory efficiency
    scaler = torch.cuda.amp.GradScaler()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    num_training_steps = len(train_dl) * epochs // gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=num_training_steps
    )
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        # Set epoch for DistributedSampler to ensure proper shuffling
        if use_ddp and hasattr(train_dl.sampler, 'set_epoch'):
            train_dl.sampler.set_epoch(epoch)
        
        # ===== TRAINING PHASE =====
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        if show_progress:
            train_pbar = tqdm(
                train_dl,
                desc=f"Epoch {epoch+1}/{epochs} [Train]",
                disable=not show_progress
            )
        else:
            train_pbar = train_dl
        
        for batch_idx, batch in enumerate(train_pbar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps  # Scale loss
            
            # Backward pass
            scaler.scale(loss).backward()
            total_loss += loss.item()
            
            # Gradient accumulation step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                # Optional periodic memory cleanup (can be costly if too frequent)
                if cleanup_interval > 0 and ((batch_idx + 1) % cleanup_interval == 0):
                    clear_memory()
            
            # Update progress bar
            if show_progress:
                avg_loss = total_loss / (batch_idx + 1)
                train_pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
        
        avg_train_loss = total_loss / max(1, len(train_dl))
        
        # Synchronize loss across GPUs
        if use_ddp:
            loss_tensor = torch.tensor(avg_train_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_train_loss = loss_tensor.item()
        
        if show_progress:
            print(f"✅ Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # ===== VALIDATION PHASE =====
        model.eval()
        correct = total = 0
        val_loss = 0.0
        
        if show_progress:
            val_pbar = tqdm(
                val_dl,
                desc=f"Epoch {epoch+1}/{epochs} [Val]",
                disable=not show_progress
            )
        else:
            val_pbar = val_dl
        
        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    logits = outputs.logits
                    val_loss += outputs.loss.item()
                
                # Compute accuracy
                preds = logits.argmax(-1).cpu()
                labels = batch["labels"].cpu()
                correct += (preds == labels).sum().item()
                total += len(labels)
                
                # Update progress bar
                if show_progress:
                    current_acc = correct / max(1, total)
                    val_pbar.set_postfix({"acc": f"{current_acc:.3f}"})
                
                # Avoid per-batch cleanup in validation; defer to end of phase
        
        # Single cleanup at end of validation phase
        clear_memory()

        val_acc = correct / max(1, total)
        avg_val_loss = val_loss / max(1, len(val_dl))
        
        # Synchronize metrics across GPUs
        if use_ddp:
            metrics = torch.tensor(
                [val_acc, avg_val_loss, float(correct), float(total)],
                device=device
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            val_acc = metrics[2].item() / metrics[3].item()
            avg_val_loss = metrics[1].item() / dist.get_world_size()
        
        if show_progress:
            print(f"✅ Epoch {epoch+1}/{epochs} - Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.3f}\n")
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
        
        # Log to WandB (only on main process)
        if use_wandb and is_main_process:
            try:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": scheduler.get_last_lr()[0],
                })
            except Exception as e:
                if show_progress:
                    print(f"⚠️  WandB logging failed: {e}")
    
    if show_progress:
        print(f"\n🎯 Best Val Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
        print(f"{'='*70}\n")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune video models (TimeSformer, VideoMAE, ViViT) on UCF101.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune single model
  python scripts/train_multimodel.py --model videomae --epochs 5
  
  # Fine-tune all models
  python scripts/train_multimodel.py --model all --epochs 5
  
  # With DDP (multi-GPU)
  torchrun --nproc_per_node=2 scripts/train_multimodel.py --model vivit --ddp
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="timesformer",
        choices=["timesformer", "videomae", "vivit", "all"],
        help="Which model(s) to fine-tune"
    )
    parser.add_argument(
        "--video-root",
        type=str,
        help="Root folder of UCF101 videos"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save models"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        help="WandB run name"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging"
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Use Distributed Data Parallel"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (for larger effective batch)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--cleanup-interval",
        type=int,
        help="Clean CUDA/CPU memory every N steps (0 disables per-step cleanup)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Helper to fetch either new train_* keys or legacy keys
    def _cfg(name, legacy=None, default=None, cast=lambda x: x):
        if name in config:
            return cast(config[name])
        if legacy and legacy in config:
            return cast(config[legacy])
        return default
    
    # Merge CLI args with config (CLI takes precedence)
    video_root = args.video_root or _cfg("train_video_root", "video_root", "data/UCF101_data/UCF-101")
    epochs = args.epochs if args.epochs is not None else int(_cfg("train_epochs", "epochs", 2, int))
    batch_size = args.batch_size if args.batch_size is not None else int(_cfg("train_batch_size", "batch_size", 4, int))
    lr = args.lr if args.lr is not None else float(_cfg("train_learning_rate", "learning_rate", 1e-5, float))
    save_path = args.save_path or _cfg("train_save_path", "save_path", "fine_tuned_models")
    wandb_project = args.wandb_project or _cfg("train_wandb_project", "wandb_project", "inforates-ucf101")
    disable_wandb = args.no_wandb or bool(_cfg("train_disable_wandb", "disable_wandb", False, bool))
    use_ddp = args.ddp or bool(_cfg("train_use_ddp", "use_ddp", False, bool))
    num_workers = int(_cfg("train_num_workers", "num_workers", 4, int))
    pin_memory = bool(_cfg("train_pin_memory", "pin_memory", True, bool))
    gradient_accumulation_steps = args.gradient_accumulation_steps or int(_cfg("train_gradient_accumulation_steps", None, 1, int))
    device = args.device or _cfg("train_device", None, "cuda")
    cleanup_interval = (args.cleanup_interval if args.cleanup_interval is not None
                        else int(_cfg("train_cleanup_interval", None, 0, int)))
    
    # Setup DDP if requested
    local_rank = 0
    if use_ddp:
        local_rank = setup_ddp()
        torch.cuda.set_device(local_rank)
        device = "cuda"
    
    is_main_process = (local_rank == 0)
    
    # Initialize WandB (only on rank 0)
    if not disable_wandb and is_main_process:
        wandb.init(
            project=wandb_project,
            name=args.wandb_run_name or f"multimodel-finetuning-{args.model}",
            config={
                "model": args.model,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            }
        )
    
    # Load class names
    class_names = list_classes(video_root)

    # Use official UCF101 train/test manifests
    import pandas as pd
    train_manifest = "data/UCF101_data/manifests/ucf101_trainlist01.csv"
    test_manifest = "data/UCF101_data/manifests/ucf101_testlist01.csv"
    train_df = pd.read_csv(train_manifest)
    test_df = pd.read_csv(test_manifest)
    train_files = train_df["video_path"].tolist()
    val_files = test_df["video_path"].tolist()

    if is_main_process:
        print(f"Found {len(class_names)} classes")
        print(f"Train samples: {len(train_files)}")
        print(f"Val samples: {len(val_files)}")
    
    # Determine which models to train
    models_to_train = [args.model] if args.model != "all" else ["timesformer", "videomae", "vivit"]
    
    for model_name in models_to_train:
        if is_main_process:
            print(f"\n{'='*70}")
            print(f"Starting fine-tuning for {model_name.upper()}")
            print(f"{'='*70}")
        
        try:
            # Get model info and frame count
            model_info = ModelFactory.get_model_info(model_name)
            num_frames = model_info["default_frames"]
            
            # Load processor for this model
            processor = ModelFactory.load_processor(model_name)
            
            # Build dataloaders with model-specific frame count
            train_dl, val_dl = build_dataloaders(
                train_files=train_files,
                val_files=val_files,
                class_names=class_names,
                processor=processor,
                batch_size=batch_size,
                num_frames=num_frames,
                size=224,
                use_ddp=use_ddp,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            
            # Fine-tune model
            model = fine_tune_model(
                train_dl=train_dl,
                val_dl=val_dl,
                num_classes=len(class_names),
                model_name=model_name,
                model_id=model_info["model_id"],
                epochs=epochs,
                lr=lr,
                device=device,
                use_wandb=not disable_wandb,
                use_ddp=use_ddp,
                local_rank=local_rank,
                gradient_accumulation_steps=gradient_accumulation_steps,
                cleanup_interval=cleanup_interval,
            )
            
            # Save model (only on main process)
            if is_main_process:
                model_to_save = model.module if use_ddp else model
                model_save_dir = os.path.join(save_path, f"fine_tuned_{model_name}_ucf101")
                save_model(model_save_dir, model_to_save, processor, class_names)
                print(f"✅ Saved {model_name} to {model_save_dir}")
            
            # Cleanup memory between models
            del model, train_dl, val_dl, processor
            clear_memory()
            
        except Exception as e:
            if is_main_process:
                print(f"❌ Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
            continue
    
    # Cleanup DDP
    if use_ddp:
        cleanup_ddp()
    
    if not disable_wandb and is_main_process:
        wandb.finish()
    
    if is_main_process:
        print(f"\n{'='*70}")
        print("✅ All fine-tuning complete!")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
