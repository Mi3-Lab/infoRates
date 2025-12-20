import argparse
import torch
import torch.distributed as dist
import wandb
import os
import yaml
from transformers import AutoImageProcessor

from info_rates.data.ucf101 import list_classes, train_val_test_split
from info_rates.models.timesformer import build_dataloaders, fine_tune_timesformer, save_model


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_ddp():
    """Initialize DDP environment"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return local_rank


def cleanup_ddp():
    """Cleanup DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TimeSformer on UCF101.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    # Override options (optional, will override config file values)
    parser.add_argument("--video-root", type=str, help="Root folder of UCF101 videos")
    parser.add_argument("--model-id", type=str, help="Model ID")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size per GPU")
    parser.add_argument("--num-frames", type=int, help="Number of frames per clip")
    parser.add_argument("--size", type=int, help="Image size")
    parser.add_argument("--save-path", type=str, help="Path to save model")
    parser.add_argument("--wandb-project", type=str, help="WandB project name")
    parser.add_argument("--wandb-run-name", type=str, help="WandB run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--ddp", action="store_true", help="Use Distributed Data Parallel")
    parser.add_argument("--lr", type=float, help="Learning rate")
    args = parser.parse_args()
    
    # Load config file
    config = load_config(args.config)
    
    # Override config with command-line arguments if provided
    video_root = args.video_root or config.get("video_root", "UCF101_data/UCF-101")
    model_id = args.model_id or config.get("model_id", "facebook/timesformer-base-finetuned-k400")
    epochs = args.epochs if args.epochs is not None else int(config.get("epochs", 2))
    batch_size = args.batch_size if args.batch_size is not None else int(config.get("batch_size", 4))
    num_frames = args.num_frames if args.num_frames is not None else int(config.get("num_frames", 8))
    size = args.size if args.size is not None else int(config.get("size", 224))
    save_path = args.save_path or config.get("save_path", "fine_tuned_timesformer_ucf101")
    wandb_project = args.wandb_project or config.get("wandb_project", "inforates-ucf101")
    wandb_run_name = args.wandb_run_name or config.get("wandb_run_name")
    disable_wandb = args.no_wandb or config.get("disable_wandb", False)
    use_ddp = args.ddp or config.get("use_ddp", False)
    lr = args.lr if args.lr is not None else float(config.get("learning_rate", 1e-5))
    num_workers = int(config.get("num_workers", 4))
    pin_memory = config.get("pin_memory", True)
    
    # Setup DDP if requested
    local_rank = 0
    if use_ddp:
        local_rank = setup_ddp()
        torch.cuda.set_device(local_rank)

    # Initialize WandB (only on rank 0)
    if not disable_wandb and local_rank == 0:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_id": model_id,
                "epochs": epochs,
                "batch_size": batch_size,
                "num_frames": num_frames,
                "size": size,
                "learning_rate": lr,
            }
        )

    class_names = list_classes(video_root)
    train_files, val_files, _ = train_val_test_split(video_root)

    processor = AutoImageProcessor.from_pretrained(model_id)
    train_dl, val_dl = build_dataloaders(
        train_files=train_files,
        val_files=val_files,
        class_names=class_names,
        processor=processor,
        batch_size=batch_size,
        num_frames=num_frames,
        size=size,
        use_ddp=use_ddp,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = fine_tune_timesformer(
        train_dl=train_dl,
        val_dl=val_dl,
        num_classes=len(class_names),
        model_id=model_id,
        epochs=epochs,
        lr=lr,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_wandb=not disable_wandb,
        use_ddp=use_ddp,
        local_rank=local_rank,
    )

    # Save model (only rank 0)
    if local_rank == 0:
        # Unwrap DDP model before saving
        model_to_save = model.module if use_ddp else model
        save_model(save_path, model_to_save, processor, class_names)
    
    if not disable_wandb and local_rank == 0:
        wandb.finish()
    
    # Cleanup DDP
    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
