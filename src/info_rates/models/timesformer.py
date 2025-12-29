import os
import cv2
import numpy as np
import torch
import torch.distributed as dist
from typing import List, Tuple
from glob import glob
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVideoClassification
from torch.optim import AdamW


class UCFDataset(Dataset):
    def __init__(self, files, processor, num_frames: int = 8, size: int = 224, class_names: List[str] = None):
        self.files = files
        self.processor = processor
        self.num_frames = num_frames
        self.size = size
        self.class_names = class_names or []
        self.label2id = {l: i for i, l in enumerate(self.class_names)} if self.class_names else None
        
        # Check if files is a list of tuples (path, label) or just paths
        self.has_labels = isinstance(self.files[0], tuple) if self.files else False

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.has_labels:
            path, label = self.files[idx]
        else:
            path = self.files[idx]
            label_name = os.path.basename(os.path.dirname(path))
            label = self.label2id[label_name] if self.label2id else 0

        if not os.path.exists(path):
            # File was deleted or missing, raise error to skip sample
            raise FileNotFoundError(f"Video file not found: {path}")
        else:
            try:
                vr = VideoReader(path, ctx=cpu(0))
                total = len(vr)
                idxs = np.linspace(0, max(total - 1, 0), self.num_frames).astype(int)
                frames = vr.get_batch(idxs).asnumpy()
            except Exception as e:
                # If video is corrupted or unreadable, delete it from the folder and raise error
                os.remove(path)
                raise RuntimeError(f"Corrupted video removed: {path} ({e})")

        frames = [cv2.resize(f, (self.size, self.size)) for f in frames]

        inputs = self.processor(frames, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(label)
        return inputs


def load_pretrained_timesformer(model_id: str = "facebook/timesformer-base-finetuned-k400", device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForVideoClassification.from_pretrained(model_id).to(device)
    return processor, model


def build_dataloaders(
    train_files: List[str],
    val_files: List[str],
    class_names: List[str],
    processor,
    batch_size: int = 4,
    num_frames: int = 8,
    size: int = 224,
    use_ddp: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
):
    train_ds = UCFDataset(train_files, processor, num_frames=num_frames, size=size, class_names=class_names)
    val_ds = UCFDataset(val_files, processor, num_frames=num_frames, size=size, class_names=class_names)
    
    if use_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
    else:
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
    
    return train_dl, val_dl


def fine_tune_timesformer(train_dl: DataLoader, val_dl: DataLoader, num_classes: int, model_id: str = "facebook/timesformer-base-finetuned-k400", epochs: int = 2, lr: float = 1e-5, device: str = None, use_wandb: bool = True, use_ddp: bool = False, local_rank: int = 0):
    from tqdm import tqdm
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set device for DDP
    if use_ddp:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    
    model = AutoModelForVideoClassification.from_pretrained(
        model_id,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)
    
    # Wrap model with DDP
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Only show progress bars on rank 0
    show_progress = (local_rank == 0)

    for epoch in range(epochs):
        # Set epoch for DistributedSampler
        if use_ddp and hasattr(train_dl.sampler, 'set_epoch'):
            train_dl.sampler.set_epoch(epoch)
        
        # Training phase with progress bar
        model.train()
        total_loss = 0.0
        
        if show_progress:
            train_pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=True)
        else:
            train_pbar = train_dl
        
        for batch_idx, batch in enumerate(train_pbar):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            
            # Update progress bar with current loss
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

        # Validation phase with progress bar
        model.eval()
        correct = total = 0
        val_loss = 0.0
        
        if show_progress:
            val_pbar = tqdm(val_dl, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=True)
        else:
            val_pbar = val_dl
        
        with torch.no_grad():
            for batch in val_pbar:
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                logits = outputs.logits
                val_loss += outputs.loss.item() if hasattr(outputs, 'loss') else 0
                preds = logits.argmax(-1).cpu()
                labels = batch["labels"]
                correct += (preds == labels).sum().item()
                total += len(labels)
                
                # Update progress bar with current accuracy
                if show_progress:
                    current_acc = correct / max(1, total)
                    val_pbar.set_postfix({"acc": f"{current_acc:.3f}"})
        
        val_acc = correct / max(1, total)
        avg_val_loss = val_loss / max(1, len(val_dl))
        
        # Synchronize metrics across GPUs
        if use_ddp:
            metrics = torch.tensor([val_acc, avg_val_loss, correct, total], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            val_acc = metrics[2].item() / metrics[3].item()
            avg_val_loss = metrics[1].item() / dist.get_world_size()
        
        if show_progress:
            print(f"✅ Epoch {epoch+1}/{epochs} - Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.3f}\n")
        
        # Log to WandB (only rank 0)
        if use_wandb and show_progress:
            try:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_acc,
                })
            except ImportError:
                pass

    return model


def save_model(save_path: str, model, processor, class_names: List[str]):
    os.makedirs(save_path, exist_ok=True)
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in enumerate(class_names)}
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.config.num_labels = len(class_names)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Saved model to {save_path} with {len(class_names)} classes.")
