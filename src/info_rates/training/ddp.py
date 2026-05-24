"""DDP setup/teardown utilities shared by all ACCV training scripts."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_ddp() -> int:
    """Initialize DDP environment. Returns local_rank."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    if local_rank == 0:
        print(f"[DDP] Initialized with backend={backend}, world_size={world_size}, local_rank={local_rank}")
    return local_rank


def cleanup_ddp() -> None:
    """Tear down DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
