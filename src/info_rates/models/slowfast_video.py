"""SlowFast R50 baseline via PyTorchVideo for ACCV 2026 non-transformer experiments."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


# SlowFast R50 fixed temporal requirements: 8 slow frames + 32 fast frames.
# The budget (evidence frames decoded from video) is adapted to these sizes.
SLOWFAST_SLOW_FRAMES = 8
SLOWFAST_FAST_FRAMES = 32
SLOWFAST_ALPHA = SLOWFAST_FAST_FRAMES // SLOWFAST_SLOW_FRAMES  # 4


def _resample_to(tensor: torch.Tensor, target_t: int) -> torch.Tensor:
    """Evenly resample or pad a C,T,H,W tensor to target_t frames."""
    T = tensor.shape[1]
    if T == target_t:
        return tensor
    idx = torch.linspace(0, T - 1, target_t).long()
    return tensor[:, idx, :, :]


class SlowFastBatch(dict):
    """Dict-based batch compatible with `model(**inputs)` and `.to(device)`."""

    def to(self, device):
        return SlowFastBatch({k: v.to(device) if torch.is_tensor(v) else v for k, v in self.items()})


class SlowFastVideoProcessor:
    """Convert RGB numpy frames to slow+fast pathway tensors for SlowFast R50.

    Always outputs exactly SLOWFAST_SLOW_FRAMES slow frames and
    SLOWFAST_FAST_FRAMES fast frames, resampled from whatever evidence
    frames are passed in. This matches the architecture's fixed pooling kernel.
    """

    def __init__(
        self,
        size: int = 224,
        mean: tuple[float, float, float] = (0.45, 0.45, 0.45),
        std: tuple[float, float, float] = (0.225, 0.225, 0.225),
    ):
        self.size = size
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def _frames_to_tensor(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Stack list of HWC uint8 frames → C,T,H,W float tensor."""
        arr = np.stack(frames, axis=0).astype("float32") / 255.0  # T,H,W,C
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)  # C,T,H,W
        return (tensor - self.mean) / self.std

    def __call__(self, videos, return_tensors: str = "pt") -> SlowFastBatch:
        if videos and isinstance(videos[0], np.ndarray):
            videos = [videos]

        slow_list, fast_list = [], []
        for frames in videos:
            t = self._frames_to_tensor(frames)  # C,T,H,W
            fast_list.append(_resample_to(t, SLOWFAST_FAST_FRAMES))
            slow_list.append(_resample_to(t, SLOWFAST_SLOW_FRAMES))

        return SlowFastBatch(
            {
                "slow_frames": torch.stack(slow_list, dim=0),  # B,C,8,H,W
                "fast_frames": torch.stack(fast_list, dim=0),  # B,C,32,H,W
            }
        )


class SlowFastVideoClassifier(nn.Module):
    """Wraps PyTorchVideo SlowFast with a HF-like `.logits` interface."""

    def __init__(self, model: nn.Module, num_labels: int):
        super().__init__()
        self.model = model
        self.config = SimpleNamespace(
            num_labels=num_labels,
            id2label={i: str(i) for i in range(num_labels)},
        )

    def forward(self, slow_frames: torch.Tensor, fast_frames: torch.Tensor):
        # PyTorchVideo SlowFast expects a list [slow, fast]
        out = self.model([slow_frames, fast_frames])
        return SimpleNamespace(logits=out)


def create_slowfast_model(num_labels: int, pretrained: bool = True) -> SlowFastVideoClassifier:
    from pytorchvideo.models import create_slowfast

    model = create_slowfast(
        input_channels=(3, 3),
        model_depth=50,
        model_num_class=400,  # load with Kinetics-400 head
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
    )

    if pretrained:
        try:
            import torch.hub as hub

            state = hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST_8x8_R50.pyth",
                map_location="cpu",
                check_hash=False,
            )
            # PyTorchVideo checkpoint is wrapped under 'model_state'
            sd = state.get("model_state", state)
            model.load_state_dict(sd, strict=False)
            print("[SlowFast] Loaded Kinetics-400 pretrained weights")
        except Exception as e:
            print(f"[SlowFast] WARNING: Could not load pretrained weights: {e}")

    # Make spatial pooling resolution-agnostic (hardcoded 7×7 only works at 224px)
    # Replace AvgPool3d(kernel=[T,7,7]) with AdaptiveAvgPool3d(1) — works at any resolution
    pool_block = model.blocks[5]
    if hasattr(pool_block, "pool"):
        pool_block.pool[0] = nn.AdaptiveAvgPool3d(1)
        pool_block.pool[1] = nn.AdaptiveAvgPool3d(1)

    # Replace classification head with target num_labels
    # PyTorchVideo SlowFast head is model.blocks[-1].proj
    head = model.blocks[-1]
    if hasattr(head, "proj"):
        in_features = head.proj.in_features
        head.proj = nn.Linear(in_features, num_labels)
    else:
        # Fallback: find the last Linear layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear) and module.out_features == 400:
                in_features = module.in_features
                parent = model
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], nn.Linear(in_features, num_labels))
                break

    return SlowFastVideoClassifier(model, num_labels=num_labels)


def save_slowfast_checkpoint(
    save_dir: str | Path,
    model: SlowFastVideoClassifier,
    class_names: list[str],
    num_frames: int,
    input_size: int,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.model.state_dict(), save_dir / "model.pt")
    config = {
        "backend": "slowfast_video",
        "model_name": "slowfast_r50",
        "num_labels": len(class_names),
        "class_names": [str(c) for c in class_names],
        "num_frames": num_frames,
        "input_size": input_size,
        "alpha": SLOWFAST_ALPHA,
    }
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_slowfast_checkpoint(checkpoint: str | Path, device: str = "cuda"):
    checkpoint = Path(checkpoint)
    with open(checkpoint / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("backend") != "slowfast_video":
        raise ValueError(f"Not a SlowFast checkpoint: {checkpoint}")
    model = create_slowfast_model(num_labels=int(config["num_labels"]), pretrained=False)
    state = torch.load(checkpoint / "model.pt", map_location="cpu")
    model.model.load_state_dict(state)
    model.to(device)
    model.eval()
    processor = SlowFastVideoProcessor(size=int(config.get("input_size", 224)))
    return model, processor, config
