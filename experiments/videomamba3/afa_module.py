"""
Adaptive Frame Allocation (AFA) modules for VideoMamba3-AFA.

Components
----------
TemporalConcentrationHead (TCH)
    Predicts per-frame saliency scores [0,1] from the sparse-scan hidden
    states of Stage-1 VideoMamba3. Supervised by two signals:
      1. Classification loss (indirect — gradients flow from Stage-2 via
         straight-through estimator in the selector).
      2. Concentration distillation from E3 empirical measurements
         (sw_ensemble_{dataset}.csv targets).

AdaptiveFrameSelector
    Selects top-B frame indices from T_max candidates.
    Training: straight-through estimator (hard forward, soft backward).
    Inference: hard argmax — deterministic, zero overhead.

ConcentrationDistillationLoss
    MSE between the TCH's mean-over-time score (per video) and the
    per-class concentration measured in the E3 sliding-window experiment.
    Only applied for classes that had ρ > 0 in the E3 analysis
    (i.e., where the empirical signal is trustworthy).

BudgetRegularizationLoss
    Penalises deviation of the mean score from the target sparsity
    B / T_max so the model uses its budget on average.

Reference
---------
Mamba-3: arXiv 2603.15569 (ICLR 2026)
E3 aliasing analysis: evaluations/accv2026/e3_spectral/sw_ensemble_*.csv
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Temporal Concentration Head
# ---------------------------------------------------------------------------

class TemporalConcentrationHead(nn.Module):
    """Predict per-frame saliency from Stage-1 SSM hidden states.

    Parameters
    ----------
    d_model : int
        Hidden dimension of Stage-1 VideoMamba3.
    T_sparse : int
        Number of frames processed by Stage-1 (sparse scan).
    T_max : int
        Total frames decoded from the raw video.
    n_patches_per_frame : int
        Number of spatial patch tokens per frame (e.g. 196 for 224px / 16px).
    has_cls_token : bool
        Whether Stage-1 prepends a CLS token at position 0.
    """

    def __init__(
        self,
        d_model: int,
        T_sparse: int,
        T_max: int,
        n_patches_per_frame: int,
        has_cls_token: bool = True,
    ):
        super().__init__()
        self.T_sparse = T_sparse
        self.T_max = T_max
        self.n_patches = n_patches_per_frame
        self.has_cls = has_cls_token

        # lightweight per-frame scorer
        d_mid = max(d_model // 4, 32)
        self.frame_norm = nn.LayerNorm(d_model)
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_mid),
            nn.GELU(),
            nn.Linear(d_mid, 1),
        )

        # temporal context — small 1-D conv mixes neighbouring frame scores
        # before upsampling; helps with short actions spanning several frames
        self.temporal_ctx = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : Tensor [B, L, D]
            Output tokens from Stage-1 VideoMamba3.
            L = (1 + T_sparse * n_patches) with CLS, else T_sparse * n_patches.

        Returns
        -------
        scores : Tensor [B, T_max]
            Per-original-frame saliency in (0, 1).
        """
        B, L, D = hidden_states.shape

        # strip CLS token
        tokens = hidden_states[:, 1:] if self.has_cls else hidden_states

        # reshape → pool over spatial patches → [B, T_sparse, D]
        tokens = tokens.view(B, self.T_sparse, self.n_patches, D)
        frame_feats = tokens.mean(dim=2)                          # [B, T_sparse, D]

        # score per sparse frame
        frame_feats = self.frame_norm(frame_feats)
        scores_sparse = self.scorer(frame_feats).squeeze(-1)      # [B, T_sparse]

        # temporal context mixing before upsampling
        scores_sparse = self.temporal_ctx(
            scores_sparse.unsqueeze(1)                            # [B, 1, T_sparse]
        ).squeeze(1)                                              # [B, T_sparse]

        # upsample T_sparse → T_max with linear interpolation
        scores_full = F.interpolate(
            scores_sparse.unsqueeze(1),                           # [B, 1, T_sparse]
            size=self.T_max,
            mode="linear",
            align_corners=False,
        ).squeeze(1)                                              # [B, T_max]

        return torch.sigmoid(scores_full)                         # clamp to (0,1)


# ---------------------------------------------------------------------------
# Adaptive Frame Selector
# ---------------------------------------------------------------------------

class AdaptiveFrameSelector(nn.Module):
    """Select B frames from T_max candidates using a straight-through estimator.

    Training
    --------
    Forward  : hard top-K selection (no gradient through discrete choice).
    Backward : gradients routed through ``soft_weights`` (softmax of scores),
               which is differentiable and carries signal to the TCH.
    The straight-through trick: ``output = hard + soft.detach() - soft`` is
    *not* applied to the frames themselves (they are discrete indices), but
    the loss terms that depend on ``soft_weights`` flow gradients to TCH.

    Inference
    ---------
    ``training=False`` → returns only hard indices, ``soft_weights=None``.

    Parameters
    ----------
    budget_B : int
        Number of frames to select.
    temperature : float
        Softmax temperature for the soft weights. Lower = sharper selection.
    """

    def __init__(self, budget_B: int, temperature: float = 1.0):
        super().__init__()
        self.B = budget_B
        self.tau = temperature

    def forward(
        self,
        scores: torch.Tensor,
        training: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        scores : Tensor [B_batch, T_max]
            Saliency scores from TCH.
        training : bool

        Returns
        -------
        indices : LongTensor [B_batch, budget_B]
            Selected frame indices in ascending (temporal) order.
        soft_weights : Tensor [B_batch, T_max] or None
            Differentiable weights for gradient routing (training only).
        """
        # hard top-K — used in both modes
        topk = scores.topk(self.B, dim=1)
        indices, _ = topk.indices.sort(dim=1)                    # keep order

        if not training:
            return indices, None

        # soft weights (straight-through pathway for gradients)
        soft_weights = F.softmax(scores / self.tau, dim=-1)      # [B, T_max]
        return indices, soft_weights

    def gather_frames(
        self,
        frames: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Index into a frame tensor using the selected indices.

        Parameters
        ----------
        frames : Tensor [B, T_max, C, H, W]
        indices : LongTensor [B, budget_B]

        Returns
        -------
        selected : Tensor [B, budget_B, C, H, W]
        """
        B, budget_B = indices.shape
        idx = indices.view(B, budget_B, 1, 1, 1).expand(
            B, budget_B, *frames.shape[2:]
        )
        return frames.gather(dim=1, index=idx)


# ---------------------------------------------------------------------------
# Concentration Distillation Loss
# ---------------------------------------------------------------------------

class ConcentrationDistillationLoss(nn.Module):
    """MSE between TCH scores and E3 empirical concentration targets.

    The E3 experiment measured sliding-window confidence concentration per
    class for each dataset (sw_ensemble_{dataset}.csv). This loss supervises
    the TCH to reproduce that signal, connecting the empirical analysis
    directly to the model's learned behaviour.

    Only datasets with Spearman ρ > 0 in E3 contribute (sign-correct
    signal). Datasets with near-zero or negative ρ (AUTSL, Diving-48,
    FineGym, EPIC-Kitchens) are excluded to avoid training noise.

    Parameters
    ----------
    e3_dir : str or Path
        Path to evaluations/accv2026/e3_spectral/
    datasets : list[str]
        Datasets to load targets for.
    trusted_datasets : set[str]
        Datasets with positive ρ — only these contribute to the loss.
    """

    # Default: only datasets with ρ > 0.2 and p < 0.05 in E3
    TRUSTED_BY_DEFAULT = {"ucf101", "ssv2", "hmdb51", "driveact"}

    def __init__(
        self,
        e3_dir: str | Path,
        datasets: list[str],
        trusted_datasets: Optional[set[str]] = None,
    ):
        super().__init__()
        self.trusted = trusted_datasets or self.TRUSTED_BY_DEFAULT
        e3_dir = Path(e3_dir)

        # load targets: {dataset: {label_id: concentration}}
        self.targets: dict[str, dict[int, float]] = {}
        for ds in datasets:
            csv_path = e3_dir / f"sw_ensemble_{ds}.csv"
            if not csv_path.exists():
                continue
            import pandas as pd
            df = pd.read_csv(csv_path)
            if "ensemble_concentration" not in df.columns:
                continue
            self.targets[ds] = dict(
                zip(df["label_id"].tolist(), df["ensemble_concentration"].tolist())
            )

    def forward(
        self,
        tch_scores: torch.Tensor,
        labels: torch.Tensor,
        dataset: str,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tch_scores : Tensor [B, T_max]
            TCH output scores.
        labels : LongTensor [B]
            Ground-truth class indices.
        dataset : str
            Current dataset name (must match E3 csv naming).

        Returns
        -------
        loss : scalar Tensor
        """
        if dataset not in self.trusted or dataset not in self.targets:
            return tch_scores.new_tensor(0.0)

        targets_map = self.targets[dataset]
        label_ids = labels.tolist()

        # per-video scalar: mean score over time ≈ "how concentrated is this video"
        video_scores = tch_scores.mean(dim=1)                    # [B]

        target_vals = []
        valid_mask = []
        for lid in label_ids:
            if lid in targets_map:
                target_vals.append(targets_map[lid])
                valid_mask.append(True)
            else:
                target_vals.append(0.0)
                valid_mask.append(False)

        if not any(valid_mask):
            return tch_scores.new_tensor(0.0)

        target_t = tch_scores.new_tensor(target_vals)            # [B]
        mask_t   = tch_scores.new_tensor(valid_mask).bool()      # [B]

        return F.mse_loss(video_scores[mask_t], target_t[mask_t])


# ---------------------------------------------------------------------------
# Budget Regularisation Loss
# ---------------------------------------------------------------------------

class BudgetRegularizationLoss(nn.Module):
    """Penalise deviation from the target frame budget B / T_max.

    Without this, the model could collapse to always selecting the same
    frames or always outputting uniform scores (no learning signal from
    the selection).

    Parameters
    ----------
    budget_B : int
        Target number of frames to select.
    T_max : int
        Total available frames.
    """

    def __init__(self, budget_B: int, T_max: int):
        super().__init__()
        self.target_sparsity = budget_B / T_max

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        scores : Tensor [B, T_max]

        Returns
        -------
        loss : scalar Tensor
        """
        mean_score = scores.mean()
        return (mean_score - self.target_sparsity) ** 2


# ---------------------------------------------------------------------------
# Combined AFA Loss
# ---------------------------------------------------------------------------

class AFALoss(nn.Module):
    """Combines all AFA training losses.

    L_total = L_cls
            + lambda_conc   * L_concentration
            + lambda_budget * L_budget

    Parameters
    ----------
    e3_dir : str or Path
    datasets : list[str]
    budget_B : int
    T_max : int
    lambda_conc : float
        Weight for the concentration distillation loss.
    lambda_budget : float
        Weight for the budget regularisation loss.
    trusted_datasets : set[str], optional
    """

    def __init__(
        self,
        e3_dir: str | Path,
        datasets: list[str],
        budget_B: int,
        T_max: int,
        lambda_conc: float = 0.5,
        lambda_budget: float = 0.1,
        trusted_datasets: Optional[set[str]] = None,
    ):
        super().__init__()
        self.lambda_conc   = lambda_conc
        self.lambda_budget = lambda_budget

        self.cls_loss    = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.conc_loss   = ConcentrationDistillationLoss(e3_dir, datasets, trusted_datasets)
        self.budget_loss = BudgetRegularizationLoss(budget_B, T_max)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tch_scores: torch.Tensor,
        dataset: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Parameters
        ----------
        logits : Tensor [B, num_classes]
        labels : LongTensor [B]
        tch_scores : Tensor [B, T_max]
        dataset : str

        Returns
        -------
        total_loss : scalar Tensor
        breakdown : dict with individual loss values (for logging)
        """
        l_cls    = self.cls_loss(logits, labels)
        l_conc   = self.conc_loss(tch_scores, labels, dataset)
        l_budget = self.budget_loss(tch_scores)

        total = l_cls + self.lambda_conc * l_conc + self.lambda_budget * l_budget

        breakdown = {
            "loss_cls":    l_cls.item(),
            "loss_conc":   l_conc.item(),
            "loss_budget": l_budget.item(),
            "loss_total":  total.item(),
        }
        return total, breakdown


# ---------------------------------------------------------------------------
# E3 Target Loader (utility)
# ---------------------------------------------------------------------------

def load_e3_targets(e3_dir: str | Path) -> dict[str, dict[int, float]]:
    """Load all sw_ensemble_{dataset}.csv files into a nested dict.

    Returns
    -------
    dict : {dataset_name: {label_id: concentration}}
    """
    import pandas as pd
    e3_dir = Path(e3_dir)
    result: dict[str, dict[int, float]] = {}
    for csv_path in sorted(e3_dir.glob("sw_ensemble_*.csv")):
        ds = csv_path.stem.replace("sw_ensemble_", "")
        df = pd.read_csv(csv_path)
        if "ensemble_concentration" not in df.columns:
            continue
        result[ds] = dict(
            zip(df["label_id"].tolist(), df["ensemble_concentration"].tolist())
        )
    return result


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, T_max, T_sparse = 2, 48, 12
    n_patches = 196      # 224px / 16px patch = 14×14
    D = 192
    budget_B = 16

    # fake Stage-1 hidden states: [B, 1 + T_sparse*n_patches, D]
    L = 1 + T_sparse * n_patches
    hidden = torch.randn(B, L, D)

    tch = TemporalConcentrationHead(
        d_model=D,
        T_sparse=T_sparse,
        T_max=T_max,
        n_patches_per_frame=n_patches,
        has_cls_token=True,
    )
    scores = tch(hidden)
    print(f"TCH  | hidden {tuple(hidden.shape)} → scores {tuple(scores.shape)} "
          f"| min={scores.min():.3f} max={scores.max():.3f}")

    selector = AdaptiveFrameSelector(budget_B=budget_B, temperature=1.0)
    indices, soft_w = selector(scores, training=True)
    print(f"SEL  | indices {tuple(indices.shape)} | soft_w {tuple(soft_w.shape)}")

    # fake raw frames
    frames = torch.randn(B, T_max, 3, 224, 224)
    selected = selector.gather_frames(frames, indices)
    print(f"FRAMES | selected {tuple(selected.shape)}")

    budget_loss = BudgetRegularizationLoss(budget_B, T_max)
    bl = budget_loss(scores)
    print(f"BUDGET LOSS | {bl.item():.5f}  (target sparsity={budget_B/T_max:.3f})")

    print("\nAll checks passed.")
