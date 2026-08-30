"""Composite training loss for EVA.

L = L_cls + lambda_KL L_KL + lambda_evid L_evid
  + lambda_phase L_phase + lambda_ent L_ent
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EVALoss(nn.Module):
    """Training objective for stride-robust EVA models.

    Args:
        lambda_kl: weight for dense-to-sparse KL distillation.
        lambda_evid: weight for evidence-alignment loss.
        lambda_phase: weight for sampling-phase consistency loss.
        lambda_ent: weight for target-entropy regularisation.
        target_events: expected useful subevents per window. ``1`` gives
            near-selector behavior; larger values allow multi-event windows.
        T_distill: distillation temperature.
        lambda_sp: deprecated alias for ``lambda_ent`` kept for older scripts.
    """

    def __init__(
        self,
        lambda_kl: float = 1.0,
        lambda_evid: float = 0.5,
        lambda_phase: float = 0.5,
        lambda_ent: float = 0.1,
        target_events: float = 1.0,
        T_distill: float = 3.0,
        lambda_sp: float | None = None,
    ):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_evid = lambda_evid
        self.lambda_phase = lambda_phase
        self.lambda_ent = lambda_ent if lambda_sp is None else lambda_sp
        self.target_events = max(float(target_events), 1.0)
        self.T = T_distill

    def forward(
        self,
        out: dict[str, Tensor | None],
        labels: Tensor,
        s: int,
        out_phase2: dict[str, Tensor | None] | None = None,
    ) -> dict[str, Tensor]:
        """Compute the loss dictionary.

        ``out`` is the dictionary returned by ``EVAWrapper``. If it contains an
        ``evidence`` tensor of shape ``(B, T)`` or ``(B, T, 1)``, that map is
        used as the dense evidence teacher. Otherwise the dense s=1 EVA event
        weights are used as a lightweight fallback.
        """
        losses: dict[str, Tensor] = {}

        logits = out["logits"]
        if logits is None:
            raise ValueError("EVALoss requires out['logits']")

        losses["cls"] = F.cross_entropy(logits, labels)

        if s > 1 and out.get("logits_dense") is not None:
            T = self.T
            p_dense = F.softmax(out["logits_dense"].detach() / T, dim=-1)
            lp_sparse = F.log_softmax(logits / T, dim=-1)
            losses["kl"] = (
                self.lambda_kl
                * T**2
                * F.kl_div(lp_sparse, p_dense, reduction="batchmean")
            )

        a_evt = out.get("a_evt")
        evidence = out.get("evidence")
        if evidence is None:
            evidence = out.get("a_evt_dense")

        if s > 1 and a_evt is not None and evidence is not None:
            a_sp_flat = a_evt.reshape(a_evt.shape[0], -1)
            a_sp_flat = a_sp_flat / (a_sp_flat.sum(-1, keepdim=True) + 1e-8)

            a_dn_flat = evidence.reshape(evidence.shape[0], -1).detach()
            a_dn_flat = a_dn_flat / (a_dn_flat.sum(-1, keepdim=True) + 1e-8)

            if a_sp_flat.shape != a_dn_flat.shape:
                raise ValueError(
                    "Sparse and dense evidence maps must flatten to the same "
                    f"shape, got {tuple(a_sp_flat.shape)} and {tuple(a_dn_flat.shape)}"
                )

            losses["evid"] = self.lambda_evid * (a_sp_flat - a_dn_flat).abs().mean()

        if out_phase2 is not None:
            logits2 = out_phase2["logits"]
            if logits2 is None:
                raise ValueError("phase loss requires out_phase2['logits']")
            p1 = F.softmax(logits, dim=-1)
            p2 = F.softmax(logits2, dim=-1)
            m = 0.5 * (p1 + p2)
            eps = 1e-8
            js = 0.5 * (
                (p1 * ((p1 + eps).log() - (m + eps).log())).sum(-1).mean()
                + (p2 * ((p2 + eps).log() - (m + eps).log())).sum(-1).mean()
            )
            losses["phase"] = self.lambda_phase * js

        if a_evt is not None:
            entropy = -(a_evt * (a_evt + 1e-8).log()).sum(-1)
            target = torch.log(
                torch.as_tensor(
                    self.target_events,
                    device=a_evt.device,
                    dtype=a_evt.dtype,
                )
            )
            losses["ent"] = self.lambda_ent * (entropy - target).pow(2).mean()

        losses["total"] = sum(losses.values())
        return losses
