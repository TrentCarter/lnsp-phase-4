"""Loss helpers for vector-native LVM training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    """Weights for auxiliary loss terms."""

    tau: float = 0.07
    mse: float = 0.0
    moment: float = 0.0
    variance: float = 0.0


def _info_nce(pred_cos: torch.Tensor, target_cos: torch.Tensor, tau: float) -> torch.Tensor:
    labels = torch.arange(pred_cos.size(0), device=pred_cos.device)
    logits_y2t = pred_cos @ target_cos.t()
    logits_t2y = target_cos @ pred_cos.t()
    logits_y2t = logits_y2t / tau
    logits_t2y = logits_t2y / tau
    loss_y2t = F.cross_entropy(logits_y2t, labels)
    loss_t2y = F.cross_entropy(logits_t2y, labels)
    return loss_y2t + loss_t2y


def _moment_loss(pred_raw: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
    pred_mean = pred_raw.mean(dim=0)
    target_mean = target_raw.mean(dim=0)
    pred_std = pred_raw.std(dim=0)
    target_std = target_raw.std(dim=0)
    mean_term = (pred_mean - target_mean).pow(2).mean()
    std_term = (pred_std - target_std).pow(2).mean()
    return mean_term + std_term


def _variance_loss(pred_cos: torch.Tensor) -> torch.Tensor:
    std = pred_cos.std(dim=0)
    return torch.relu(1.0 - std).mean()


def compute_losses(
    pred_raw: torch.Tensor,
    pred_cos: torch.Tensor,
    targets: torch.Tensor,
    weights: LossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return total loss and diagnostics."""

    weights = weights or LossWeights()
    target_cos = F.normalize(targets, dim=-1)

    info = _info_nce(pred_cos, target_cos, weights.tau)
    mse = F.mse_loss(pred_raw, targets)
    moment = _moment_loss(pred_raw, targets)
    variance = _variance_loss(pred_cos)

    total = info + weights.mse * mse + weights.moment * moment + weights.variance * variance
    stats = {
        "loss_total": float(total.detach().item()),
        "loss_info": float(info.detach().item()),
        "loss_mse": float(mse.detach().item()),
        "loss_moment": float(moment.detach().item()),
        "loss_variance": float(variance.detach().item()),
    }
    return total, stats
