"""Shared helpers for LVM training experiments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import requests
import torch


def sample_anchors(target_vectors: np.ndarray, num_anchors: int) -> Tuple[torch.Tensor, float]:
    """Return a tensor of anchor vectors and the median RBF sigma."""

    if num_anchors <= 0 or target_vectors.shape[0] == 0:
        raise ValueError("num_anchors must be >0 and target vectors non-empty")

    num_anchors = min(num_anchors, target_vectors.shape[0])
    idx = np.random.choice(target_vectors.shape[0], size=num_anchors, replace=False)
    anchors = target_vectors[idx]
    # Ensure unit norm (defensive)
    anchors = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-8)

    # Median pairwise Euclidean distance as bandwidth heuristic
    diffs = anchors[:, None, :] - anchors[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    median_sigma = float(np.median(dists))
    if not math.isfinite(median_sigma) or median_sigma == 0.0:
        median_sigma = 1.0

    anchor_tensor = torch.from_numpy(anchors.astype(np.float32))
    return anchor_tensor, median_sigma


def compute_mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Mini-batch RBF MMD."""

    if sigma <= 0:
        sigma = 1.0
    gamma = 1.0 / (2 * sigma * sigma)

    def _kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dist_sq = torch.cdist(a, b, p=2).pow(2)
        return torch.exp(-gamma * dist_sq)

    k_xx = _kernel(x, x)
    k_yy = _kernel(y, y)
    k_xy = _kernel(x, y)
    mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return mmd


def compute_batch_stats(target_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return global mean/std statistics for target vectors."""

    mean = target_vectors.mean(axis=0)
    std = target_vectors.std(axis=0)
    std[std < 1e-6] = 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def mean_variance_penalty(
    preds: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    batch_mean = preds.mean(dim=0)
    batch_std = preds.std(dim=0)
    mean_term = (batch_mean - target_mean).pow(2).mean()
    std_term = (batch_std - target_std).pow(2).mean()
    return mean_term + std_term


@dataclass
class CycleConfig:
    pct: float = 0.0
    weight: float = 0.0
    steps: int = 1
    decoder_endpoint: str = "http://127.0.0.1:8766/decode"
    encoder_endpoint: str = "http://127.0.0.1:8767/embed"
    timeout: float = 30.0

    def enabled(self) -> bool:
        return self.pct > 0.0 and self.weight > 0.0


def maybe_cycle_penalty(
    pred_raw: torch.Tensor,
    cycle_cfg: CycleConfig,
    rng: random.Random,
) -> Tuple[Optional[torch.Tensor], Optional[float]]:
    """Optionally compute cycle penalty; returns (penalty_tensor, cosine)."""

    if not cycle_cfg.enabled():
        return None, None
    if rng.random() >= cycle_cfg.pct:
        return None, None

    vector = pred_raw.detach().cpu().numpy().tolist()
    decode_payload = {
        "vectors": [vector],
        "subscribers": "jxe",
        "steps": max(1, cycle_cfg.steps),
        "device": "cpu",
    }
    try:
        decode_resp = requests.post(
            cycle_cfg.decoder_endpoint,
            json=decode_payload,
            timeout=cycle_cfg.timeout,
        )
        decode_resp.raise_for_status()
        decoded = decode_resp.json()
        text = decoded["results"][0]["subscribers"]["gtr → jxe"]["output"]
        encode_resp = requests.post(
            cycle_cfg.encoder_endpoint,
            json={"texts": [text]},
            timeout=cycle_cfg.timeout,
        )
        encode_resp.raise_for_status()
        cycled = torch.tensor(encode_resp.json()["embeddings"][0], dtype=torch.float32)
    except Exception:
        return None, None

    cycled = cycled.to(pred_raw.device)
    cycled = cycled / (cycled.norm() + 1e-8)
    pred_norm = pred_raw / (pred_raw.norm() + 1e-8)
    cosine = torch.dot(pred_norm, cycled)
    penalty = (1.0 - cosine) * cycle_cfg.weight
    return penalty, float(cosine.item())
