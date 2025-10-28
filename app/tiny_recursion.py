#!/usr/bin/env python3
"""
Tiny Recursion inference helper.

Provides a lightweight interface used by tooling to generate Tiny Recursion
predictions for retrieval experiments.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from app.lvm.models import AttentionMixtureNetwork

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = (
    "artifacts/lvm/production_model/best_model.pt",
    "artifacts/lvm/production_model/final_model.pt",
)
DEFAULT_BATCH_SIZE = 128


class _TinyRecursionRunner:
    def __init__(self, checkpoint: Path):
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        params = ckpt.get("hyperparameters", {})
        model = AttentionMixtureNetwork(
            input_dim=params.get("input_dim", 768),
            d_model=params.get("d_model", 256),
            hidden_dim=params.get("hidden_dim", 512),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        self.model = model
        self.device = device

    def predict(
        self,
        contexts: np.ndarray,
        temp: float,
        seed: int,
        attempts: int,
        batch_size: int,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        outputs: list[np.ndarray] = []
        total = contexts.shape[0]

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            ctx_np = contexts[start:end]
            ctx_t = torch.from_numpy(ctx_np).to(self.device)
            ctx_t = F.normalize(ctx_t, dim=-1)

            preds = self._run_recursion(ctx_t, temp=temp, rng=rng, attempts=attempts)
            outputs.append(preds.cpu().numpy())

        return np.vstack(outputs).astype(np.float32)

    def _run_recursion(
        self,
        context_t: torch.Tensor,
        temp: float,
        rng: np.random.Generator,
        attempts: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            current = self.model(context_t)

        if attempts <= 1:
            return current

        base_context = context_t

        for attempt in range(1, attempts):
            blend_weight = min(0.55, 0.35 + temp * 3.0)
            noise_scale = max(temp * 0.1, 1e-4)
            noise_np = rng.normal(
                loc=0.0, scale=noise_scale, size=current.shape
            ).astype(np.float32)
            noise = torch.from_numpy(noise_np).to(self.device)

            blended = (
                (1.0 - blend_weight) * base_context[:, -1, :]
                + blend_weight * current
                + noise
            )
            blended = F.normalize(blended, dim=-1)

            context_mod = base_context.clone()
            context_mod[:, -1, :] = blended

            with torch.no_grad():
                current = self.model(context_mod)

        return current


@lru_cache(maxsize=1)
def _get_runner(checkpoint: Path) -> _TinyRecursionRunner:
    return _TinyRecursionRunner(checkpoint)


def _resolve_checkpoint() -> Path:
    override = os.getenv("TINY_RECURSION_CHECKPOINT")
    if override:
        path = (REPO_ROOT / override).resolve()
        if not path.exists():
            raise FileNotFoundError(f"TINY_RECURSION_CHECKPOINT not found: {path}")
        return path

    for rel in DEFAULT_CHECKPOINTS:
        path = (REPO_ROOT / rel).resolve()
        if path.exists():
            return path

    raise FileNotFoundError(
        "No Tiny Recursion checkpoint found under artifacts/lvm/production_model"
    )


def predict(
    contexts: Sequence[np.ndarray] | np.ndarray,
    *,
    temp: float = 0.06,
    seed: int = 1337,
    attempts: int = 2,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    contexts_arr = np.asarray(contexts, dtype=np.float32)
    if contexts_arr.ndim != 3:
        raise ValueError(
            f"contexts must have shape [N, seq_len, dim]; received {contexts_arr.shape}"
        )

    checkpoint = _resolve_checkpoint()
    runner = _get_runner(checkpoint)

    return runner.predict(
        contexts_arr,
        temp=temp,
        seed=seed,
        attempts=max(1, attempts),
        batch_size=max(1, batch_size),
    )
