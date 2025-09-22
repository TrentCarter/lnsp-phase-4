from __future__ import annotations
import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Unit-normalize along axis for cosine/IP usage."""
    denom = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom
