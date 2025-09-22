"""LightRAG functional helpers (MIT License)."""

# The original implementations are derived from LightRAG
# (c) SylphAI, Inc., MIT License. Adapted for compatibility
# with the LNSP pipeline where optional dependencies are trimmed.

from __future__ import annotations

from typing import Iterable, List, Tuple, Union

import numpy as np

VECTOR_TYPE = Union[List[float], np.ndarray]

__all__ = [
    "get_top_k_indices_scores",
    "normalize_np_array",
    "normalize_vector",
]


def normalize_np_array(v: np.ndarray) -> np.ndarray:
    """Normalize a NumPy vector with zero-norm safeguards."""

    norm = float(np.linalg.norm(v))
    if norm == 0 or not np.isfinite(norm):
        return v
    return v / norm


def normalize_vector(v: VECTOR_TYPE) -> List[float]:
    """Normalize a Python list/array, returning a list of floats."""

    arr = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0 or not np.isfinite(norm):
        return arr.tolist()
    return (arr / norm).tolist()


def get_top_k_indices_scores(
    scores: Union[List[float], np.ndarray], top_k: int
) -> Tuple[List[int], List[float]]:
    """Return the indices and values of the top-k scores."""

    arr = np.asarray(scores, dtype=np.float32)
    if arr.size == 0:
        return [], []

    k = min(max(top_k, 0), arr.size)
    if k == 0:
        return [], []

    top_indices = np.argsort(arr)[-k:][::-1]
    top_scores = arr[top_indices]
    return top_indices.tolist(), top_scores.astype(float).tolist()

