"""Synchronous FAISS miner to avoid races during stabilization.
- Runs in the training process (no multiprocessing).
- Returns only (ids, sims). Caller gathers vectors from CPU bank.
"""
from __future__ import annotations
import faiss
import numpy as np
from typing import Tuple

class SyncFaissMiner:
    def __init__(self, index: faiss.Index, nprobe: int = 8):
        self.index = index
        try:
            faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)
        except Exception:
            pass

    def search(self, queries: np.ndarray, k: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        # queries: (B, 768) float32, unit‑norm
        D, I = self.index.search(queries.astype(np.float32), k)
        return I, D
