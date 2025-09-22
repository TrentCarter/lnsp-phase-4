from __future__ import annotations
import numpy as np
from typing import Dict, Any

try:
    import faiss
except Exception:
    faiss = None  # type: ignore

from ..utils.norms import l2_normalize


class FaissShard:
    """Simple in-memory shard that can be serialized by caller.
    Caller decides sharding policy (e.g., per lane_index).
    """

    def __init__(self, dim: int, nlist: int = 256):
        self.dim = dim
        self.nlist = nlist
        self.index = None

    def build(self, vectors: np.ndarray):
        if faiss is None:
            self.index = None
            return False
        import faiss as _fa
        q = _fa.IndexFlatIP(self.dim)
        self.index = _fa.IndexIVFFlat(q, self.dim, self.nlist, _fa.METRIC_INNER_PRODUCT)
        vecs = l2_normalize(vectors.astype(np.float32))
        # train on 5% or up to 5k
        N = len(vecs)
        k = max(1, min(5000, int(N * 0.05)))
        ids = np.random.default_rng(0).choice(N, size=k, replace=False)
        self.index.train(vecs[ids])
        self.index.add(vecs)
        self.index.nprobe = min(16, max(4, self.nlist // 32))
        return True

    def add(self, vectors: np.ndarray):
        if self.index is None:
            return False
        vecs = l2_normalize(vectors.astype(np.float32))
        self.index.add(vecs)
        return True

    def search(self, queries: np.ndarray, topk: int = 10):
        if self.index is None:
            return None
        q = l2_normalize(queries.astype(np.float32))
        return self.index.search(q, topk)
