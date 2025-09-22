from __future__ import annotations
import numpy as np

try:
    import faiss
except Exception:  # allow import before faiss is installed
    faiss = None  # type: ignore

from .utils.norms import l2_normalize


def build_ivf_flat_cosine(vectors: np.ndarray, nlist: int = 256, train_frac: float = 0.05):
    """Build an IVF-Flat index with cosine similarity (IP over L2-normalized).
    Returns (index, trained_bool). If faiss unavailable, returns (None, False).
    """
    if faiss is None:
        return None, False

    assert vectors.ndim == 2, "vectors must be (N, D)"
    vecs = l2_normalize(vectors.astype(np.float32))
    N, D = vecs.shape
    q = faiss.IndexFlatIP(D)
    index = faiss.IndexIVFFlat(q, D, max(8, nlist), faiss.METRIC_INNER_PRODUCT)

    # training sample
    k = max(1, int(N * train_frac))
    ids = np.random.default_rng(0).choice(N, size=min(k, N), replace=False)
    index.train(vecs[ids])
    index.add(vecs)
    index.nprobe = min(16, max(4, nlist // 32))
    return index, True


def search(index, queries: np.ndarray, topk: int = 10):
    if faiss is None or index is None:
        return None
    q = l2_normalize(queries.astype(np.float32))
    scores, ids = index.search(q, topk)
    return scores, ids
