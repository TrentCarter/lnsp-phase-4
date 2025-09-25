from __future__ import annotations
import sys
import os
import numpy as np
import argparse

# Add the parent directory to sys.path for direct execution
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

try:
    import faiss
except Exception:  # allow import before faiss is installed
    faiss = None  # type: ignore

from utils.norms import l2_normalize


def build_ivf_flat_cosine(vectors: np.ndarray, nlist: int = 256, train_frac: float = 0.05):
    """Build an IVF-Flat index with cosine similarity (IP over L2-normalized).
    Returns (index, trained_bool). If faiss unavailable, returns (None, False).
    """
    if faiss is None:
        return None, False

    assert vectors.ndim == 2, "vectors must be (N, D)"
    vecs = l2_normalize(vectors.astype(np.float32))
    N, D = vecs.shape
    # Build IVF-Flat via index_factory with inner-product metric
    spec = f"IVF{max(8, int(nlist))},Flat"
    index = faiss.index_factory(D, spec, faiss.METRIC_INNER_PRODUCT)

    # training sample
    k = max(1, int(N * train_frac))
    ids = np.random.default_rng(0).choice(N, size=min(k, N), replace=False)
    index.train(vecs[ids])
    index.add(vecs)
    # Default nprobe can be overridden via env
    try:
        default_nprobe = int(os.getenv("FAISS_NPROBE", "16"))
    except Exception:
        default_nprobe = 16
    index.nprobe = max(1, default_nprobe)

    # Startup guard: warn if vectors < nlist*4
    if N < nlist * 4:
        recommended_nprobe = max(8, nlist // 8)
        print(f"[WARN] vectors ({N}) < nlist*4 ({nlist*4}). Setting nprobe={recommended_nprobe}")
        index.nprobe = recommended_nprobe
    return index, True


def search(index, queries: np.ndarray, topk: int = 10):
    if faiss is None or index is None:
        return None
    q = l2_normalize(queries.astype(np.float32))
    scores, ids = index.search(q, topk)
    return scores, ids


def _detect_npz() -> str | None:
    candidates = [
        os.getenv("FAISS_NPZ_PATH"),
        "artifacts/fw10k_vectors.npz",
        "artifacts/fw1k_vectors.npz",
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and save FAISS IVF_FLAT index from NPZ vectors")
    parser.add_argument("--npz", type=str, default=None, help="Path to NPZ file containing 'vectors'")
    parser.add_argument("--index-type", type=str, default="IVF_FLAT", help="Index type (only IVF_FLAT supported)")
    parser.add_argument("--nlist", type=int, default=128, help="Number of lists (nlist) for IVF")
    parser.add_argument("--out", type=str, default=None, help="Output index path (.index)")
    args = parser.parse_args()

    npz_path = args.npz or _detect_npz()
    if not npz_path:
        print("[faiss_index] No NPZ file found. Use --npz to specify path.")
        return 2
    if not os.path.isfile(npz_path):
        print(f"[faiss_index] NPZ file not found: {npz_path}")
        return 2

    if args.index_type and args.index_type.upper() != "IVF_FLAT":
        print(f"[faiss_index] Unsupported index type: {args.index_type}. Only IVF_FLAT is supported.")
        return 2

    try:
        npz = np.load(npz_path)
        if "vectors" not in npz.files:
            print(f"[faiss_index] NPZ missing 'vectors' key: {npz_path}")
            return 2
        vectors = npz["vectors"].astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            print(f"[faiss_index] No vectors to index in {npz_path}")
            return 1
    except Exception as exc:
        print(f"[faiss_index] Error loading NPZ: {exc}")
        return 2

    if faiss is None:
        print("[faiss_index] FAISS not available; please install faiss-cpu.")
        return 2

    index, trained = build_ivf_flat_cosine(vectors, nlist=int(args.nlist))
    if not trained or index is None:
        print("[faiss_index] Failed to build index")
        return 2

    out_path = args.out or ("artifacts/fw10k_ivf.index" if "fw10k" in os.path.basename(npz_path) else "artifacts/fw1k_ivf.index")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    faiss.write_index(index, out_path)
    print(f"[faiss_index] Wrote index to {out_path} (nlist={getattr(index, 'nlist', 'NA')})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
