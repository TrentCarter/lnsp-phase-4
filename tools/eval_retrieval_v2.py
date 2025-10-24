#!/usr/bin/env python3
"""
Retrieval Evaluation Framework v2 (Production-Ready)
====================================================

Production-grade implementation from consultant with:
- Shim-based architecture (decoupled from data formats)
- Command-line driven (no hardcoded paths)
- Clean separation of concerns
- Pre-prepared evaluation datasets

Based on consultant code review and architecture recommendations.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np

# Local imports
from rerank_strategies_v2 import (
    dedup_candidates, mmr, rerank_with_sequence_bias, cosine_softmax_weights, _l2norm, Candidate
)

"""
Expected host environment provides:
- A retrieval adapter that exposes: search(query_vec: np.ndarray, K: int) -> List[Candidate]
  Candidate = (text, cosine, meta, vec768)
- A dataset iterator of dicts with keys: {
    "pred_vec": np.ndarray[768],        # AMN (or other LVM) predicted vector
    "last_meta": {"article_index":int, "chunk_index":int},
    "truth_key": (article_index, chunk_index),
}
"""


# ============================================================================
# SHIMS - Adapt to your environment
# ============================================================================

class RetrievalShim:
    """Adapter for FAISS-based retrieval."""

    def __init__(self, faiss_index, id_to_payload):
        """
        Args:
            faiss_index: FAISS index object
            id_to_payload: Dict mapping index IDs to (text, meta, vec) tuples
        """
        self.index = faiss_index
        self.payload = id_to_payload  # dict id -> (text, meta, vec)

    def search(self, query_vec: np.ndarray, K: int) -> list:
        """
        Search for K nearest neighbors.

        Args:
            query_vec: Query vector [768]
            K: Number of candidates to retrieve

        Returns:
            List of Candidate tuples: (text, cosine, meta, vec)
        """
        # Assume index is cosine-ready (inner product on normalized vectors)
        q = _l2norm(query_vec.reshape(1, -1)).astype(np.float32)
        D, I = self.index.search(q, K)  # FAISS returns (scores, ids)

        out = []
        for score, idx in zip(D[0], I[0]):
            text, meta, vec = self.payload[int(idx)]
            out.append((text, float(score), meta, vec))
        return out


class DatasetShim:
    """Adapter for evaluation dataset."""

    def __init__(self, npz_path: Path, limit: int | None = None):
        """
        Load evaluation dataset from NPZ file.

        Expected keys:
            - pred_vecs: [N, 768] predicted vectors from LVM
            - last_meta: [N] list of dicts with article_index, chunk_index
            - truth_keys: [N, 2] ground truth (article_index, chunk_index) pairs

        Args:
            npz_path: Path to evaluation NPZ file
            limit: Optional limit on number of samples to use
        """
        data = np.load(npz_path, allow_pickle=True)
        self.pred = data["pred_vecs"]  # [N,768]
        self.last_meta = data["last_meta"].tolist()
        self.truth_keys = data["truth_keys"].tolist()
        self.N = len(self.truth_keys) if limit is None else min(limit, len(self.truth_keys))

    def __len__(self):
        return self.N

    def __iter__(self):
        for i in range(self.N):
            yield {
                "pred_vec": self.pred[i],
                "last_meta": self.last_meta[i],
                "truth_key": tuple(self.truth_keys[i]),
            }


# ============================================================================
# EVALUATION ENGINE
# ============================================================================

def evaluate(
    retriever: RetrievalShim,
    dataset: DatasetShim,
    K_retrieve: int = 50,
    do_mmr: bool = True,
    mmr_lambda: float = 0.7,
    top_final: int = 10,
    use_seq_bias: bool = True,
    w_same_article: float = 0.05,
    w_next_gap: float = 0.12,
    tau: float = 3.0,
    directional_bonus: float = 0.0,
) -> dict:
    """
    Evaluate retrieval quality with IR metrics.

    Args:
        retriever: Retrieval adapter
        dataset: Evaluation dataset
        K_retrieve: Number of candidates to retrieve initially
        do_mmr: Whether to apply MMR diversity
        mmr_lambda: MMR lambda parameter
        top_final: Final number of candidates after reranking
        use_seq_bias: Whether to apply sequence-bias reranking
        w_same_article: Same-article bonus weight
        w_next_gap: Next-chunk bonus weight
        tau: Gap penalty temperature
        directional_bonus: Directional alignment bonus weight

    Returns:
        Dictionary with metrics: R@1, R@5, R@10, MRR@10, latency
    """
    n = len(dataset)
    r1 = r5 = r10 = 0
    mrr = 0.0
    lat = []

    for ex in dataset:
        qv = ex["pred_vec"].astype(np.float32)
        last_meta = ex["last_meta"]
        truth_key = tuple(ex["truth_key"])  # (article_idx, chunk_idx)

        # Measure retrieval latency
        t0 = time.perf_counter()

        # 1. Initial retrieval
        cands = retriever.search(qv, K_retrieve)

        # 2. Deduplication
        cands = dedup_candidates(cands)

        # 3. MMR diversity (optional)
        if do_mmr and len(cands) > top_final:
            vecs = np.stack([c[3] for c in cands], axis=0).astype(np.float32)
            sel = mmr(qv, vecs, lambda_=mmr_lambda, k=top_final)
            cands = [cands[i] for i in sel]
        else:
            cands = cands[:top_final]

        # 4. Sequence-bias reranking (optional)
        if use_seq_bias and cands:
            ranked = rerank_with_sequence_bias(
                candidates=cands,
                last_ctx_meta=last_meta,
                w_cos=1.0,
                w_same_article=w_same_article,
                w_next_gap=w_next_gap,
                tau=tau,
                directional_bonus=directional_bonus,
                pred_vec=qv,
                last_vec=None,  # If you have last_vec, pass it to enable directional bonus
            )
            cands_sorted = [c for _, c in ranked]
        else:
            cands_sorted = cands

        # Extract keys for matching
        keys = [(int(c[2]["article_index"]), int(c[2]["chunk_index"])) for c in cands_sorted]

        # Measure latency
        dt = (time.perf_counter() - t0) * 1000.0
        lat.append(dt)

        # Check if ground truth is in results
        if truth_key in keys:
            idx = keys.index(truth_key)
            if idx == 0:
                r1 += 1
            if idx < 5:
                r5 += 1
            if idx < 10:
                r10 += 1
            mrr += 1.0 / (idx + 1)

    # Compute metrics
    lat = np.array(lat)
    return {
        "N": n,
        "R@1": r1 / n,
        "R@5": r5 / n,
        "R@10": r10 / n,
        "MRR@10": mrr / n,
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
    }


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Evaluate LVM-based retrieval with reranking strategies")
    ap.add_argument("--npz", type=Path, required=True,
                    help="OOD or val set NPZ with pred_vecs/last_meta/truth_keys")
    ap.add_argument("--payload", type=Path, required=True,
                    help=".npz or .npy payload mapping IDs→(text,meta,vec)")
    ap.add_argument("--faiss", type=Path, required=True,
                    help="FAISS index file")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of evaluation samples (default: all)")
    ap.add_argument("--no_mmr", action="store_true",
                    help="Disable MMR diversity")
    ap.add_argument("--top_final", type=int, default=10,
                    help="Final number of candidates after reranking (default: 10)")
    ap.add_argument("--mmr_lambda", type=float, default=0.7,
                    help="MMR lambda parameter (default: 0.7)")
    ap.add_argument("--w_same_article", type=float, default=0.05,
                    help="Same-article bonus weight (default: 0.05)")
    ap.add_argument("--w_next_gap", type=float, default=0.12,
                    help="Next-chunk bonus weight (default: 0.12)")
    ap.add_argument("--tau", type=float, default=3.0,
                    help="Gap penalty temperature (default: 3.0)")
    ap.add_argument("--directional_bonus", type=float, default=0.0,
                    help="Directional alignment bonus weight (default: 0.0, disabled)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/lvm/eval_retrieval_results.json"),
                    help="Output JSON file for results")
    args = ap.parse_args()

    # Load FAISS index and payload
    # NOTE: Replace these loaders with your actual ones
    import faiss  # type: ignore

    print(f"Loading FAISS index from {args.faiss}...")
    faiss_index = faiss.read_index(str(args.faiss))

    print(f"Loading payload from {args.payload}...")
    payload_obj = np.load(args.payload, allow_pickle=True).item()

    # Create retrieval adapter
    retriever = RetrievalShim(faiss_index, payload_obj)

    # Load evaluation dataset
    print(f"Loading evaluation dataset from {args.npz}...")
    dataset = DatasetShim(args.npz, limit=args.limit)
    print(f"Loaded {len(dataset):,} evaluation samples")

    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluate(
        retriever, dataset,
        K_retrieve=50,
        do_mmr=(not args.no_mmr),
        mmr_lambda=args.mmr_lambda,
        top_final=args.top_final,
        use_seq_bias=True,
        w_same_article=args.w_same_article,
        w_next_gap=args.w_next_gap,
        tau=args.tau,
        directional_bonus=args.directional_bonus,
    )

    # Save results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(json.dumps(results, indent=2))
    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
