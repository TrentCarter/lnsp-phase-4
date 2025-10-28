#!/usr/bin/env python3
"""
Adaptive-K Retrieval Evaluation
================================

Widens search only for low-confidence queries to improve containment
without global latency spikes.

Policy:
  conf = mean(top-10 cosine scores from raw retrieval)
  K = 10 + int(max(0.0, 0.72 - conf) * 80)  # 10..90
  K = min(100, max(10, K))

High-confidence queries (conf > 0.72): K ≈ 10-30
Low-confidence queries (conf < 0.65): K ≈ 60-100

Based on consultant recommendation.
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import faiss

from rerank_strategies_v2 import (
    dedup_candidates, mmr, rerank_with_sequence_bias, _l2norm
)


class AdaptiveRetrievalShim:
    """Retrieval adapter with confidence-gated adaptive K."""

    def __init__(self, faiss_index, id_to_payload):
        self.index = faiss_index
        self.payload = id_to_payload

    def search_adaptive(self, query_vec: np.ndarray, base_K: int = 50):
        """
        Search with adaptive K based on confidence.

        Returns:
            (candidates, K_used, confidence)
        """
        # Initial search with base_K to estimate confidence
        q = _l2norm(query_vec.reshape(1, -1)).astype(np.float32)
        D, I = self.index.search(q, base_K)

        # Estimate confidence from top-10 mean cosine
        conf = float(D[0][:10].mean())

        # Adaptive K formula (INVERTED LOGIC)
        # High confidence (>0.78): Use fewer candidates (K=20-30)
        # Low confidence (<0.66): Use more candidates (K=60-100)
        # Median confidence (0.66-0.78): Use baseline (K=50)
        if conf > 0.78:
            # High confidence - shrink K
            K_adaptive = max(20, 50 - int((conf - 0.78) * 300))
        elif conf < 0.66:
            # Low confidence - expand K
            K_adaptive = min(100, 50 + int((0.66 - conf) * 500))
        else:
            # Medium confidence - use baseline
            K_adaptive = 50

        # If we need more candidates, re-search
        if K_adaptive > base_K:
            D, I = self.index.search(q, K_adaptive)

        # Build candidate list
        cands = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            text, meta, vec = self.payload[int(idx)]
            cands.append((text, float(score), meta, vec))

        return cands, K_adaptive, conf


def evaluate_adaptive(
    retriever: AdaptiveRetrievalShim,
    dataset,
    base_K: int = 50,
    do_mmr: bool = True,
    mmr_lambda: float = 0.55,
    mmr_top_n: int = 20,  # Apply MMR only to top-N
    top_final: int = 10,
    use_seq_bias: bool = True,
    w_same_article: float = 0.05,
    w_next_gap: float = 0.12,
    tau: float = 3.0,
    directional_bonus: float = 0.03,
):
    """Evaluate with adaptive K retrieval."""
    n = len(dataset)
    r1 = r5 = r10 = 0
    mrr = 0.0
    lat = []
    contain_20 = contain_50 = 0

    # Track K usage
    k_values = []
    conf_values = []

    for ex in dataset:
        qv = ex["pred_vec"].astype(np.float32)
        last_meta = ex["last_meta"]
        truth_key = tuple(ex["truth_key"])

        t0 = time.perf_counter()

        # 1. Adaptive retrieval
        cands, K_used, conf = retriever.search_adaptive(qv, base_K=base_K)
        k_values.append(K_used)
        conf_values.append(conf)

        # Track containment BEFORE reranking
        raw_keys = [(int(c[2]["article_index"]), int(c[2]["chunk_index"])) for c in cands]
        if truth_key in raw_keys[:20]:
            contain_20 += 1
        if truth_key in raw_keys[:50]:
            contain_50 += 1

        # 2. Deduplication
        cands = dedup_candidates(cands)

        # 3. MMR diversity (apply to full pool, not just top-N)
        if do_mmr and len(cands) > top_final:
            if mmr_top_n > 0:
                # Limited MMR pool
                mmr_pool_size = min(mmr_top_n, len(cands))
                mmr_pool = cands[:mmr_pool_size]
                rest = cands[mmr_pool_size:]
                vecs = np.stack([c[3] for c in mmr_pool], axis=0).astype(np.float32)
                sel = mmr(qv, vecs, lambda_=mmr_lambda, k=min(top_final, len(mmr_pool)))
                cands = [mmr_pool[i] for i in sel] + rest[:max(0, top_final - len(sel))]
            else:
                # Full MMR pool (mmr_top_n=0 means all)
                vecs = np.stack([c[3] for c in cands], axis=0).astype(np.float32)
                sel = mmr(qv, vecs, lambda_=mmr_lambda, k=top_final)
                cands = [cands[i] for i in sel]
        else:
            cands = cands[:top_final]

        # 4. Sequence-bias reranking
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
                last_vec=None,
            )
            cands_sorted = [c for _, c in ranked]
        else:
            cands_sorted = cands

        # Extract keys
        keys = [(int(c[2]["article_index"]), int(c[2]["chunk_index"])) for c in cands_sorted]

        dt = (time.perf_counter() - t0) * 1000.0
        lat.append(dt)

        # Check ground truth
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
    k_values = np.array(k_values)
    conf_values = np.array(conf_values)

    return {
        "N": n,
        "R@1": r1 / n,
        "R@5": r5 / n,
        "R@10": r10 / n,
        "MRR@10": mrr / n,
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "Contain@20": contain_20 / n,
        "Contain@50": contain_50 / n,
        "K_mean": float(k_values.mean()),
        "K_p50": float(np.percentile(k_values, 50)),
        "K_p95": float(np.percentile(k_values, 95)),
        "conf_mean": float(conf_values.mean()),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate adaptive-K retrieval")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--payload", type=Path, required=True)
    ap.add_argument("--faiss", type=Path, required=True)
    ap.add_argument("--nprobe", type=int, default=64, help="FAISS nprobe (default: 64)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base_K", type=int, default=50, help="Base K for initial search")
    ap.add_argument("--mmr_lambda", type=float, default=0.55, help="MMR lambda (default: 0.55)")
    ap.add_argument("--mmr_top_n", type=int, default=20, help="Apply MMR only to top-N (default: 20)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/lvm/eval_adaptive_k.json"))
    args = ap.parse_args()

    # Load resources
    print(f"Loading FAISS index (nprobe={args.nprobe})...")
    faiss_index = faiss.read_index(str(args.faiss))
    faiss_index.nprobe = args.nprobe

    print(f"Loading payload...")
    payload = np.load(args.payload, allow_pickle=True).item()

    print(f"Loading dataset...")
    data = np.load(args.npz, allow_pickle=True)
    pred_vecs = data["pred_vecs"]
    last_meta = data["last_meta"].tolist()
    truth_keys = data["truth_keys"].tolist()

    N = len(truth_keys) if args.limit is None else min(args.limit, len(truth_keys))
    dataset = [
        {
            "pred_vec": pred_vecs[i],
            "last_meta": last_meta[i],
            "truth_key": tuple(truth_keys[i]),
        }
        for i in range(N)
    ]
    print(f"Loaded {N:,} samples\n")

    # Create adaptive retriever
    retriever = AdaptiveRetrievalShim(faiss_index, payload)

    # Run evaluation
    print("Running adaptive-K evaluation...")
    results = evaluate_adaptive(
        retriever, dataset,
        base_K=args.base_K,
        do_mmr=True,
        mmr_lambda=args.mmr_lambda,
        mmr_top_n=args.mmr_top_n,
        top_final=10,
        use_seq_bias=True,
        w_same_article=0.05,
        w_next_gap=0.12,
        tau=3.0,
        directional_bonus=0.03,
    )

    # Save results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    print("\n" + "=" * 80)
    print("ADAPTIVE-K RESULTS")
    print("=" * 80)
    print(json.dumps(results, indent=2))
    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
