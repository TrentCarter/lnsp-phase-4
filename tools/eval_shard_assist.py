#!/usr/bin/env python3
"""
Shard-Assist Retrieval Evaluation
==================================

Runs per-article subsearch in parallel with global IVF search for queries
that likely need the next chunk from the same article.

Strategy:
1. Run global IVF search (K=50, nprobe=64)
2. If article_index present, run local shard search (K_local=20)
3. Union + dedup → keep top ~60 by cosine
4. Apply existing MMR → sequence-bias → directional pipeline

Expected lift: +0.8-2.0pp R@1, ≤+0.3ms P95 on gated queries
"""

import argparse
import json
import time
import pickle
from pathlib import Path
import numpy as np
import faiss

from rerank_strategies_v2 import (
    dedup_candidates, mmr, rerank_with_sequence_bias, _l2norm
)


class ShardAssistedRetrievalShim:
    """Retrieval adapter with per-article shard assist."""

    def __init__(self, faiss_index, global_payload, article_shards):
        self.global_index = faiss_index
        self.global_payload = global_payload
        self.article_shards = article_shards

    def search_with_shard_assist(
        self,
        query_vec: np.ndarray,
        last_meta: dict,
        K_global: int = 50,
        K_local: int = 20,
        K_union: int = 60,
    ):
        """
        Search with optional shard assist.

        Args:
            query_vec: Query vector [768]
            last_meta: Last context metadata (has article_index)
            K_global: Candidates from global IVF search
            K_local: Candidates from local shard search
            K_union: Keep top-K after union

        Returns:
            (candidates, shard_used: bool)
        """
        # 1. Global IVF search
        q = _l2norm(query_vec.reshape(1, -1)).astype(np.float32)
        D_global, I_global = self.global_index.search(q, K_global)

        global_cands = []
        for score, idx in zip(D_global[0], I_global[0]):
            if idx == -1:
                continue
            text, meta, vec = self.global_payload[int(idx)]
            global_cands.append((text, float(score), meta, vec))

        # 2. Local shard search (if article_index present)
        shard_used = False
        local_cands = []

        article_idx = last_meta.get("article_index")
        if article_idx is not None and article_idx in self.article_shards:
            shard_used = True
            shard = self.article_shards[article_idx]

            # Search local shard
            D_local, I_local = shard["index"].search(q, min(K_local, shard["n_chunks"]))

            for score, local_idx in zip(D_local[0], I_local[0]):
                if local_idx == -1:
                    continue
                text, meta, vec, global_idx = shard["payload"][int(local_idx)]
                local_cands.append((text, float(score), meta, vec))

        # 3. Union + dedup
        all_cands = global_cands + local_cands

        # Dedup by (article_index, chunk_index)
        seen = set()
        deduped = []
        for cand in all_cands:
            key = (int(cand[2]["article_index"]), int(cand[2]["chunk_index"]))
            if key not in seen:
                seen.add(key)
                deduped.append(cand)

        # Sort by cosine (descending) and keep top K_union
        deduped.sort(key=lambda x: x[1], reverse=True)
        merged = deduped[:K_union]

        return merged, shard_used


def evaluate_shard_assist(
    retriever: ShardAssistedRetrievalShim,
    dataset,
    K_global: int = 50,
    K_local: int = 20,
    K_union: int = 60,
    do_mmr: bool = True,
    mmr_lambda: float = 0.7,
    top_final: int = 10,
    use_seq_bias: bool = True,
    w_same_article: float = 0.05,
    w_next_gap: float = 0.12,
    tau: float = 3.0,
    directional_bonus: float = 0.03,
):
    """Evaluate with shard-assist retrieval."""
    n = len(dataset)
    r1 = r5 = r10 = 0
    mrr = 0.0
    lat = []
    contain_20 = contain_50 = 0

    # Track shard usage
    shard_gated = 0
    shard_hits = 0  # How many times shard found ground truth

    for ex in dataset:
        qv = ex["pred_vec"].astype(np.float32)
        last_meta = ex["last_meta"]
        truth_key = tuple(ex["truth_key"])

        t0 = time.perf_counter()

        # 1. Shard-assisted retrieval
        cands, shard_used = retriever.search_with_shard_assist(
            qv, last_meta, K_global=K_global, K_local=K_local, K_union=K_union
        )

        if shard_used:
            shard_gated += 1

        # Track containment BEFORE reranking
        raw_keys = [(int(c[2]["article_index"]), int(c[2]["chunk_index"])) for c in cands]
        if truth_key in raw_keys[:20]:
            contain_20 += 1
        if truth_key in raw_keys[:50]:
            contain_50 += 1
            if shard_used:
                shard_hits += 1

        # 2. Deduplication (already done in search_with_shard_assist, but keep for consistency)
        cands = dedup_candidates(cands)

        # 3. MMR diversity
        if do_mmr and len(cands) > top_final:
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
        "shard_gated_pct": shard_gated / n,
        "shard_hit_rate": shard_hits / max(1, shard_gated),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate shard-assist retrieval")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--payload", type=Path, required=True)
    ap.add_argument("--faiss", type=Path, required=True)
    ap.add_argument("--shards", type=Path, required=True, help="Article shards pickle file")
    ap.add_argument("--nprobe", type=int, default=64, help="FAISS nprobe (default: 64)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--K_global", type=int, default=50, help="Global IVF K (default: 50)")
    ap.add_argument("--K_local", type=int, default=20, help="Local shard K (default: 20)")
    ap.add_argument("--K_union", type=int, default=60, help="Keep top-K after union (default: 60)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/lvm/eval_shard_assist.json"))
    args = ap.parse_args()

    # Load resources
    print(f"Loading FAISS index (nprobe={args.nprobe})...")
    faiss_index = faiss.read_index(str(args.faiss))
    faiss_index.nprobe = args.nprobe

    print(f"Loading global payload...")
    global_payload = np.load(args.payload, allow_pickle=True).item()

    print(f"Loading article shards...")
    with open(args.shards, "rb") as f:
        article_shards = pickle.load(f)
    print(f"  Loaded {len(article_shards):,} article shards")

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

    # Create shard-assisted retriever
    retriever = ShardAssistedRetrievalShim(faiss_index, global_payload, article_shards)

    # Run evaluation
    print("Running shard-assist evaluation...")
    results = evaluate_shard_assist(
        retriever, dataset,
        K_global=args.K_global,
        K_local=args.K_local,
        K_union=args.K_union,
        do_mmr=True,
        mmr_lambda=0.7,
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
    print("SHARD-ASSIST RESULTS")
    print("=" * 80)
    print(json.dumps(results, indent=2))
    print(f"\nResults saved to: {args.out}")

    # Show lift vs baseline
    print("\n" + "=" * 80)
    print("COMPARISON TO BASELINE (Fixed-K, nprobe=64)")
    print("=" * 80)
    print(f"Contain@50:  {results['Contain@50']:.1%} (baseline: 67.2%)")
    print(f"R@10:        {results['R@10']:.1%} (baseline: 54.2%)")
    print(f"R@5:         {results['R@5']:.1%} (baseline: 51.8%)")
    print(f"R@1:         {results['R@1']:.1%} (baseline: 1.0%)")
    print(f"P95:         {results['p95_ms']:.2f}ms (baseline: 1.30ms)")
    print(f"\nShard usage: {results['shard_gated_pct']:.1%} of queries")
    print(f"Shard hit rate: {results['shard_hit_rate']:.1%} (when gated)")


if __name__ == "__main__":
    main()
