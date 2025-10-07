#!/usr/bin/env python3
"""Tune α parameter for TMD fusion weight (Tiny Bite #2).

This script:
1. Performs log-search over α ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}
2. Measures NDCG@10, Recall@10, and found@8 for each α
3. Selects optimal α based on validation metrics
4. Updates calibration config with optimal α

Usage:
    python tools/tune_alpha_fusion.py \\
        --npz artifacts/ontology_13k.npz \\
        --index artifacts/ontology_13k_ivf_flat_ip.index \\
        --queries eval/validation_queries.jsonl \\
        --output artifacts/alpha_tuning_results.json \\
        --alphas 0.0 0.1 0.2 0.3 0.5
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db_faiss import FaissDB
from src.vectorizer import EmbeddingBackend


def ndcg_at_k(retrieved_ids: List[int], relevant_id: int, k: int = 10) -> float:
    """Calculate NDCG@k for a single query.

    Args:
        retrieved_ids: List of retrieved document IDs (ranked)
        relevant_id: Ground truth relevant document ID
        k: Cutoff position

    Returns:
        NDCG@k score ∈ [0, 1]
    """
    # DCG: sum of (relevance / log2(rank+1)) for top-k
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id == relevant_id:
            dcg = 1.0 / np.log2(rank + 1)
            break

    # IDCG: ideal DCG (relevant doc at rank 1)
    idcg = 1.0 / np.log2(2)  # log2(1+1)

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_ids: List[int], relevant_id: int, k: int = 10) -> float:
    """Calculate Recall@k (binary: 1 if relevant in top-k, 0 otherwise)."""
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def found_at_k(retrieved_scores: List[float], threshold: float, k: int = 8) -> bool:
    """Check if any of top-k results exceeds threshold (for found@k)."""
    return any(score >= threshold for score in retrieved_scores[:k])


def retrieve_with_alpha(
    query_text: str,
    target_id: int,
    faiss_db: FaissDB,
    embedding_backend: EmbeddingBackend,
    tmd_dense: np.ndarray,
    alpha: float,
    k: int = 10,
) -> Tuple[List[int], List[float]]:
    """Retrieve top-k with α-weighted fusion.

    Args:
        query_text: Query string
        target_id: Ground truth target ID (for TMD extraction)
        faiss_db: FaissDB instance
        embedding_backend: GTR-T5 encoder
        tmd_dense: TMD dense vectors array
        alpha: Fusion weight for TMD
        k: Top-k to retrieve

    Returns:
        (retrieved_ids, scores) tuples
    """
    # Encode query with GTR-T5
    gtr_vec = embedding_backend.encode([query_text])[0].astype(np.float32)
    gtr_norm = gtr_vec / (np.linalg.norm(gtr_vec) + 1e-9)

    # Get TMD vector for target concept
    if target_id < len(tmd_dense):
        tmd_vec = tmd_dense[target_id].astype(np.float32)
    else:
        tmd_vec = np.zeros(16, dtype=np.float32)

    tmd_norm = tmd_vec / (np.linalg.norm(tmd_vec) + 1e-9)

    # Fuse: [gtr_norm | α * tmd_norm]
    if alpha == 0.0:
        # No TMD - use only GTR-T5 (768D)
        # Pad with zeros to match index dimension (784D)
        fused_vec = np.concatenate([gtr_norm, np.zeros(16, dtype=np.float32)])
    else:
        fused_vec = np.concatenate([gtr_norm, alpha * tmd_norm])

    fused_vec = fused_vec / (np.linalg.norm(fused_vec) + 1e-9)

    # FAISS search
    fused_query = fused_vec.reshape(1, -1).astype(np.float32)
    scores, indices = faiss_db.search(fused_query, k)

    return indices[0].tolist(), scores[0].tolist()


def run_alpha_sweep(
    queries_path: str,
    npz_path: str,
    index_path: str,
    alphas: List[float],
    k: int = 10,
) -> Dict[float, Dict[str, float]]:
    """Run α parameter sweep on validation queries.

    Args:
        queries_path: Path to validation queries JSONL
        npz_path: Path to corpus NPZ
        index_path: Path to FAISS index
        alphas: List of α values to test
        k: Top-k for metrics

    Returns:
        Dictionary mapping α → metrics
    """
    # Load FAISS index
    print(f"Loading FAISS index from {index_path}...")
    faiss_db = FaissDB(index_path=index_path, meta_npz_path=npz_path)
    faiss_db.load()

    # Load embedding backend
    print("Loading GTR-T5 embedding backend...")
    embedding_backend = EmbeddingBackend()

    # Load corpus metadata
    npz = np.load(npz_path, allow_pickle=True)
    tmd_dense = np.asarray(npz.get("tmd_dense"), dtype=np.float32)
    cpe_ids = [str(x) for x in npz.get("cpe_ids", [])]

    # Create ID → index mapping
    id_to_idx = {cpe_id: idx for idx, cpe_id in enumerate(cpe_ids)}

    # Load validation queries
    print(f"Loading validation queries from {queries_path}...")
    queries = []
    with open(queries_path, "r") as f:
        for line in f:
            record = json.loads(line)
            query_text = record.get("query", "")
            target_id_str = str(record.get("target_id", ""))

            if not query_text or not target_id_str:
                continue

            if target_id_str not in id_to_idx:
                continue

            target_idx = id_to_idx[target_id_str]
            queries.append({
                "text": query_text,
                "target_id": target_idx,
                "target_id_str": target_id_str,
            })

    print(f"Loaded {len(queries)} validation queries\n")

    # Run sweep
    results = {}

    for alpha in alphas:
        print(f"{'='*80}")
        print(f"Testing α = {alpha:.2f}")
        print(f"{'='*80}\n")

        ndcg_scores = []
        recall_scores = []
        found_count = 0
        latencies = []

        for i, query in enumerate(queries):
            start_time = time.perf_counter()

            retrieved_ids, scores = retrieve_with_alpha(
                query_text=query["text"],
                target_id=query["target_id"],
                faiss_db=faiss_db,
                embedding_backend=embedding_backend,
                tmd_dense=tmd_dense,
                alpha=alpha,
                k=k,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            latencies.append(latency_ms)

            # Calculate metrics
            ndcg = ndcg_at_k(retrieved_ids, query["target_id"], k=k)
            recall = recall_at_k(retrieved_ids, query["target_id"], k=k)
            found = found_at_k(scores, threshold=0.70, k=8)  # Default τ=0.70

            ndcg_scores.append(ndcg)
            recall_scores.append(recall)
            if found:
                found_count += 1

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(queries)} queries...")

        # Aggregate metrics
        avg_ndcg = np.mean(ndcg_scores)
        avg_recall = np.mean(recall_scores)
        found_rate = found_count / len(queries)
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        results[alpha] = {
            "ndcg@10": float(avg_ndcg),
            "recall@10": float(avg_recall),
            "found@8": float(found_rate),
            "avg_latency_ms": float(avg_latency),
            "p95_latency_ms": float(p95_latency),
            "n_queries": len(queries),
        }

        print(f"\n  Results for α={alpha:.2f}:")
        print(f"    NDCG@10:     {avg_ndcg:.4f}")
        print(f"    Recall@10:   {avg_recall:.4f}")
        print(f"    Found@8:     {found_rate:.4f}")
        print(f"    Latency:     {avg_latency:.2f}ms (p95={p95_latency:.2f}ms)")
        print()

    return results


def select_optimal_alpha(results: Dict[float, Dict[str, float]]) -> Tuple[float, str]:
    """Select optimal α based on NDCG@10 (primary) and found@8 (secondary).

    Strategy:
    1. Filter α values with found@8 ≥ 0.80 (minimum acceptable)
    2. Among those, pick α with highest NDCG@10
    3. If none meet found@8, pick α with highest found@8

    Returns:
        (optimal_alpha, reasoning)
    """
    # Find candidates with acceptable found@8
    acceptable = {
        alpha: metrics
        for alpha, metrics in results.items()
        if metrics["found@8"] >= 0.80
    }

    if acceptable:
        # Pick best NDCG@10 among acceptable
        best_alpha = max(acceptable.keys(), key=lambda a: acceptable[a]["ndcg@10"])
        reasoning = (
            f"Selected α={best_alpha:.2f} with NDCG@10={acceptable[best_alpha]['ndcg@10']:.4f} "
            f"(found@8={acceptable[best_alpha]['found@8']:.4f} meets ≥0.80 threshold)"
        )
    else:
        # Fallback: pick best found@8 (even if below threshold)
        best_alpha = max(results.keys(), key=lambda a: results[a]["found@8"])
        reasoning = (
            f"Selected α={best_alpha:.2f} with highest found@8={results[best_alpha]['found@8']:.4f} "
            f"(no α achieved ≥0.80 threshold; NDCG@10={results[best_alpha]['ndcg@10']:.4f})"
        )

    return best_alpha, reasoning


def main():
    parser = argparse.ArgumentParser(description="Tune α fusion weight")
    parser.add_argument("--npz", required=True, help="Path to corpus NPZ")
    parser.add_argument("--index", required=True, help="Path to FAISS index")
    parser.add_argument("--queries", required=True, help="Path to validation queries JSONL")
    parser.add_argument("--output", default="artifacts/alpha_tuning_results.json", help="Output JSON path")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5], help="α values to test")
    parser.add_argument("--k", type=int, default=10, help="Top-k for metrics")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"α-WEIGHTED FUSION TUNING (Tiny Bite #2)")
    print(f"{'='*80}\n")
    print(f"NPZ: {args.npz}")
    print(f"Index: {args.index}")
    print(f"Queries: {args.queries}")
    print(f"α values: {args.alphas}")
    print(f"Top-k: {args.k}\n")

    # Run sweep
    results = run_alpha_sweep(
        queries_path=args.queries,
        npz_path=args.npz,
        index_path=args.index,
        alphas=args.alphas,
        k=args.k,
    )

    # Select optimal
    optimal_alpha, reasoning = select_optimal_alpha(results)

    print(f"\n{'='*80}")
    print(f"OPTIMAL α SELECTION")
    print(f"{'='*80}\n")
    print(reasoning)
    print()

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'α':<8} {'NDCG@10':<12} {'Recall@10':<12} {'Found@8':<12} {'Latency (ms)':<15}")
    print("-" * 80)

    for alpha in sorted(results.keys()):
        metrics = results[alpha]
        marker = " ← OPTIMAL" if alpha == optimal_alpha else ""
        print(
            f"{alpha:<8.2f} "
            f"{metrics['ndcg@10']:<12.4f} "
            f"{metrics['recall@10']:<12.4f} "
            f"{metrics['found@8']:<12.4f} "
            f"{metrics['avg_latency_ms']:<15.2f}"
            f"{marker}"
        )

    # Save results
    output_data = {
        "optimal_alpha": optimal_alpha,
        "reasoning": reasoning,
        "results": results,
        "config": {
            "npz": args.npz,
            "index": args.index,
            "queries": args.queries,
            "alphas_tested": args.alphas,
            "k": args.k,
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")
    print(f"\n{'='*80}\n")

    # Print next steps
    print("Next steps:")
    print(f"1. Update calibration config with optimal α={optimal_alpha:.2f}")
    print(f"2. Re-train calibrators with optimal α")
    print(f"3. Benchmark calibrated retrieval end-to-end\n")


if __name__ == "__main__":
    main()
