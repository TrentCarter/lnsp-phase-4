#!/usr/bin/env python3
"""
Test different fusion weight configurations for sentence-aware retrieval.

Tries multiple weight combinations to find optimal balance between
paragraph-level and sentence-level similarity.

Usage:
    python tools/test_fusion_weights.py \
        --sent-bank artifacts/arxiv_sentence_bank.npz \
        --para-npz artifacts/lvm/arxiv_papers_210_768d.npz \
        --out-table artifacts/fusion_weight_results.json
"""
import argparse
import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retriever.sent_para_rerank import SentenceParaReranker


def test_fusion_config(
    reranker,
    para_vecs,
    queries,
    targets,
    a,  # sentence weight
    b,  # paragraph weight
    k=10
):
    """Test a specific fusion weight configuration."""
    all_results = []

    for query_vec, target_id in queries:
        # Stage 1: Paragraph retrieval
        scores = para_vecs @ query_vec
        top_k_indices = np.argsort(scores)[-k:][::-1]
        para_results = [(int(idx), float(scores[idx])) for idx in top_k_indices]

        # Stage 2: Sentence reranking with custom weights
        para_candidates = [(pid, score) for pid, score in para_results]
        selected_vecs = para_vecs[top_k_indices]

        reranked = []
        for i, (para_id, cos_para) in enumerate(para_candidates):
            cos_sent_hit, _ = reranker.compute_sent_hit(query_vec, para_id, top_k=2)

            # Custom fusion
            fused_score = a * cos_sent_hit + b * cos_para

            reranked.append((para_id, fused_score, cos_para, cos_sent_hit))

        reranked.sort(key=lambda x: x[1], reverse=True)
        results = [(pid, fused) for pid, fused, _, _ in reranked]

        all_results.append(results)

    # Compute metrics
    metrics = {}
    for k_val in [1, 5, 10]:
        hits = 0
        for result, target in zip(all_results, targets):
            retrieved_ids = [r[0] for r in result[:k_val]]
            if target in retrieved_ids:
                hits += 1
        metrics[f"R@{k_val}"] = hits / len(targets) if targets else 0.0

    # MRR
    mrr_sum = 0.0
    for result, target in zip(all_results, targets):
        retrieved_ids = [r[0] for r in result]
        if target in retrieved_ids:
            rank = retrieved_ids.index(target) + 1
            mrr_sum += 1.0 / rank
    metrics["MRR"] = mrr_sum / len(targets) if targets else 0.0

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sent-bank", required=True)
    ap.add_argument("--para-npz", required=True)
    ap.add_argument("--out-table", required=True)
    ap.add_argument("--n-queries", type=int, default=50)
    args = ap.parse_args()

    print("="*60)
    print("Loading data...")

    # Load paragraph data
    para_data = np.load(args.para_npz, allow_pickle=True)
    para_vecs = para_data["vectors"].astype(np.float32)
    para_vecs /= (np.linalg.norm(para_vecs, axis=1, keepdims=True) + 1e-9)

    print(f"  Loaded {len(para_vecs)} paragraphs")

    # Load reranker
    reranker = SentenceParaReranker(args.sent_bank)

    # Create test queries
    queries = []
    for i in range(min(args.n_queries, len(para_vecs) - 1)):
        queries.append((para_vecs[i], i + 1))

    targets = [q[1] for q in queries]

    print(f"  Created {len(queries)} test queries")
    print("="*60)

    # Test different fusion weight configurations
    configs = [
        {"name": "baseline_para_only", "a": 0.0, "b": 1.0},
        {"name": "conservative_favor_para", "a": 0.3, "b": 0.6},
        {"name": "balanced", "a": 0.5, "b": 0.4},
        {"name": "slight_sent_favor", "a": 0.6, "b": 0.35},
        {"name": "original_sent_heavy", "a": 0.75, "b": 0.15},
    ]

    results = []

    for config in configs:
        print(f"\nTesting: {config['name']}")
        print(f"  Weights: a={config['a']:.2f} (sent), b={config['b']:.2f} (para)")

        metrics = test_fusion_config(
            reranker, para_vecs, queries, targets,
            a=config["a"], b=config["b"]
        )

        result = {
            "config": config["name"],
            "weights": {"sent": config["a"], "para": config["b"]},
            "metrics": metrics
        }

        print(f"  R@1: {metrics['R@1']:.3f}")
        print(f"  R@5: {metrics['R@5']:.3f}")
        print(f"  R@10: {metrics['R@10']:.3f}")
        print(f"  MRR: {metrics['MRR']:.3f}")

        results.append(result)

    # Save results
    os.makedirs(os.path.dirname(args.out_table) or ".", exist_ok=True)
    with open(args.out_table, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Config':<30} {'a(sent)':<10} {'b(para)':<10} {'R@5':<8} {'MRR':<8}")
    print("-"*70)

    for r in results:
        w = r["weights"]
        m = r["metrics"]
        print(f"{r['config']:<30} {w['sent']:<10.2f} {w['para']:<10.2f} {m['R@5']:<8.3f} {m['MRR']:<8.3f}")

    print("="*70)

    # Find best config
    best = max(results, key=lambda x: x["metrics"]["R@5"])
    baseline = results[0]  # Paragraph-only

    improvement = (best["metrics"]["R@5"] - baseline["metrics"]["R@5"]) * 100

    print(f"\n✓ Best config: {best['config']}")
    print(f"  R@5: {best['metrics']['R@5']:.3f} ({improvement:+.1f}pp vs baseline)")
    print(f"  MRR: {best['metrics']['MRR']:.3f}")
    print(f"  Weights: sent={best['weights']['sent']:.2f}, para={best['weights']['para']:.2f}")

    print(f"\n✓ Results saved to: {args.out_table}")

if __name__ == "__main__":
    main()
