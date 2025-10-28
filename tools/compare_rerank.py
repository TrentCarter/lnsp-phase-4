#!/usr/bin/env python3
"""
Reranker Lift Comparison

Compares retrieval metrics before and after reranking to validate reranker effectiveness.
Expected: ΔR@5 ≈ +5-10pp when containment ≥82%.

Usage:
    python tools/compare_rerank.py \
        --hits artifacts/eval/hits50.jsonl \
        --reranked artifacts/eval/hits50_reranked.jsonl
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict


def load_hits(path: str) -> List[Dict]:
    """Load hits JSONL file (one query result per line)."""
    hits = []
    with open(path) as f:
        for line in f:
            hits.append(json.loads(line))
    return hits


def compute_metrics(hits: List[Dict], gold_key: str = 'gold_id') -> Dict:
    """
    Compute retrieval metrics from hits.

    Expected format per line:
    {
        "query_id": int,
        "gold_id": int,
        "top_k": [id1, id2, ..., id50]
    }
    """
    n_queries = len(hits)

    # Extract top-K lists and gold IDs
    gold_ids = []
    top_k_lists = []

    for hit in hits:
        gold_ids.append(hit[gold_key])
        top_k_lists.append(hit['top_k'])

    gold_ids = np.array(gold_ids)

    # Compute R@k for various k
    metrics = {}
    for k in [1, 3, 5, 10, 20, 50]:
        hits_at_k = []
        for i, top_k in enumerate(top_k_lists):
            top_k_subset = top_k[:k]
            hit = gold_ids[i] in top_k_subset
            hits_at_k.append(hit)

        metrics[f'R_at_{k}'] = float(np.mean(hits_at_k))

    # Compute MRR
    mrr_scores = []
    for i, top_k in enumerate(top_k_lists):
        gold_id = gold_ids[i]
        try:
            rank = top_k.index(gold_id) + 1
            mrr_scores.append(1.0 / rank)
        except ValueError:
            mrr_scores.append(0.0)

    metrics['MRR'] = float(np.mean(mrr_scores))

    # Compute containment
    contain_scores = []
    for i, top_k in enumerate(top_k_lists):
        gold_id = gold_ids[i]
        contain_scores.append(gold_id in top_k)

    metrics['Contain_at_50'] = float(np.mean(contain_scores))
    metrics['n_queries'] = n_queries

    return metrics


def compare_metrics(before: Dict, after: Dict) -> Dict:
    """Compute deltas between before and after metrics."""
    deltas = {}

    for key in ['R_at_1', 'R_at_3', 'R_at_5', 'R_at_10', 'R_at_20', 'R_at_50', 'MRR', 'Contain_at_50']:
        if key in before and key in after:
            delta = after[key] - before[key]
            deltas[key] = {
                'before': before[key],
                'after': after[key],
                'delta': delta,
                'delta_pp': delta * 100  # percentage points
            }

    return deltas


def print_comparison(deltas: Dict, n_queries: int):
    """Print formatted comparison."""
    print("\n" + "="*70)
    print("RERANKER LIFT ANALYSIS")
    print("="*70)
    print(f"Queries: {n_queries}")
    print()

    # Format table
    print(f"{'Metric':<15} {'Before':<10} {'After':<10} {'Δ (pp)':<10} {'Lift':<8}")
    print("-" * 70)

    for key in ['R_at_1', 'R_at_3', 'R_at_5', 'R_at_10', 'R_at_20', 'R_at_50', 'MRR', 'Contain_at_50']:
        if key in deltas:
            d = deltas[key]
            before_pct = d['before'] * 100
            after_pct = d['after'] * 100
            delta_pp = d['delta_pp']

            # Color code lift
            if delta_pp > 0:
                lift_str = f"+{delta_pp:.2f}pp ✅"
            elif delta_pp < 0:
                lift_str = f"{delta_pp:.2f}pp ⚠️"
            else:
                lift_str = f"{delta_pp:.2f}pp"

            print(f"{key:<15} {before_pct:>7.2f}%  {after_pct:>7.2f}%  {delta_pp:>7.2f}pp  {lift_str}")

    print()

    # Gate checks
    r5_lift = deltas['R_at_5']['delta_pp']
    contain = deltas['Contain_at_50']['after']

    print("Reranker Effectiveness:")
    if r5_lift >= 3.0:
        print(f"  ✅ GOOD: R@5 lift = +{r5_lift:.2f}pp (target: ≥+3pp)")
    else:
        print(f"  ⚠️  WEAK: R@5 lift = +{r5_lift:.2f}pp (target: ≥+3pp)")

    if contain >= 0.82:
        print(f"  ✅ GOOD: Contain@50 = {contain*100:.2f}% (target: ≥82%)")
    else:
        print(f"  ⚠️  LOW: Contain@50 = {contain*100:.2f}% (target: ≥82%)")

    print()

    # Decision
    if r5_lift >= 3.0 and contain >= 0.82:
        print("✅ RECOMMENDATION: Ship with reranker enabled (clear lift + good containment)")
        return True
    elif r5_lift >= 3.0:
        print("⚠️  RECOMMENDATION: Reranker helps, but containment low (tune retrieval first)")
        return False
    else:
        print("❌ RECOMMENDATION: Reranker not effective (debug features or disable)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compare reranker lift")
    parser.add_argument('--hits', type=str, required=True, help="Path to pre-rerank hits JSONL")
    parser.add_argument('--reranked', type=str, required=True, help="Path to post-rerank hits JSONL")
    parser.add_argument('--gold-key', type=str, default='gold_id', help="Key for gold ID in JSONL")
    parser.add_argument('--out', type=str, help="Optional output JSON path")

    args = parser.parse_args()

    # Validate paths
    if not Path(args.hits).exists():
        print(f"❌ ERROR: Pre-rerank hits not found: {args.hits}")
        return 1

    if not Path(args.reranked).exists():
        print(f"❌ ERROR: Post-rerank hits not found: {args.reranked}")
        return 1

    # Load hits
    print(f"Loading pre-rerank hits: {args.hits}")
    before_hits = load_hits(args.hits)

    print(f"Loading post-rerank hits: {args.reranked}")
    after_hits = load_hits(args.reranked)

    assert len(before_hits) == len(after_hits), "Query counts must match!"

    # Compute metrics
    print(f"\nComputing metrics for {len(before_hits)} queries...")
    before_metrics = compute_metrics(before_hits, gold_key=args.gold_key)
    after_metrics = compute_metrics(after_hits, gold_key=args.gold_key)

    # Compare
    deltas = compare_metrics(before_metrics, after_metrics)

    # Print results
    passed = print_comparison(deltas, len(before_hits))

    # Save results if requested
    if args.out:
        results = {
            'before': before_metrics,
            'after': after_metrics,
            'deltas': deltas
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved results to: {args.out}")

    return 0 if passed else 1


if __name__ == '__main__':
    exit(main())
