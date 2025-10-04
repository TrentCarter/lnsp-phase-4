#!/usr/bin/env python3
"""Compute summary metrics from alpha tuning results."""
import json
import sys
from pathlib import Path
from typing import Dict, List

def compute_metrics(results: List[Dict]) -> Dict:
    """Compute P@k, MRR, nDCG from per-query results."""
    if not results:
        return {}

    total = len(results)

    # Handle None values by treating as 999 (not found)
    # NOTE: gold_rank uses 1-based indexing (1=first, 2=second, etc.)
    def get_rank(r):
        rank = r.get('gold_rank')
        return 999 if rank is None else rank

    p_at_1 = sum(1 for r in results if get_rank(r) == 1) / total  # rank=1 is first position
    p_at_5 = sum(1 for r in results if get_rank(r) <= 5) / total
    p_at_10 = sum(1 for r in results if get_rank(r) <= 10) / total

    # MRR (1-based ranking: 1/1 for rank=1, 1/2 for rank=2, etc.)
    mrr_sum = 0.0
    for r in results:
        rank = get_rank(r)
        if rank < 999:
            mrr_sum += 1.0 / rank  # 1-based: rank=1 gives 1/1, rank=2 gives 1/2
    mrr = mrr_sum / total

    # nDCG (simplified, 1-based ranking)
    import math
    ndcg_sum = 0.0
    for r in results:
        rank = get_rank(r)
        if rank < 999:
            ndcg_sum += 1.0 / math.log2(rank + 1)  # rank=1 gives log2(2)=1, rank=2 gives log2(3)
    ndcg = ndcg_sum / total

    return {
        "p_at_1": p_at_1,
        "p_at_5": p_at_5,
        "p_at_10": p_at_10,
        "mrr": mrr,
        "ndcg": ndcg,
    }

def main():
    results_dir = Path("RAG/results")
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6]

    print("=" * 80)
    print("TMD ALPHA PARAMETER TUNING RESULTS")
    print("=" * 80)
    print()
    print("Alpha = TMD weight (1-alpha = vector weight)")
    print()
    print(f"{'Alpha':<8} {'P@1':>8} {'P@5':>8} {'P@10':>8} {'MRR':>8} {'nDCG':>8}")
    print("-" * 80)

    all_metrics = []

    for alpha in alphas:
        filepath = results_dir / f"tmd_alpha_{alpha}_oct4.jsonl"

        if not filepath.exists():
            print(f"{alpha:<8.1f} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            continue

        # Load per-query results
        results = []
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        # Compute metrics
        metrics = compute_metrics(results)

        if not metrics:
            print(f"{alpha:<8.1f} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            continue

        p1 = metrics['p_at_1']
        p5 = metrics['p_at_5']
        p10 = metrics['p_at_10']
        mrr = metrics['mrr']
        ndcg = metrics['ndcg']

        all_metrics.append((alpha, metrics))

        print(f"{alpha:<8.1f} {p1*100:>7.1f}% {p5*100:>7.1f}% {p10*100:>7.1f}% {mrr:>8.4f} {ndcg:>8.4f}")

    print("-" * 80)
    print()

    if all_metrics:
        # Find best
        best_p1 = max(all_metrics, key=lambda x: x[1]['p_at_1'])
        best_p5 = max(all_metrics, key=lambda x: x[1]['p_at_5'])
        best_mrr = max(all_metrics, key=lambda x: x[1]['mrr'])

        print("BEST RESULTS:")
        print(f"  Best P@1:  alpha={best_p1[0]:.1f} ({best_p1[1]['p_at_1']*100:.1f}%)")
        print(f"  Best P@5:  alpha={best_p5[0]:.1f} ({best_p5[1]['p_at_5']*100:.1f}%)")
        print(f"  Best MRR:  alpha={best_mrr[0]:.1f} ({best_mrr[1]['mrr']:.4f})")
        print()

        print("RECOMMENDATION:")
        if best_p5[1]['p_at_5'] == best_p1[1]['p_at_1']:
            print(f"  ✨ Use alpha={best_p5[0]:.1f} for optimal performance")
        else:
            print(f"  ✨ Use alpha={best_p5[0]:.1f} for best P@5 ({best_p5[1]['p_at_5']*100:.1f}%)")
            print(f"     Or alpha={best_p1[0]:.1f} for best P@1 ({best_p1[1]['p_at_1']*100:.1f}%)")

    print("=" * 80)

if __name__ == "__main__":
    main()
