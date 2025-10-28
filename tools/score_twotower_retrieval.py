#!/usr/bin/env python3
"""
Score two-tower retrieval results.

Usage:
    python3 tools/score_twotower_retrieval.py \
        --hits artifacts/eval/flat_hits_ep1.jsonl \
        --out artifacts/eval/flat_scores_ep1.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hits', type=Path, required=True,
                        help='JSONL with retrieval hits')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    print("=" * 80)
    print("TWO-TOWER RETRIEVAL SCORING")
    print("=" * 80)
    print(f"Hits: {args.hits}")
    print()

    # Load hits
    print("Loading hits...")
    hits = []
    with open(args.hits) as f:
        for line in f:
            hits.append(json.loads(line))
    print(f"  Loaded {len(hits)} queries")
    print()

    # Score metrics (diagonal pairing: truth_idx = query_idx)
    print("Computing metrics...")

    n = len(hits)
    r_at_1 = 0
    r_at_5 = 0
    r_at_10 = 0
    r_at_20 = 0
    r_at_50 = 0
    contain_at_20 = 0
    contain_at_50 = 0
    mrr_sum = 0.0

    for record in hits:
        query_idx = record['query_idx']
        hit_indices = record['hit_indices']

        # Ground truth: diagonal pairing (query i → target i)
        truth_idx = query_idx

        # Check if truth_idx in top-k
        if truth_idx in hit_indices[:1]:
            r_at_1 += 1
        if truth_idx in hit_indices[:5]:
            r_at_5 += 1
        if truth_idx in hit_indices[:10]:
            r_at_10 += 1
        if truth_idx in hit_indices[:20]:
            r_at_20 += 1
            contain_at_20 += 1
        if truth_idx in hit_indices[:50]:
            r_at_50 += 1
            contain_at_50 += 1

        # MRR
        try:
            rank = hit_indices.index(truth_idx) + 1
            mrr_sum += 1.0 / rank
        except ValueError:
            pass

    # Metrics
    metrics = {
        'n_queries': n,
        'R@1': 100.0 * r_at_1 / n,
        'R@5': 100.0 * r_at_5 / n,
        'R@10': 100.0 * r_at_10 / n,
        'R@20': 100.0 * r_at_20 / n,
        'R@50': 100.0 * r_at_50 / n,
        'Contain@20': 100.0 * contain_at_20 / n,
        'Contain@50': 100.0 * contain_at_50 / n,
        'MRR': mrr_sum / n,
    }

    # Print
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Queries: {n}")
    print(f"R@1:         {metrics['R@1']:.1f}%")
    print(f"R@5:         {metrics['R@5']:.1f}%")
    print(f"R@10:        {metrics['R@10']:.1f}%")
    print(f"R@20:        {metrics['R@20']:.1f}%")
    print(f"R@50:        {metrics['R@50']:.1f}%")
    print(f"Contain@20:  {metrics['Contain@20']:.1f}%")
    print(f"Contain@50:  {metrics['Contain@50']:.1f}%")
    print(f"MRR:         {metrics['MRR']:.4f}")
    print("=" * 80)
    print()

    # Gate check
    print("GATE CHECK (R@5 > 5%)")
    if metrics['R@5'] > 5.0:
        print(f"✅ PASS: R@5 = {metrics['R@5']:.1f}% > 5%")
    else:
        print(f"❌ FAIL: R@5 = {metrics['R@5']:.1f}% ≤ 5%")
    print("=" * 80)

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved: {args.out}")
    print()


if __name__ == '__main__':
    main()
