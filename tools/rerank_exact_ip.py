#!/usr/bin/env python3
"""
Re-rank retrieval hits with exact inner product.

Strategy: Retrieve top-K candidates (e.g., 200) via IVF, then re-rank to top-k (e.g., 50)
using exact IP computation. Often gives +1-3pp R@5 for free.

Usage:
    python3 tools/rerank_exact_ip.py \
        --hits artifacts/eval/hits200.jsonl \
        --queries artifacts/eval/q_clean.npy \
        --corpus artifacts/eval/p_full_corpus.npy \
        --topk 50 \
        --out artifacts/eval/hits50_reranked.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hits', type=Path, required=True,
                        help='JSONL with initial retrieval hits')
    parser.add_argument('--queries', type=Path, required=True,
                        help='NPY with query vectors')
    parser.add_argument('--corpus', type=Path, required=True,
                        help='NPY with corpus vectors')
    parser.add_argument('--topk', type=int, default=50,
                        help='Top-k after re-ranking')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    print("=" * 80)
    print("EXACT IP RE-RANKING")
    print("=" * 80)
    print(f"Hits: {args.hits}")
    print(f"Queries: {args.queries}")
    print(f"Corpus: {args.corpus}")
    print(f"Re-rank to top-{args.topk}")
    print()

    # Load
    print("Loading...")
    queries = np.load(args.queries)
    corpus = np.load(args.corpus)
    print(f"  Queries: {queries.shape}")
    print(f"  Corpus: {corpus.shape}")
    print()

    # Load hits
    hits = []
    with open(args.hits) as f:
        for line in f:
            hits.append(json.loads(line))
    print(f"  Loaded {len(hits)} hit records")
    print()

    # Re-rank
    print("Re-ranking with exact IP...")
    reranked = []

    for i, record in enumerate(hits):
        query_idx = record['query_idx']
        candidate_indices = record['hit_indices']

        # Get query vector
        q = queries[query_idx]

        # Get candidate vectors
        candidates = corpus[candidate_indices]

        # Compute exact IP
        scores = np.dot(candidates, q)

        # Sort by score (descending)
        sorted_idxs = np.argsort(-scores)

        # Re-rank
        reranked_indices = [candidate_indices[idx] for idx in sorted_idxs[:args.topk]]
        reranked_scores = [float(scores[idx]) for idx in sorted_idxs[:args.topk]]

        reranked.append({
            'query_idx': query_idx,
            'hit_indices': reranked_indices,
            'hit_scores': reranked_scores,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(hits)}")

    print(f"  Re-ranked {len(reranked)} queries")
    print()

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        for record in reranked:
            f.write(json.dumps(record) + '\n')

    print(f"✅ Saved: {args.out}")
    print("=" * 80)


if __name__ == '__main__':
    main()
