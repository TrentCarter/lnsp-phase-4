#!/usr/bin/env python3
"""
Mine near-miss negatives from top-K FAISS hits (excluding gold).

Usage:
    python3 tools/mine_nearmiss.py \
        --queries-npy artifacts/eval/q_clean.npy \
        --corpus-npy artifacts/eval/p_full_corpus.npy \
        --gold-npz artifacts/lvm/eval_clean_disjoint.npz \
        --topk 5 --per-query 1 \
        --out artifacts/mined/nearmiss.jsonl
"""
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm


def load_npy(path):
    """Load and L2-normalize vectors."""
    a = np.load(path).astype('float32')
    norms = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return a / norms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries-npy", required=True, help="Query vectors")
    ap.add_argument("--corpus-npy", required=True, help="Corpus vectors")
    ap.add_argument("--gold-npz", required=True, help="NPZ with ground truth")
    ap.add_argument("--corpus-offset", type=int, default=None,
                    help="Offset for truth indices (auto-detect from gold NPZ if not provided)")
    ap.add_argument("--topk", type=int, default=5, help="Top-K candidates to mine from")
    ap.add_argument("--per-query", type=int, default=1, help="Negatives per query")
    ap.add_argument("--out", required=True, help="Output JSONL")
    args = ap.parse_args()

    print("=" * 80)
    print("NEAR-MISS NEGATIVE MINING")
    print("=" * 80)
    print(f"Queries: {args.queries_npy}")
    print(f"Corpus: {args.corpus_npy}")
    print(f"Top-K: {args.topk}, Per-query: {args.per_query}")
    print()

    # Load
    print("Loading vectors...")
    queries = load_npy(args.queries_npy)
    corpus = load_npy(args.corpus_npy)
    print(f"  Queries: {queries.shape}")
    print(f"  Corpus: {corpus.shape}")
    print()

    # Load gold NPZ to determine corpus offset
    if args.corpus_offset is None:
        print("Auto-detecting corpus offset from gold NPZ...")
        data = np.load(args.gold_npz, allow_pickle=True)
        # Assume eval targets are at end of corpus
        n_eval = len(data['target_vectors'])
        n_corpus = len(corpus)
        args.corpus_offset = n_corpus - n_eval
        print(f"  Detected offset: {args.corpus_offset}")
        print()

    # Build FLAT IP index
    print("Building FLAT IP index...")
    d = corpus.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(corpus)
    print(f"  Index size: {index.ntotal}")
    print()

    # Mine
    print(f"Mining top-{args.topk} candidates...")
    distances, indices = index.search(queries, args.topk)
    print()

    # Filter out gold and sample negatives
    print("Filtering and sampling negatives...")
    mined = []

    for i in tqdm(range(len(queries)), desc="Mining"):
        gold_idx = args.corpus_offset + i
        candidates = indices[i]

        # Remove gold
        negatives = [int(c) for c in candidates if c != gold_idx]

        if len(negatives) == 0:
            # Fallback: use random from corpus
            negatives = list(np.random.choice(len(corpus), args.per_query, replace=False))

        # Sample per_query negatives
        if len(negatives) >= args.per_query:
            sampled = list(np.random.choice(negatives, args.per_query, replace=False))
        else:
            sampled = negatives + list(np.random.choice(negatives, args.per_query - len(negatives), replace=True))

        mined.append({
            'query_idx': i,
            'gold_idx': gold_idx,
            'negative_indices': [int(x) for x in sampled],
        })

    print()

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        for record in mined:
            f.write(json.dumps(record) + '\n')

    print(f"✅ Mined {len(mined)} queries")
    print(f"✅ Saved: {args.out}")
    print("=" * 80)


if __name__ == '__main__':
    main()
