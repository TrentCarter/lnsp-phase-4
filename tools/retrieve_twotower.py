#!/usr/bin/env python3
"""
Retrieve using FAISS index for two-tower evaluation.

Usage:
    python3 tools/retrieve_twotower.py \
        --index artifacts/faiss/p_ep1_flat_ip.faiss \
        --queries artifacts/eval/q_ep1.npy \
        --topk 50 \
        --out artifacts/eval/flat_hits_ep1.jsonl
"""
import argparse
import json
from pathlib import Path

import faiss
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload-vectors', type=Path, required=True,
                        help='NPZ or NPY with payload vectors')
    parser.add_argument('--queries', type=Path, required=True,
                        help='NPY with query vectors')
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--metric', type=str, default='ip', choices=['ip', 'l2'])
    args = parser.parse_args()

    print("=" * 80)
    print("TWO-TOWER RETRIEVAL (FLAT INDEX)")
    print("=" * 80)
    print(f"Payload: {args.payload_vectors}")
    print(f"Queries: {args.queries}")
    print(f"Top-K: {args.topk}")
    print(f"Metric: {args.metric}")
    print()

    # Load payload vectors
    print("Loading payload vectors...")
    if args.payload_vectors.suffix == '.npy':
        payload = np.load(args.payload_vectors)
    else:
        data = np.load(args.payload_vectors, allow_pickle=True)
        payload = data['target_vectors']
    print(f"  Payload: {payload.shape}")

    # Load queries
    print("Loading queries...")
    queries = np.load(args.queries)
    print(f"  Queries: {queries.shape}")
    print()

    # Build FLAT index
    print("Building FLAT index...")
    d = payload.shape[1]

    if args.metric == 'ip':
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)

    index.add(payload.astype(np.float32))
    print(f"  Index size: {index.ntotal}")
    print()

    # Retrieve
    print("Retrieving...")
    distances, indices = index.search(queries.astype(np.float32), args.topk)
    print(f"  Retrieved {len(indices)} × {args.topk}")
    print()

    # Save hits
    print("Saving hits...")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, 'w') as f:
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            record = {
                'query_idx': i,
                'hit_indices': idxs.tolist(),
                'hit_scores': dists.tolist(),
            }
            f.write(json.dumps(record) + '\n')

    print(f"✅ Saved: {args.out}")
    print("=" * 80)


if __name__ == '__main__':
    main()
