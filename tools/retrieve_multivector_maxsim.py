#!/usr/bin/env python3
"""
Retrieve with multi-vector queries using MaxSim scoring.

For each query q1..q5, retrieve top-K candidates, union, then score by:
    score(q1:5, p) = max_i(qi · p)

Usage:
    python3 tools/retrieve_multivector_maxsim.py \
        --queries artifacts/eval/q_multivec.npy \
        --corpus artifacts/eval/p_full_corpus.npy \
        --topk-per-vector 50 \
        --topk-final 50 \
        --out artifacts/eval/hits_maxsim.jsonl
"""
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=Path, required=True,
                        help='Multi-vector queries [N, K, 768]')
    parser.add_argument('--corpus', type=Path, required=True,
                        help='Corpus vectors [M, 768]')
    parser.add_argument('--topk-per-vector', type=int, default=50,
                        help='Top-K per query vector')
    parser.add_argument('--topk-final', type=int, default=50,
                        help='Final top-K after MaxSim scoring')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    print("=" * 80)
    print("MULTI-VECTOR RETRIEVAL (MaxSim)")
    print("=" * 80)
    print(f"Queries: {args.queries}")
    print(f"Corpus: {args.corpus}")
    print(f"Strategy: Retrieve {args.topk_per_vector} per vector, score by MaxSim")
    print()

    # Load
    print("Loading...")
    queries = np.load(args.queries)  # [N, K, 768]
    corpus = np.load(args.corpus).astype('float32')  # [M, 768]
    print(f"  Queries: {queries.shape}")
    print(f"  Corpus: {corpus.shape}")
    print()

    N, K, D = queries.shape
    print(f"  {N} queries × {K} vectors/query × {D}D")
    print()

    # Build FLAT IP index
    print("Building FLAT IP index...")
    index = faiss.IndexFlatIP(D)
    index.add(corpus)
    print(f"  Index size: {index.ntotal}")
    print()

    # Retrieve and score
    print(f"Retrieving top-{args.topk_per_vector} per vector, scoring by MaxSim...")
    results = []

    for i in tqdm(range(N), desc="Queries"):
        q_vecs = queries[i].astype('float32')  # [K, 768]

        # Retrieve top-K for each query vector
        all_candidates = set()
        distances_per_vec = []

        for k in range(K):
            dists, idxs = index.search(q_vecs[k:k+1], args.topk_per_vector)
            all_candidates.update(idxs[0].tolist())
            distances_per_vec.append((idxs[0], dists[0]))

        # Union candidates
        candidates = list(all_candidates)

        # Compute MaxSim scores for all candidates
        maxsim_scores = []
        for cand_idx in candidates:
            cand_vec = corpus[cand_idx]
            # MaxSim = max over K query vectors
            scores = [np.dot(q_vecs[k], cand_vec) for k in range(K)]
            maxsim_scores.append(max(scores))

        # Sort by MaxSim (descending)
        sorted_pairs = sorted(zip(candidates, maxsim_scores),
                             key=lambda x: x[1], reverse=True)

        # Take top-K
        top_indices = [int(idx) for idx, _ in sorted_pairs[:args.topk_final]]
        top_scores = [float(score) for _, score in sorted_pairs[:args.topk_final]]

        results.append({
            'query_idx': i,
            'hit_indices': top_indices,
            'hit_scores': top_scores,
        })

    print()

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        for record in results:
            f.write(json.dumps(record) + '\n')

    print(f"✅ Retrieved {len(results)} queries")
    print(f"✅ Saved: {args.out}")
    print("=" * 80)


if __name__ == '__main__':
    main()
