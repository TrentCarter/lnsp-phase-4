#!/usr/bin/env python3
"""
Oracle check: query P index with its own vectors (should get @1 ≈ 100%).

Usage:
    python3 tools/oracle_check_p_index.py \
        --payload-vectors artifacts/eval/p_ep1.npy \
        --n-samples 500
"""
import argparse
from pathlib import Path

import faiss
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload-vectors', type=Path, required=True)
    parser.add_argument('--n-samples', type=int, default=500)
    args = parser.parse_args()

    print("=" * 80)
    print("ORACLE CHECK: P INDEX")
    print("=" * 80)
    print(f"Payload: {args.payload_vectors}")
    print(f"Samples: {args.n_samples}")
    print()

    # Load payload
    print("Loading payload vectors...")
    payload = np.load(args.payload_vectors)
    print(f"  Shape: {payload.shape}")
    print()

    # Build FLAT IP index
    print("Building FLAT IP index...")
    d = payload.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(payload.astype(np.float32))
    print(f"  Index size: {index.ntotal}")
    print()

    # Sample queries (use payload itself)
    n_samples = min(args.n_samples, len(payload))
    sample_indices = np.random.choice(len(payload), n_samples, replace=False)
    queries = payload[sample_indices]

    # Retrieve
    print("Querying index with sampled payload vectors...")
    distances, indices = index.search(queries.astype(np.float32), k=1)
    print()

    # Check if top-1 matches original index
    matches = (indices[:, 0] == sample_indices).sum()
    accuracy = 100.0 * matches / n_samples

    print("=" * 80)
    print("ORACLE CHECK RESULTS")
    print("=" * 80)
    print(f"Samples: {n_samples}")
    print(f"Top-1 matches: {matches}")
    print(f"Accuracy: {accuracy:.1f}%")
    print()

    if accuracy >= 99.5:
        print("✅ PASS: Index correctly retrieves its own vectors")
    elif accuracy >= 95.0:
        print("⚠️  WARN: Some mismatches (likely quantization)")
    else:
        print("❌ FAIL: Index not retrieving correctly!")
        print("   Check: metric type, normalization, vector corruption")

    print("=" * 80)


if __name__ == '__main__':
    main()
