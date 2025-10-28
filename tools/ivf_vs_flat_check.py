#!/usr/bin/env python3
"""
IVF vs FLAT Index Sanity Check

Validates that serving IVF index agrees with truth FLAT index on random queries.
Target: ≥95% overlap at K=10 (IVF approximation should be highly accurate).

Usage:
    python tools/ivf_vs_flat_check.py --n 100 --k 10 \
        --flat artifacts/faiss/p_flat_ip.faiss \
        --ivf artifacts/faiss/p_ivf.faiss
"""

import argparse
import numpy as np
import faiss
from pathlib import Path


def check_index_agreement(flat_path: str, ivf_path: str, n_queries: int, k: int):
    """
    Compare IVF vs FLAT retrieval on random queries.

    Args:
        flat_path: Path to FLAT IP index (truth)
        ivf_path: Path to IVF-Flat index (serving)
        n_queries: Number of random queries to test
        k: Top-K results to compare

    Returns:
        dict with overlap statistics
    """
    print(f"Loading FLAT index: {flat_path}")
    flat_index = faiss.read_index(flat_path)

    print(f"Loading IVF index: {ivf_path}")
    ivf_index = faiss.read_index(ivf_path)

    print(f"FLAT: {flat_index.ntotal} vectors, dimension {flat_index.d}")
    print(f"IVF: {ivf_index.ntotal} vectors, {ivf_index.nlist} clusters, nprobe={ivf_index.nprobe}")

    # Sanity checks
    assert flat_index.ntotal == ivf_index.ntotal, "Index sizes must match!"
    assert flat_index.d == ivf_index.d, "Dimensions must match!"

    # Generate random queries (sample from index for realistic distribution)
    print(f"\nGenerating {n_queries} random queries...")
    dimension = flat_index.d
    query_indices = np.random.randint(0, flat_index.ntotal, size=n_queries)

    # Reconstruct vectors from flat index (use as queries)
    queries = np.zeros((n_queries, dimension), dtype=np.float32)
    for i, idx in enumerate(query_indices):
        queries[i] = flat_index.reconstruct(int(idx))

    # L2 normalize queries
    norms = np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
    queries = queries / norms

    print(f"Running searches (K={k})...")

    # Search both indexes
    flat_distances, flat_indices = flat_index.search(queries.astype('float32'), k)
    ivf_distances, ivf_indices = ivf_index.search(queries.astype('float32'), k)

    # Compute overlap statistics
    overlaps = []
    for i in range(n_queries):
        flat_set = set(flat_indices[i])
        ivf_set = set(ivf_indices[i])
        overlap = len(flat_set & ivf_set)
        overlaps.append(overlap / k)

    overlaps = np.array(overlaps)

    results = {
        'n_queries': n_queries,
        'k': k,
        'mean_overlap': float(overlaps.mean()),
        'min_overlap': float(overlaps.min()),
        'p5_overlap': float(np.percentile(overlaps, 5)),
        'p95_overlap': float(np.percentile(overlaps, 95)),
        'pct_perfect': float((overlaps == 1.0).mean()),
        'pct_above_95': float((overlaps >= 0.95).mean()),
        'nprobe': ivf_index.nprobe,
        'nlist': ivf_index.nlist
    }

    return results, overlaps


def print_results(results: dict, overlaps: np.ndarray):
    """Print formatted results."""
    print("\n" + "="*60)
    print("IVF vs FLAT INDEX AGREEMENT CHECK")
    print("="*60)
    print(f"Queries tested: {results['n_queries']}")
    print(f"Top-K: {results['k']}")
    print(f"IVF config: nlist={results['nlist']}, nprobe={results['nprobe']}")
    print()
    print("Overlap Statistics (% of K results matching FLAT truth):")
    print(f"  Mean:    {results['mean_overlap']*100:.2f}%")
    print(f"  Min:     {results['min_overlap']*100:.2f}%")
    print(f"  P5:      {results['p5_overlap']*100:.2f}%")
    print(f"  P95:     {results['p95_overlap']*100:.2f}%")
    print()
    print(f"  Perfect (100%): {results['pct_perfect']*100:.1f}% of queries")
    print(f"  ≥95% overlap:   {results['pct_above_95']*100:.1f}% of queries")
    print()

    # Gate check
    if results['mean_overlap'] >= 0.95:
        print("✅ PASS: Mean overlap ≥95% (IVF approximation is highly accurate)")
        return True
    else:
        print(f"⚠️  WARN: Mean overlap {results['mean_overlap']*100:.2f}% < 95% target")
        print(f"   Consider: increase nprobe (current: {results['nprobe']})")
        return False


def main():
    parser = argparse.ArgumentParser(description="IVF vs FLAT index sanity check")
    parser.add_argument('--flat', type=str, required=True, help="Path to FLAT IP index")
    parser.add_argument('--ivf', type=str, required=True, help="Path to IVF-Flat index")
    parser.add_argument('--n', type=int, default=100, help="Number of queries to test")
    parser.add_argument('--k', type=int, default=10, help="Top-K results to compare")
    parser.add_argument('--out', type=str, help="Optional output JSON path")

    args = parser.parse_args()

    # Validate paths
    if not Path(args.flat).exists():
        print(f"❌ ERROR: FLAT index not found: {args.flat}")
        return 1

    if not Path(args.ivf).exists():
        print(f"❌ ERROR: IVF index not found: {args.ivf}")
        return 1

    # Run check
    results, overlaps = check_index_agreement(args.flat, args.ivf, args.n, args.k)

    # Print results
    passed = print_results(results, overlaps)

    # Save results if requested
    if args.out:
        import json
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Saved results to: {args.out}")

    return 0 if passed else 1


if __name__ == '__main__':
    exit(main())
