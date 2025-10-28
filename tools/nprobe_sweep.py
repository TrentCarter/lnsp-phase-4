#!/usr/bin/env python3
"""
FAISS nprobe Tuning Sweep

Runs queries with different nprobe values to find optimal latency/accuracy trade-off.
Compares IVF results against FLAT truth to measure accuracy degradation.

Usage:
    python tools/nprobe_sweep.py \
        --index artifacts/faiss/p_ivf.faiss \
        --flat artifacts/faiss/p_flat_ip.faiss \
        --queries artifacts/eval/eval_queries.npy \
        --nprobe-values 4,8,12,16 \
        --n-queries 500 \
        --out artifacts/tuning/nprobe_sweep.json
"""

import argparse
import json
import time
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict


def run_nprobe_sweep(
    ivf_path: str,
    flat_path: str,
    queries: np.ndarray,
    nprobe_values: List[int],
    k: int = 10
) -> List[Dict]:
    """
    Run nprobe sweep and measure latency + accuracy.

    Args:
        ivf_path: Path to IVF index
        flat_path: Path to FLAT truth index
        queries: Query vectors (N, D)
        nprobe_values: List of nprobe values to test
        k: Top-K results to retrieve

    Returns:
        List of results dicts (one per nprobe value)
    """
    print(f"Loading IVF index: {ivf_path}")
    ivf_index = faiss.read_index(ivf_path)

    print(f"Loading FLAT index: {flat_path}")
    flat_index = faiss.read_index(flat_path)

    print(f"Running sweep with {len(queries)} queries, K={k}")
    print()

    # Get FLAT truth (once)
    print("Computing FLAT truth...")
    start = time.perf_counter()
    flat_distances, flat_indices = flat_index.search(queries.astype('float32'), k)
    flat_time = (time.perf_counter() - start) / len(queries) * 1000  # ms/query
    print(f"  FLAT: {flat_time:.3f} ms/query (truth baseline)")
    print()

    results = []

    for nprobe in nprobe_values:
        print(f"Testing nprobe={nprobe}...")

        # Set nprobe
        ivf_index.nprobe = nprobe

        # Warmup (5 queries)
        _ = ivf_index.search(queries[:5].astype('float32'), k)

        # Measure latency
        latencies = []
        for i in range(len(queries)):
            start = time.perf_counter()
            _, _ = ivf_index.search(queries[i:i+1].astype('float32'), k)
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)

        latencies = np.array(latencies)

        # Get IVF results (batch for accuracy comparison)
        ivf_distances, ivf_indices = ivf_index.search(queries.astype('float32'), k)

        # Compute accuracy metrics
        recall_at_k = []
        for i in range(len(queries)):
            flat_set = set(flat_indices[i])
            ivf_set = set(ivf_indices[i])
            recall = len(flat_set & ivf_set) / k
            recall_at_k.append(recall)

        recall_at_k = np.array(recall_at_k)

        # Collect results
        result = {
            'nprobe': nprobe,
            'latency_ms': {
                'mean': float(latencies.mean()),
                'p50': float(np.percentile(latencies, 50)),
                'p95': float(np.percentile(latencies, 95)),
                'p99': float(np.percentile(latencies, 99))
            },
            'accuracy': {
                'mean_recall': float(recall_at_k.mean()),
                'min_recall': float(recall_at_k.min()),
                'p5_recall': float(np.percentile(recall_at_k, 5)),
                'pct_perfect': float((recall_at_k == 1.0).mean())
            },
            'delta_vs_flat': {
                'r_at_k_pp': float((recall_at_k.mean() - 1.0) * 100),  # percentage points
                'latency_ratio': float(latencies.mean() / flat_time)
            }
        }

        results.append(result)

        # Print summary
        print(f"  Latency: {result['latency_ms']['mean']:.2f} ms (P95: {result['latency_ms']['p95']:.2f} ms)")
        print(f"  Recall@{k}: {result['accuracy']['mean_recall']*100:.2f}% (ΔR@{k}: {result['delta_vs_flat']['r_at_k_pp']:.2f}pp)")
        print()

    return results


def print_summary(results: List[Dict], target_latency_ms: float = 8.0):
    """Print formatted summary and recommendation."""
    print("="*80)
    print("NPROBE SWEEP SUMMARY")
    print("="*80)
    print()

    # Table header
    print(f"{'nprobe':<8} {'P95 (ms)':<10} {'Mean (ms)':<10} {'Recall':<10} {'ΔR@10 (pp)':<12} {'Perfect':<10}")
    print("-" * 80)

    # Table rows
    for r in results:
        nprobe = r['nprobe']
        p95 = r['latency_ms']['p95']
        mean = r['latency_ms']['mean']
        recall = r['accuracy']['mean_recall'] * 100
        delta_pp = r['delta_vs_flat']['r_at_k_pp']
        perfect = r['accuracy']['pct_perfect'] * 100

        print(f"{nprobe:<8} {p95:<10.2f} {mean:<10.2f} {recall:<9.2f}% {delta_pp:<11.2f}pp {perfect:<9.1f}%")

    print()

    # Recommendation
    print("RECOMMENDATIONS:")
    print()

    # Find best nprobe for different scenarios
    latency_critical = None
    balanced = None
    accuracy_critical = None

    for r in results:
        # Latency-critical: Lowest P95, accept up to -1.5pp accuracy loss
        if r['delta_vs_flat']['r_at_k_pp'] >= -1.5:
            if latency_critical is None or r['latency_ms']['p95'] < latency_critical['latency_ms']['p95']:
                latency_critical = r

        # Balanced: Best trade-off (P95 ≤ target, ΔR@10 ≥ -0.5pp)
        if r['latency_ms']['p95'] <= target_latency_ms and r['delta_vs_flat']['r_at_k_pp'] >= -0.5:
            if balanced is None or r['accuracy']['mean_recall'] > balanced['accuracy']['mean_recall']:
                balanced = r

        # Accuracy-critical: Highest recall, accept P95 up to 15ms
        if r['latency_ms']['p95'] <= 15.0:
            if accuracy_critical is None or r['accuracy']['mean_recall'] > accuracy_critical['accuracy']['mean_recall']:
                accuracy_critical = r

    # Print recommendations
    if latency_critical:
        print(f"⚡ LATENCY-CRITICAL: nprobe={latency_critical['nprobe']}")
        print(f"   P95: {latency_critical['latency_ms']['p95']:.2f} ms, ΔR@10: {latency_critical['delta_vs_flat']['r_at_k_pp']:.2f}pp")
        print(f"   Use case: High-throughput serving, accept slight accuracy loss")
        print()

    if balanced:
        print(f"⭐ BALANCED (RECOMMENDED): nprobe={balanced['nprobe']}")
        print(f"   P95: {balanced['latency_ms']['p95']:.2f} ms, ΔR@10: {balanced['delta_vs_flat']['r_at_k_pp']:.2f}pp")
        print(f"   Use case: Production default (good latency + accuracy)")
        print()
    else:
        print(f"⚠️  WARNING: No nprobe value meets balanced criteria (P95 ≤ {target_latency_ms}ms, ΔR@10 ≥ -0.5pp)")
        print(f"   Consider: Larger nlist, better quantization, or scale up hardware")
        print()

    if accuracy_critical:
        print(f"🎯 ACCURACY-CRITICAL: nprobe={accuracy_critical['nprobe']}")
        print(f"   P95: {accuracy_critical['latency_ms']['p95']:.2f} ms, ΔR@10: {accuracy_critical['delta_vs_flat']['r_at_k_pp']:.2f}pp")
        print(f"   Use case: Batch processing, accuracy paramount")
        print()

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="FAISS nprobe tuning sweep")
    parser.add_argument('--index', type=str, required=True, help="Path to IVF index")
    parser.add_argument('--flat', type=str, required=True, help="Path to FLAT truth index")
    parser.add_argument('--queries', type=str, help="Path to query vectors .npy (optional, will generate if missing)")
    parser.add_argument('--nprobe-values', type=str, default='4,8,12,16', help="Comma-separated nprobe values")
    parser.add_argument('--n-queries', type=int, default=500, help="Number of queries to test")
    parser.add_argument('--k', type=int, default=10, help="Top-K results")
    parser.add_argument('--target-p95', type=float, default=8.0, help="Target P95 latency (ms)")
    parser.add_argument('--out', type=str, help="Output JSON path")

    args = parser.parse_args()

    # Parse nprobe values
    nprobe_values = [int(x.strip()) for x in args.nprobe_values.split(',')]

    # Load or generate queries
    if args.queries and Path(args.queries).exists():
        print(f"Loading queries from: {args.queries}")
        queries = np.load(args.queries)
        queries = queries[:args.n_queries]  # Limit to n_queries
    else:
        print(f"Generating {args.n_queries} random queries...")
        flat_index = faiss.read_index(args.flat)
        dimension = flat_index.d
        queries = np.random.randn(args.n_queries, dimension).astype('float32')
        # L2 normalize
        norms = np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
        queries = queries / norms

    print(f"Queries: {queries.shape}")
    print()

    # Run sweep
    results = run_nprobe_sweep(args.index, args.flat, queries, nprobe_values, k=args.k)

    # Print summary
    print_summary(results, target_latency_ms=args.target_p95)

    # Save results
    if args.out:
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_queries': len(queries),
            'k': args.k,
            'target_p95_ms': args.target_p95,
            'results': results
        }

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✅ Saved results to: {args.out}")

    return 0


if __name__ == '__main__':
    exit(main())
