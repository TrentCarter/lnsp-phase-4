#!/usr/bin/env python3
"""
nprobe Parameter Sweep
======================

Tests different nprobe values to optimize ANN recall (containment).
Runs on 1000 samples for speed.

Based on consultant recommendation: "Sweep nprobe ∈ {8,16,32,64,128}"
"""

import json
import time
from pathlib import Path
import numpy as np
import faiss

from eval_retrieval_v2 import RetrievalShim, DatasetShim, evaluate


def main():
    print("=" * 80)
    print("NPROBE PARAMETER SWEEP")
    print("=" * 80)
    print()

    # Configuration
    npz_path = Path("artifacts/lvm/wikipedia_ood_test_ctx5_v2_fresh.npz")
    payload_path = Path("artifacts/wikipedia_584k_payload.npy")
    faiss_path = Path("artifacts/wikipedia_584k_ivf_flat_ip.index")
    limit = 1000
    nprobe_values = [32, 64, 128, 256]  # User-specified sweep range

    # Load resources once
    print(f"Loading FAISS index from {faiss_path}...")
    faiss_index = faiss.read_index(str(faiss_path))

    print(f"Loading payload from {payload_path}...")
    payload = np.load(payload_path, allow_pickle=True).item()

    print(f"Loading dataset from {npz_path}...")
    dataset = DatasetShim(npz_path, limit=limit)
    print(f"Loaded {len(dataset):,} samples\n")

    # Results storage
    results = []

    # Sweep nprobe values
    for nprobe in nprobe_values:
        print(f"Testing nprobe={nprobe}...")

        # Set nprobe on index
        faiss_index.nprobe = nprobe

        # Create retriever with updated index
        retriever = RetrievalShim(faiss_index, payload)

        # Run evaluation (WITH reranking - user's config)
        start_time = time.time()
        metrics = evaluate(
            retriever, dataset,
            K_retrieve=50,
            do_mmr=True,
            mmr_lambda=0.7,
            top_final=10,
            use_seq_bias=True,
            w_same_article=0.05,
            w_next_gap=0.12,
            tau=3.0,
            directional_bonus=0.03,
        )
        elapsed = time.time() - start_time

        # Store results
        result = {
            "nprobe": nprobe,
            "Contain@20": metrics["Contain@20"],
            "Contain@50": metrics["Contain@50"],
            "R@10": metrics["R@10"],
            "R@5": metrics["R@5"],
            "R@1": metrics["R@1"],
            "p50_ms": metrics["p50_ms"],
            "p95_ms": metrics["p95_ms"],
            "total_time_s": elapsed,
        }
        results.append(result)

        print(f"  Contain@50: {metrics['Contain@50']:.1%}")
        print(f"  R@10:       {metrics['R@10']:.1%}")
        print(f"  P50 lat:    {metrics['p50_ms']:.2f}ms")
        print()

    # Save results
    output_path = Path("artifacts/lvm/nprobe_sweep_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("=" * 80)
    print("NPROBE SWEEP RESULTS")
    print("=" * 80)
    print()
    print("nprobe   Contain@20  Contain@50   R@10    R@5     R@1    P50(ms)  P95(ms)")
    print("-" * 80)
    for r in results:
        print(f"{r['nprobe']:6d}   {r['Contain@20']:9.1%}  {r['Contain@50']:9.1%}  "
              f"{r['R@10']:6.1%}  {r['R@5']:6.1%}  {r['R@1']:5.1%}  "
              f"{r['p50_ms']:7.2f}  {r['p95_ms']:7.2f}")

    print()
    print(f"Results saved to: {output_path}")
    print()

    # Analysis
    best = max(results, key=lambda x: x["Contain@50"])
    baseline = next(r for r in results if r["nprobe"] == 32)

    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print(f"Baseline (nprobe=32):  Contain@50 = {baseline['Contain@50']:.1%}")
    print(f"Best (nprobe={best['nprobe']}):       Contain@50 = {best['Contain@50']:.1%}")
    print(f"Lift:                  +{(best['Contain@50'] - baseline['Contain@50']) * 100:.2f}pp")
    print(f"Latency penalty:       +{best['p50_ms'] - baseline['p50_ms']:.2f}ms")
    print()


if __name__ == "__main__":
    main()
