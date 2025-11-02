#!/usr/bin/env python3
"""
P1 Tool: Measure Coherence Drift (584k vs 790k)
==============================================

Compares semantic coherence between baseline (584k) and current (790k) datasets.

Metrics:
- Mean adjacency cosine similarity: avg(cos(t[i], t[i+1]))
- Distribution percentiles (p10, p25, p50, p75, p90)
- Low-coherence ratio: % of pairs with cos < 0.30
- Article-level coherence (if metadata available)

Usage:
    ./.venv/bin/python tools/measure_coherence_drift.py \
      --baseline artifacts/lvm/training_sequences_ctx5_584k_fresh.npz \
      --current artifacts/lvm/training_sequences_ctx5.npz \
      --out artifacts/lvm/coherence_comparison.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

def compute_adjacency_coherence(vectors, positions=None, name="Dataset"):
    """
    Compute adjacency coherence metrics for a vector dataset.

    Args:
        vectors: (N, 768) array of target vectors
        positions: Optional (N,) array of sequence positions
        name: Dataset name for logging

    Returns:
        dict with coherence metrics
    """
    print(f"\n{'='*60}")
    print(f"Computing adjacency coherence: {name}")
    print(f"{'='*60}")

    N = len(vectors)
    print(f"Total vectors: {N:,}")

    # Compute cos(t[i], t[i+1]) for all adjacent pairs
    # Assuming vectors are already L2-normalized
    adjacency_cos = np.sum(vectors[:-1] * vectors[1:], axis=1)

    # Statistics
    mean_cos = adjacency_cos.mean()
    std_cos = adjacency_cos.std()
    p10 = np.percentile(adjacency_cos, 10)
    p25 = np.percentile(adjacency_cos, 25)
    p50 = np.percentile(adjacency_cos, 50)
    p75 = np.percentile(adjacency_cos, 75)
    p90 = np.percentile(adjacency_cos, 90)

    # Low-coherence ratio
    low_coherence_ratio = (adjacency_cos < 0.30).sum() / len(adjacency_cos)

    # Distribution histogram
    bins = np.arange(-0.2, 1.01, 0.1)
    hist, _ = np.histogram(adjacency_cos, bins=bins)

    print(f"\nAdjacency cosine statistics:")
    print(f"   Mean:     {mean_cos:.4f}")
    print(f"   Std:      {std_cos:.4f}")
    print(f"   p10:      {p10:.4f}")
    print(f"   p25:      {p25:.4f}")
    print(f"   p50:      {p50:.4f}")
    print(f"   p75:      {p75:.4f}")
    print(f"   p90:      {p90:.4f}")
    print(f"   Low (<0.30): {low_coherence_ratio:.2%}")

    print(f"\nDistribution:")
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        count = hist[i]
        pct = 100 * count / len(adjacency_cos)
        bar = "█" * int(pct / 2)
        print(f"   [{low:4.1f}, {high:4.1f}): {count:7d} ({pct:5.1f}%) {bar}")

    # Build result dict
    result = {
        "name": name,
        "total_vectors": int(N),
        "total_pairs": int(len(adjacency_cos)),
        "mean_cosine": float(mean_cos),
        "std_cosine": float(std_cos),
        "percentiles": {
            "p10": float(p10),
            "p25": float(p25),
            "p50": float(p50),
            "p75": float(p75),
            "p90": float(p90),
        },
        "low_coherence_ratio": float(low_coherence_ratio),
        "distribution": {
            "bins": bins.tolist(),
            "counts": hist.tolist(),
        },
    }

    return result


def compare_coherence(baseline_result, current_result):
    """
    Compare two coherence measurements and identify degradation.
    """
    print(f"\n{'='*60}")
    print(f"COHERENCE DRIFT ANALYSIS")
    print(f"{'='*60}")

    base_mean = baseline_result["mean_cosine"]
    curr_mean = current_result["mean_cosine"]
    drift = curr_mean - base_mean
    drift_pct = 100 * drift / base_mean if base_mean > 0 else 0.0

    print(f"\nMean adjacency cosine:")
    print(f"   Baseline (584k): {base_mean:.4f}")
    print(f"   Current (790k):  {curr_mean:.4f}")
    print(f"   Drift:           {drift:+.4f} ({drift_pct:+.1f}%)")

    if abs(drift_pct) < 2.0:
        print(f"   ✅ Negligible drift (<2%)")
    elif drift < 0 and abs(drift_pct) < 10.0:
        print(f"   ⚠️  Moderate degradation (2-10%)")
    elif drift < 0:
        print(f"   ❌ Significant degradation (>10%)")
    else:
        print(f"   ✅ Improvement!")

    # Percentile comparison
    print(f"\nPercentile comparison:")
    for key in ["p10", "p25", "p50", "p75", "p90"]:
        base_val = baseline_result["percentiles"][key]
        curr_val = current_result["percentiles"][key]
        delta = curr_val - base_val
        print(f"   {key}: {base_val:.4f} → {curr_val:.4f} ({delta:+.4f})")

    # Low-coherence ratio
    base_low = baseline_result["low_coherence_ratio"]
    curr_low = current_result["low_coherence_ratio"]
    delta_low = curr_low - base_low

    print(f"\nLow-coherence ratio (<0.30):")
    print(f"   Baseline (584k): {base_low:.2%}")
    print(f"   Current (790k):  {curr_low:.2%}")
    print(f"   Change:          {delta_low:+.2%}")

    if delta_low > 0.05:
        print(f"   ❌ Significant increase in low-coherence pairs!")
    elif delta_low > 0.02:
        print(f"   ⚠️  Moderate increase in low-coherence pairs")
    else:
        print(f"   ✅ Low-coherence ratio stable")

    # Overall verdict
    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")

    if drift_pct < -5.0 or delta_low > 0.05:
        print(f"❌ SIGNIFICANT QUALITY DEGRADATION")
        print(f"   Recommendation: Filter low-quality articles")
        print(f"   Expected improvement: {-drift_pct:.1f}% coherence recovery")
    elif drift_pct < -2.0 or delta_low > 0.02:
        print(f"⚠️  MODERATE QUALITY DEGRADATION")
        print(f"   Recommendation: Consider filtering or ctx=7")
    else:
        print(f"✅ QUALITY STABLE")
        print(f"   Root cause is likely NOT dataset quality")
        print(f"   Investigate other factors (architecture, hyperparams)")

    comparison = {
        "baseline_mean": float(base_mean),
        "current_mean": float(curr_mean),
        "drift": float(drift),
        "drift_pct": float(drift_pct),
        "baseline_low_coherence": float(base_low),
        "current_low_coherence": float(curr_low),
        "delta_low_coherence": float(delta_low),
        "verdict": "significant_degradation" if (drift_pct < -5.0 or delta_low > 0.05) else (
            "moderate_degradation" if (drift_pct < -2.0 or delta_low > 0.02) else "stable"
        ),
    }

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Measure coherence drift between datasets")
    parser.add_argument("--baseline", required=True, help="Path to 584k baseline NPZ")
    parser.add_argument("--current", required=True, help="Path to 790k current NPZ")
    parser.add_argument("--out", required=True, help="Output JSON file")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    current_path = Path(args.current)
    output_path = Path(args.out)

    if not baseline_path.exists():
        print(f"❌ Baseline file not found: {baseline_path}")
        return 1

    if not current_path.exists():
        print(f"❌ Current file not found: {current_path}")
        return 1

    print(f"{'='*60}")
    print(f"P1: COHERENCE DRIFT MEASUREMENT")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_path}")
    print(f"Current:  {current_path}")
    print(f"Output:   {output_path}")

    # Load datasets
    print(f"\n📂 Loading baseline (584k)...")
    baseline_data = np.load(baseline_path, allow_pickle=True)
    baseline_vectors = baseline_data['target_vectors']
    baseline_positions = baseline_data.get('sequence_positions', None)

    print(f"📂 Loading current (790k)...")
    current_data = np.load(current_path, allow_pickle=True)
    current_vectors = current_data['target_vectors']
    current_positions = current_data.get('sequence_positions', None)

    # Compute coherence
    baseline_result = compute_adjacency_coherence(
        baseline_vectors,
        baseline_positions,
        name="Baseline (584k)"
    )

    current_result = compute_adjacency_coherence(
        current_vectors,
        current_positions,
        name="Current (790k)"
    )

    # Compare
    comparison = compare_coherence(baseline_result, current_result)

    # Save results
    output = {
        "analysis_date": datetime.now().isoformat(),
        "baseline": baseline_result,
        "current": current_result,
        "comparison": comparison,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
