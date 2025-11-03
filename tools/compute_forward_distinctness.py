#!/usr/bin/env python3
"""
Compute forward-distinctness scores for curriculum learning.

For each training sample, compute:
  Δ = similarity(target, next_in_article) - similarity(target, prev_in_article)

High Δ → target is more similar to next than to prev (forward-distinct, good for Stage A)
Low/negative Δ → target is ambiguous or copy-friendly (defer to Stage C)

Output: NPZ with 'forward_distinctness' array (same length as training sequences)
"""

import argparse
import numpy as np
from pathlib import Path


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors or batches."""
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    # Normalize
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)

    # Dot product
    return np.sum(a_norm * b_norm, axis=-1)


def compute_forward_distinctness(npz_path: Path, output_path: Path = None):
    """
    Compute forward-advantage scores for curriculum learning.

    Computes richer metrics beyond simple similarity:
    - sim_prev: cos(target, ctx[-1]) - similarity to last context
    - sim_prev2: cos(target, ctx[-2]) - similarity to second-to-last
    - sim_other_max: max(cos(target, ctx[i])) for i in [0..3] - best match in earlier context
    - adv_prev: sim_prev - sim_other_max - FORWARD ADVANTAGE (key signal!)
    - delta_prev2: sim_prev - sim_prev2 - tie-breaker

    Forward advantage encodes both direction (target similar to ctx[-1]) AND uniqueness
    (ctx[-1] is the BEST match, not just one of many similar slots).

    Args:
        npz_path: Path to training NPZ (with context_sequences, target_vectors)
        output_path: Where to save scores (default: {input}_forward_scores.npz)
    """
    print(f"📥 Loading {npz_path}...")
    data = np.load(npz_path)

    contexts = data['context_sequences']  # (N, 5, 768)
    targets = data['target_vectors']      # (N, 768)

    N = len(contexts)
    ctx_len = contexts.shape[1]  # Should be 5
    print(f"   Found {N:,} training samples (context length: {ctx_len})")

    print(f"\n🧮 Computing forward-advantage metrics...")

    # 1. Similarity to last context position (ctx[-1])
    sim_prev = cosine_similarity(targets, contexts[:, -1, :])

    # 2. Similarity to second-to-last position (ctx[-2])
    sim_prev2 = cosine_similarity(targets, contexts[:, -2, :])

    # 3. Maximum similarity to any earlier context position (ctx[0..3])
    sim_others = []
    for i in range(ctx_len - 1):  # Positions 0, 1, 2, 3 (exclude last)
        sim_i = cosine_similarity(targets, contexts[:, i, :])
        sim_others.append(sim_i)

    sim_others = np.stack(sim_others, axis=0)  # (4, N)
    sim_other_max = np.max(sim_others, axis=0)  # (N,)

    # 4. Forward advantage: is ctx[-1] the BEST match?
    # High adv_prev → ctx[-1] is uniquely good match (direction + uniqueness)
    # Low/negative adv_prev → other positions match just as well (ambiguous)
    adv_prev = sim_prev - sim_other_max

    # 5. Delta from prev2: tie-breaker for ranking
    delta_prev2 = sim_prev - sim_prev2

    print(f"\n📊 Forward-Advantage Statistics:")
    print(f"   sim_prev (last):         mean={sim_prev.mean():.4f}, std={sim_prev.std():.4f}")
    print(f"   sim_prev2 (2nd-last):    mean={sim_prev2.mean():.4f}, std={sim_prev2.std():.4f}")
    print(f"   sim_other_max (best 0-3): mean={sim_other_max.mean():.4f}, std={sim_other_max.std():.4f}")
    print(f"   adv_prev (ADVANTAGE):    mean={adv_prev.mean():.4f}, std={adv_prev.std():.4f}")
    print(f"   delta_prev2 (tie-break): mean={delta_prev2.mean():.4f}, std={delta_prev2.std():.4f}")

    # Sanity check: % samples with positive advantage
    pct_positive_adv = 100 * (adv_prev > 0).mean()
    print(f"\n🔍 Sanity Check:")
    print(f"   % samples with adv_prev > 0: {pct_positive_adv:.1f}%")
    if pct_positive_adv < 50:
        print(f"   ⚠️  WARNING: Less than 50% have positive advantage!")
        print(f"       This suggests ctx[-1] is NOT the best match for most samples")

    # Show percentiles for advantage metric
    p30_adv = np.percentile(adv_prev, 70)  # Top 30% by advantage
    p70_adv = np.percentile(adv_prev, 30)  # Top 70% by advantage
    print(f"\n📌 Suggested Curriculum Thresholds (by advantage):")
    print(f"   Top 30% (Stage A): adv_prev ≥ {p30_adv:.4f}")
    print(f"   Top 70% (Stage B): adv_prev ≥ {p70_adv:.4f}")

    # Also show absolute thresholds (consultant's recommendation)
    print(f"\n📌 Recommended Absolute Thresholds (consultant):")
    print(f"   Stage A: sim_prev ≥ 0.66 AND adv_prev ≥ 0.08")
    print(f"   Stage B: sim_prev ≥ 0.58 OR adv_prev ≥ 0.05")

    # Estimate counts for recommended thresholds
    mask_A_recommended = (sim_prev >= 0.66) & (adv_prev >= 0.08)
    mask_B_recommended = (sim_prev >= 0.58) | (adv_prev >= 0.05)
    print(f"\n📦 Sample Counts (recommended thresholds):")
    print(f"   Stage A: {mask_A_recommended.sum():,} samples ({100*mask_A_recommended.mean():.1f}%)")
    print(f"   Stage B: {mask_B_recommended.sum():,} samples ({100*mask_B_recommended.mean():.1f}%)")
    print(f"   Stage C (full): {N:,} samples (100.0%)")

    # Save richer scores
    if output_path is None:
        output_path = npz_path.parent / f"{npz_path.stem}_forward_scores.npz"

    np.savez_compressed(
        output_path,
        # Rich metrics (P5.1)
        sim_prev=sim_prev,
        sim_prev2=sim_prev2,
        sim_other_max=sim_other_max,
        adv_prev=adv_prev,
        delta_prev2=delta_prev2,
        # Legacy metric (for backward compatibility)
        forward_distinctness=sim_prev,
        # Recommended thresholds
        tau_sim_A=0.66,
        tau_adv_A=0.08,
        tau_sim_B=0.58,
        tau_adv_B=0.05,
        # Metadata
        num_samples=N,
    )

    print(f"\n✅ Saved scores to: {output_path}")
    print(f"   Includes: sim_prev, sim_prev2, sim_other_max, adv_prev, delta_prev2")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Compute forward-distinctness scores for curriculum")
    parser.add_argument("--npz", type=Path, required=True, help="Training NPZ file")
    parser.add_argument("--output", type=Path, help="Output scores NPZ (default: auto)")

    args = parser.parse_args()

    compute_forward_distinctness(args.npz, args.output)


if __name__ == "__main__":
    main()
