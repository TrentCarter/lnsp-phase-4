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
    Compute forward-distinctness scores for all samples.

    Args:
        npz_path: Path to training NPZ (with context_sequences, target_vectors)
        output_path: Where to save scores (default: {input}_forward_scores.npz)
    """
    print(f"📥 Loading {npz_path}...")
    data = np.load(npz_path)

    contexts = data['context_sequences']  # (N, 5, 768)
    targets = data['target_vectors']      # (N, 768)

    N = len(contexts)
    print(f"   Found {N:,} training samples")

    # For each sample:
    # - prev = context[-2] (second-to-last context position)
    # - next_proxy = target (we don't have actual next, so use target as proxy)
    # - Δ = cos(target, target) - cos(target, prev)
    #     = 1.0 - cos(target, prev)  [since target is normalized]

    # Actually, let's use a better definition:
    # - prev = context[-1] (last context, the copy-last candidate)
    # - Δ = 1.0 - cos(target, prev)
    # High Δ → target is far from prev (forward-distinct)
    # Low Δ → target is close to prev (copy-friendly)

    prev = contexts[:, -1, :]  # (N, 768) - last context position

    # Compute similarity
    sim_prev = cosine_similarity(targets, prev)

    # Forward-distinctness: distance from copy-last
    # Higher score = less similar to prev = more forward-distinct
    forward_distinctness = 1.0 - sim_prev

    print(f"\n📊 Forward-Distinctness Statistics:")
    print(f"   Mean: {forward_distinctness.mean():.4f}")
    print(f"   Std:  {forward_distinctness.std():.4f}")
    print(f"   Min:  {forward_distinctness.min():.4f}")
    print(f"   Max:  {forward_distinctness.max():.4f}")

    # Show percentiles
    p30 = np.percentile(forward_distinctness, 70)  # Top 30% threshold
    p70 = np.percentile(forward_distinctness, 30)  # Top 70% threshold
    print(f"\n📌 Curriculum Thresholds:")
    print(f"   Top 30% (Stage A): Δ ≥ {p30:.4f}")
    print(f"   Top 70% (Stage B): Δ ≥ {p70:.4f}")
    print(f"   Full (Stage C):    All samples")

    # Count samples per stage
    top30_count = np.sum(forward_distinctness >= p30)
    top70_count = np.sum(forward_distinctness >= p70)
    print(f"\n📦 Sample Counts:")
    print(f"   Stage A (top 30%): {top30_count:,} samples")
    print(f"   Stage B (top 70%): {top70_count:,} samples")
    print(f"   Stage C (full):    {N:,} samples")

    # Save scores
    if output_path is None:
        output_path = npz_path.parent / f"{npz_path.stem}_forward_scores.npz"

    np.savez_compressed(
        output_path,
        forward_distinctness=forward_distinctness,
        threshold_top30=p30,
        threshold_top70=p70,
        num_samples=N,
    )

    print(f"\n✅ Saved scores to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Compute forward-distinctness scores for curriculum")
    parser.add_argument("--npz", type=Path, required=True, help="Training NPZ file")
    parser.add_argument("--output", type=Path, help="Output scores NPZ (default: auto)")

    args = parser.parse_args()

    compute_forward_distinctness(args.npz, args.output)


if __name__ == "__main__":
    main()
