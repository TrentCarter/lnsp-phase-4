#!/usr/bin/env python3
"""
P6 Direction Diagnostics: Check if training data has forward vs backward bias

Tests:
1. Forward vs Backward correlation from last context vector
2. Offset sweep heatmap (ctx[i] vs target@k for k in [-3..+3])
3. Reverse article control (flip direction and check if advantage flips)

Usage:
    python tools/diagnose_p6_direction.py \
        --train-npz artifacts/lvm/training_sequences_ctx5_p6_next_token.npz \
        --wiki-npz artifacts/wikipedia_584k_fresh.npz \
        --n-samples 5000
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List


def cosine(a, b):
    """Cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def load_p6_data(npz_path: Path):
    """Load P6 training data"""
    print(f"\n📂 Loading P6 data from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    contexts = data['context_sequences']  # (N, 5, 768)
    targets = data['target_vectors']      # (N, 768) - actually target_next
    metadata = data['metadata']           # Object array of dicts

    print(f"   Loaded {len(contexts):,} sequences")
    return contexts, targets, metadata


def load_wikipedia_vectors(npz_path: Path):
    """Load Wikipedia article vectors"""
    print(f"📂 Loading Wikipedia vectors from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    vectors = data['vectors']
    article_indices = data['article_indices']
    chunk_indices = data['chunk_indices']

    # Build lookup: (article_idx, chunk_idx) → position
    lookup = {}
    for i in range(len(vectors)):
        lookup[(article_indices[i], chunk_indices[i])] = i

    print(f"   Loaded {len(vectors):,} vectors")
    return vectors, article_indices, chunk_indices, lookup


def test_forward_vs_backward(contexts, targets, metadata, wiki_vectors, wiki_lookup, n_samples):
    """
    Test 1: Forward vs Backward correlation from last context vector

    For each sequence:
        fwd = cos(ctx[-1], target_next)  # What P6 trains on
        bwd = cos(ctx[-1], target_prev)  # What we want to avoid

    If fwd < bwd, the data has backward bias!
    """
    print(f"\n{'='*60}")
    print("TEST 1: Forward vs Backward Correlation")
    print(f"{'='*60}")

    fwd_scores = []
    bwd_scores = []
    skipped = 0

    for i in range(min(n_samples, len(contexts))):
        if i % 1000 == 0:
            print(f"   Processing {i:,}/{n_samples:,}...")

        ctx_last = contexts[i, -1, :]  # Last context vector (ctx[4])
        target_next = targets[i]        # Target (already target_next in P6 data)

        # Compute forward correlation
        fwd = cosine(ctx_last, target_next)

        # Look up target_prev (target_next - 1) from article store
        meta = metadata[i]
        article_idx = meta['article_index']
        target_chunk_idx = meta['target_chunk_index']  # Points to target_next

        # Target_prev is target_chunk_idx - 1
        prev_key = (article_idx, target_chunk_idx - 1)
        if prev_key not in wiki_lookup:
            skipped += 1
            continue

        prev_pos = wiki_lookup[prev_key]
        target_prev = wiki_vectors[prev_pos]

        # Compute backward correlation
        bwd = cosine(ctx_last, target_prev)

        fwd_scores.append(fwd)
        bwd_scores.append(bwd)

    fwd_mean = np.mean(fwd_scores)
    bwd_mean = np.mean(bwd_scores)
    delta = fwd_mean - bwd_mean

    print(f"\n📊 Results:")
    print(f"   Forward (ctx[-1] vs target_next): {fwd_mean:.4f}")
    print(f"   Backward (ctx[-1] vs target_prev): {bwd_mean:.4f}")
    print(f"   Δ (fwd - bwd): {delta:+.4f}")
    print(f"   Samples: {len(fwd_scores):,}, Skipped: {skipped:,}")

    print(f"\n🔍 Interpretation:")
    if delta > 0.05:
        print(f"   ✅ GOOD: Data has forward bias (Δ = +{delta:.4f})")
        print(f"      Model should learn forward prediction naturally")
    elif delta > 0:
        print(f"   ⚠️  WEAK: Forward advantage is small (Δ = +{delta:.4f})")
        print(f"      Directional margin loss recommended")
    else:
        print(f"   ❌ BAD: Data has BACKWARD bias (Δ = {delta:.4f})")
        print(f"      Directional margin loss REQUIRED!")

    return fwd_scores, bwd_scores, delta


def test_offset_sweep(contexts, targets, metadata, wiki_vectors, wiki_lookup, n_samples):
    """
    Test 2: Offset sweep heatmap

    For each offset k in [-3, -2, -1, 0, +1, +2, +3]:
        Compute cos(ctx[-1], target@k)

    Expect monotone increase towards k=+1 if forward signal exists
    """
    print(f"\n{'='*60}")
    print("TEST 2: Offset Sweep Heatmap")
    print(f"{'='*60}")

    offsets = [-3, -2, -1, 0, 1, 2, 3]
    scores = {k: [] for k in offsets}
    skipped = 0

    for i in range(min(n_samples, len(contexts))):
        if i % 1000 == 0:
            print(f"   Processing {i:,}/{n_samples:,}...")

        ctx_last = contexts[i, -1, :]
        meta = metadata[i]
        article_idx = meta['article_index']
        target_chunk_idx = meta['target_chunk_index']  # Points to target_next in P6

        # For P6 data, k=0 should point to target_next (current target in P6 data)
        # k=-1 should point to target_current (the original target before P6 shift)
        # k=+1 should point to target_next+1

        ok = True
        for k in offsets:
            offset_chunk_idx = target_chunk_idx + k
            offset_key = (article_idx, offset_chunk_idx)
            if offset_key not in wiki_lookup:
                ok = False
                break

        if not ok:
            skipped += 1
            continue

        for k in offsets:
            offset_chunk_idx = target_chunk_idx + k
            offset_pos = wiki_lookup[(article_idx, offset_chunk_idx)]
            offset_vec = wiki_vectors[offset_pos]
            scores[k].append(cosine(ctx_last, offset_vec))

    means = {k: np.mean(v) for k, v in scores.items()}

    print(f"\n📊 Offset Heatmap (ctx[-1] vs target@k):")
    print(f"   {'Offset':<10} {'Mean Cos':<12} {'Bar'}")
    print(f"   {'-'*10} {'-'*12} {'-'*40}")
    for k in offsets:
        bar = '█' * int(means[k] * 50)
        marker = " ← P6 target" if k == 0 else ""
        print(f"   k={k:+2d}       {means[k]:.4f}       {bar}{marker}")

    print(f"   Samples: {len(scores[0]):,}, Skipped: {skipped:,}")

    # Check if monotone increasing towards k=0 (target_next in P6)
    is_increasing = all(means[offsets[i]] < means[offsets[i+1]]
                       for i in range(offsets.index(0)))

    print(f"\n🔍 Interpretation:")
    if is_increasing:
        print(f"   ✅ GOOD: Correlation increases monotonically towards k=0 (target_next)")
        print(f"      Data has forward temporal structure")
    else:
        print(f"   ⚠️  MIXED: Non-monotonic pattern detected")
        print(f"      Check for backward or ambiguous temporal signal")

    return means


def test_reverse_control(contexts, targets, metadata, n_samples):
    """
    Test 3: Reverse article control

    Flip the context sequences and check if forward advantage flips sign
    """
    print(f"\n{'='*60}")
    print("TEST 3: Reverse Article Control")
    print(f"{'='*60}")

    normal_fwd = []
    reverse_fwd = []

    for i in range(min(n_samples, len(contexts))):
        if i % 1000 == 0:
            print(f"   Processing {i:,}/{n_samples:,}...")

        ctx_normal = contexts[i]        # (5, 768)
        ctx_reversed = ctx_normal[::-1]  # Reverse temporal order
        target = targets[i]

        # Normal: forward prediction
        cos_normal = cosine(ctx_normal[-1], target)
        # Reversed: should look like backward prediction
        cos_reverse = cosine(ctx_reversed[-1], target)

        normal_fwd.append(cos_normal)
        reverse_fwd.append(cos_reverse)

    normal_mean = np.mean(normal_fwd)
    reverse_mean = np.mean(reverse_fwd)
    delta = normal_mean - reverse_mean

    print(f"\n📊 Results:")
    print(f"   Normal order: {normal_mean:.4f}")
    print(f"   Reversed order: {reverse_mean:.4f}")
    print(f"   Δ (normal - reverse): {delta:+.4f}")

    print(f"\n🔍 Interpretation:")
    if delta > 0.02:
        print(f"   ✅ GOOD: Reversing hurts (Δ = +{delta:.4f})")
        print(f"      Model can distinguish forward from backward")
    else:
        print(f"   ❌ BAD: Reversing has minimal effect (Δ = {delta:.4f})")
        print(f"      Data is temporally ambiguous or symmetric")

    return normal_mean, reverse_mean, delta


def main():
    parser = argparse.ArgumentParser(description="P6 direction diagnostics")
    parser.add_argument("--train-npz", type=Path, required=True, help="P6 training data")
    parser.add_argument("--wiki-npz", type=Path, required=True, help="Wikipedia vectors")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of samples to test")

    args = parser.parse_args()

    # Load data
    contexts, targets, metadata = load_p6_data(args.train_npz)
    wiki_vectors, wiki_art_idx, wiki_chunk_idx, wiki_lookup = load_wikipedia_vectors(args.wiki_npz)

    # Run tests
    fwd, bwd, delta1 = test_forward_vs_backward(
        contexts, targets, metadata, wiki_vectors, wiki_lookup, args.n_samples
    )
    means = test_offset_sweep(
        contexts, targets, metadata, wiki_vectors, wiki_lookup, args.n_samples
    )
    normal, reverse, delta3 = test_reverse_control(contexts, targets, metadata, args.n_samples)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Test 1 (Fwd vs Bwd): Δ = {delta1:+.4f}")
    print(f"Test 2 (Offset Sweep): Monotonic = {all(means[k1] < means[k2] for k1, k2 in zip([-3,-2,-1], [-2,-1,0]))}")
    print(f"Test 3 (Reverse Control): Δ = {delta3:+.4f}")

    print(f"\n🎯 Recommendation:")
    if delta1 > 0.05 and delta3 > 0.02:
        print(f"   ✅ Data has strong forward bias - pure MSE should work")
    elif delta1 > 0:
        print(f"   ⚠️  Data has weak forward bias - add directional margin loss")
    else:
        print(f"   ❌ Data has backward bias - MUST use directional margin loss!")


if __name__ == "__main__":
    main()
