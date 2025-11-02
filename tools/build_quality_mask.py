#!/usr/bin/env python3
"""
P1 Tool: Build High-Quality Mask
=================================

Filters out low-quality sequences based on:
1. Adjacency coherence threshold (cos(t[i], t[i+1]) >= min_coherence)
2. Text-based filters (regex blocklist for "List of", "Disambiguation", etc.)
3. Outlier detection (sequences far from local neighborhood)

Usage:
    ./.venv/bin/python tools/build_quality_mask.py \
      --input artifacts/lvm/training_sequences_ctx5.npz \
      --min-coherence 0.35 \
      --blocklist "List of|Disambiguation|Category:" \
      --out artifacts/lvm/quality_mask_790k.npy
"""

import sys
import re
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

def compute_adjacency_scores(vectors):
    """
    Compute adjacency coherence score for each sequence position.

    For position i, score = cos(t[i-1], t[i]) * cos(t[i], t[i+1])
    (geometric mean of left and right neighbors)

    Args:
        vectors: (N, 768) array of target vectors

    Returns:
        scores: (N,) array of adjacency scores
    """
    N = len(vectors)
    scores = np.ones(N, dtype=np.float32)

    # Compute cos(t[i], t[i+1]) for all pairs
    forward_cos = np.sum(vectors[:-1] * vectors[1:], axis=1)

    # Assign scores based on forward and backward neighbors
    # Forward neighbor score
    scores[:-1] = forward_cos

    # Backward neighbor score (combine with forward using geometric mean)
    for i in range(1, N):
        backward_cos = forward_cos[i-1]
        if i < N - 1:
            # Has both neighbors - geometric mean
            scores[i] = np.sqrt(backward_cos * forward_cos[i])
        else:
            # Only has backward neighbor
            scores[i] = backward_cos

    # First position only has forward neighbor
    scores[0] = forward_cos[0] if N > 1 else 1.0

    return scores


def apply_text_blocklist(target_texts, blocklist_pattern):
    """
    Create mask for sequences that don't match blocklist patterns.

    Args:
        target_texts: (N,) array of text strings
        blocklist_pattern: regex pattern (e.g., "List of|Disambiguation")

    Returns:
        mask: (N,) boolean array (True = keep, False = filter)
    """
    if not blocklist_pattern:
        return np.ones(len(target_texts), dtype=bool)

    pattern = re.compile(blocklist_pattern, re.IGNORECASE)
    mask = np.array([not pattern.search(text) for text in target_texts])

    return mask


def main():
    parser = argparse.ArgumentParser(description="Build quality filter mask")
    parser.add_argument("--input", required=True, help="Input NPZ file")
    parser.add_argument("--min-coherence", type=float, default=0.35,
                       help="Minimum adjacency coherence (default: 0.35)")
    parser.add_argument("--blocklist", type=str, default="",
                       help="Regex pattern for text-based filtering")
    parser.add_argument("--out", required=True, help="Output mask file (.npy)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.out)
    min_coherence = args.min_coherence
    blocklist_pattern = args.blocklist

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1

    print(f"{'='*60}")
    print(f"P1: BUILD QUALITY MASK")
    print(f"{'='*60}")
    print(f"Input:         {input_path}")
    print(f"Min coherence: {min_coherence}")
    print(f"Blocklist:     {blocklist_pattern if blocklist_pattern else '(none)'}")
    print(f"Output:        {output_path}")

    # Load data
    print(f"\n📂 Loading data...")
    data = np.load(input_path, allow_pickle=True)
    target_vectors = data['target_vectors']
    target_texts = data['target_texts']

    N = len(target_vectors)
    print(f"   Total sequences: {N:,}")

    # Filter 1: Adjacency coherence
    print(f"\n🔍 Filter 1: Adjacency Coherence (>={min_coherence})")
    adjacency_scores = compute_adjacency_scores(target_vectors)

    coherence_mask = adjacency_scores >= min_coherence
    coherence_kept = coherence_mask.sum()
    coherence_filtered = N - coherence_kept

    print(f"   Kept:     {coherence_kept:7,} ({100*coherence_kept/N:5.1f}%)")
    print(f"   Filtered: {coherence_filtered:7,} ({100*coherence_filtered/N:5.1f}%)")

    # Show score distribution
    print(f"\n   Adjacency score distribution:")
    bins = np.arange(0.0, 1.01, 0.1)
    hist, _ = np.histogram(adjacency_scores, bins=bins)
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        count = hist[i]
        pct = 100 * count / N
        bar = "█" * int(pct / 2)
        marker = "│" if high == min_coherence else ""
        print(f"   [{low:.1f}, {high:.1f}): {count:7d} ({pct:5.1f}%) {bar}{marker}")

    # Filter 2: Text blocklist
    if blocklist_pattern:
        print(f"\n🔍 Filter 2: Text Blocklist")
        text_mask = apply_text_blocklist(target_texts, blocklist_pattern)

        text_kept = text_mask.sum()
        text_filtered = N - text_kept

        print(f"   Kept:     {text_kept:7,} ({100*text_kept/N:5.1f}%)")
        print(f"   Filtered: {text_filtered:7,} ({100*text_filtered/N:5.1f}%)")

        # Show some filtered examples
        filtered_idx = np.where(~text_mask)[0]
        if len(filtered_idx) > 0:
            print(f"\n   Sample filtered texts:")
            for idx in filtered_idx[:5]:
                text = target_texts[idx][:80]
                print(f"      [{idx}] {text}...")
    else:
        text_mask = np.ones(N, dtype=bool)
        print(f"\n🔍 Filter 2: Text Blocklist (skipped)")

    # Combine masks
    print(f"\n{'='*60}")
    print(f"COMBINED FILTER RESULTS")
    print(f"{'='*60}")

    final_mask = coherence_mask & text_mask
    final_kept = final_mask.sum()
    final_filtered = N - final_kept

    print(f"Original:  {N:7,} sequences")
    print(f"Kept:      {final_kept:7,} ({100*final_kept/N:5.1f}%)")
    print(f"Filtered:  {final_filtered:7,} ({100*final_filtered/N:5.1f}%)")

    # Quality metrics for kept sequences
    print(f"\nQuality metrics for KEPT sequences:")
    kept_scores = adjacency_scores[final_mask]
    print(f"   Mean coherence: {kept_scores.mean():.4f}")
    print(f"   Std coherence:  {kept_scores.std():.4f}")
    print(f"   p10:            {np.percentile(kept_scores, 10):.4f}")
    print(f"   p50:            {np.percentile(kept_scores, 50):.4f}")
    print(f"   p90:            {np.percentile(kept_scores, 90):.4f}")

    # Compare with original
    print(f"\nOriginal vs Filtered:")
    print(f"   Mean coherence:  {adjacency_scores.mean():.4f} → {kept_scores.mean():.4f} ({kept_scores.mean() - adjacency_scores.mean():+.4f})")

    # Save mask
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, final_mask)

    print(f"\n💾 Mask saved to: {output_path}")
    print(f"{'='*60}")

    # Recommendation
    improvement = kept_scores.mean() - adjacency_scores.mean()
    if improvement >= 0.10:
        print(f"✅ SIGNIFICANT IMPROVEMENT (+{improvement:.4f})")
        print(f"   Recommendation: Use filtered dataset for training")
    elif improvement >= 0.05:
        print(f"⚠️  MODERATE IMPROVEMENT (+{improvement:.4f})")
        print(f"   Recommendation: Worth trying, but consider ctx=7 as backup")
    else:
        print(f"❌ MINIMAL IMPROVEMENT (+{improvement:.4f})")
        print(f"   Recommendation: Filtering alone won't fix the issue")
        print(f"   Try: ctx=7, curriculum learning, or stick with 584k")

    return 0


if __name__ == '__main__':
    sys.exit(main())
