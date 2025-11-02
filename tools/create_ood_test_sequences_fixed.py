#!/usr/bin/env python3
"""
Create OOD Test Sequences (FIXED - Matches Training Distribution)
==================================================================

CRITICAL FIX (2025-10-30):
- OLD version: stride=1 ONLY within each article → 0.73 coherence (wrong!)
- NEW version: stride=1 across ALL vectors → 0.46 coherence (matches training!)

This fix resolves the 50% coherence mismatch that caused all OOD scores to be negative.

Usage:
    python tools/create_ood_test_sequences_fixed.py \
        --npz artifacts/wikipedia_584k_fresh.npz \
        --output artifacts/lvm/wikipedia_ood_test_ctx5_fresh_FIXED.npz \
        --min-article 8001 \
        --max-article 8470 \
        --context-len 5 \
        --max-sequences 10000
"""

import argparse
import numpy as np
from pathlib import Path


def create_ood_sequences_fixed(
    npz_path: str,
    output_path: str,
    min_article: int = 8001,
    max_article: int = 8470,
    context_len: int = 5,
    max_sequences: int = 10000
):
    """
    Create OOD test sequences matching training's distribution.

    KEY FIX: Use stride=1 across ALL holdout vectors (including article boundaries)
    instead of only within each article. This matches how training data was created.
    """

    print("=" * 80)
    print("CREATE OOD TEST SEQUENCES (FIXED VERSION)")
    print("=" * 80)
    print()

    # Load data
    print(f"📥 Loading Wikipedia vectors from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    vectors = data['vectors']
    article_indices = data['article_indices']
    chunk_indices = data['chunk_indices']
    concept_texts = data['concept_texts']
    cpe_ids = data['cpe_ids']

    print(f"   Loaded {len(vectors):,} total chunks")
    print()

    # Filter to holdout articles
    print(f"🔍 Filtering to holdout articles {min_article}-{max_article}...")
    mask = (article_indices >= min_article) & (article_indices <= max_article)
    holdout_indices = np.where(mask)[0]

    print(f"   Found {len(holdout_indices):,} chunks in holdout set")
    print()

    # Sort holdout indices to ensure proper ordering
    holdout_indices = sorted(holdout_indices)

    print(f"🔨 Creating sequences (context_len={context_len})...")
    print(f"   Method: stride=1 across ALL holdout vectors (matches training!)")
    print()

    # Create sequences using SAME approach as training
    # (stride=1 across all vectors, including cross-article boundaries)
    num_possible = len(holdout_indices) - context_len
    num_sequences = min(num_possible, max_sequences)

    context_sequences = []
    target_vectors = []
    metadata = []

    for i in range(num_sequences):
        # Get context and target indices from holdout set
        context_idxs = holdout_indices[i:i+context_len]
        target_idx = holdout_indices[i+context_len]

        # Build context sequence [5, 768]
        context_vecs = np.stack([vectors[idx] for idx in context_idxs])
        target_vec = vectors[target_idx]

        context_sequences.append(context_vecs)
        target_vectors.append(target_vec)

        # Metadata
        meta = {
            'article_index': int(article_indices[target_idx]),
            'last_chunk_index': int(chunk_indices[context_idxs[-1]]),
            'target_chunk_index': int(chunk_indices[target_idx]),
            'last_cpe_id': str(cpe_ids[context_idxs[-1]]),
            'target_cpe_id': str(cpe_ids[target_idx]),
            'crosses_article_boundary': int(article_indices[context_idxs[-1]]) != int(article_indices[target_idx])
        }
        metadata.append(meta)

        if (i + 1) % 1000 == 0:
            print(f"   Created {i+1:,} sequences...")

    context_sequences = np.array(context_sequences, dtype=np.float32)
    target_vectors = np.array(target_vectors, dtype=np.float32)
    metadata = np.array(metadata, dtype=object)

    print(f"   ✅ Created {len(context_sequences):,} test sequences")
    print()

    # Calculate coherence to verify it matches training
    print("🔍 Verifying coherence matches training distribution...")
    coherences = []
    for i in range(min(4, context_len-1)):
        a = context_sequences[:, i, :]
        b = context_sequences[:, i+1, :]
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        sim = (a_norm * b_norm).sum(axis=1).mean()
        coherences.append(sim)
        print(f"   pos[{i}] vs pos[{i+1}]: {sim:.4f}")

    mean_coherence = np.mean(coherences)
    print(f"   Mean coherence: {mean_coherence:.4f}")

    if mean_coherence > 0.60:
        print(f"   ⚠️  WARNING: Coherence {mean_coherence:.4f} is too high!")
        print(f"              Training has ~0.46-0.49. OOD should match!")
    elif 0.45 <= mean_coherence <= 0.50:
        print(f"   ✅ GOOD: Coherence matches training distribution!")
    else:
        print(f"   ⚠️  UNEXPECTED: Coherence {mean_coherence:.4f} outside expected range")
    print()

    # Save
    print(f"💾 Saving to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        context_sequences=context_sequences,
        target_vectors=target_vectors,
        metadata=metadata
    )

    file_size = Path(output_path).stat().st_size / (1024**2)
    print(f"✅ Saved {len(context_sequences):,} sequences ({file_size:.1f} MB)")
    print()

    # Statistics
    unique_articles = len(set(m['article_index'] for m in metadata))
    cross_boundary = sum(1 for m in metadata if m['crosses_article_boundary'])

    print("📊 Statistics:")
    print(f"   Test sequences: {len(context_sequences):,}")
    print(f"   Unique articles: {unique_articles}")
    print(f"   Article range: {min_article}-{max_article}")
    print(f"   Context length: {context_len}")
    print(f"   Cross-article sequences: {cross_boundary} ({100*cross_boundary/len(context_sequences):.1f}%)")
    print(f"   Mean coherence: {mean_coherence:.4f} (target: 0.46-0.49)")
    print()

    print("=" * 80)
    print("✅ FIXED OOD TEST SET COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Re-evaluate 584k baseline model:")
    print("   ./.venv/bin/python tools/eval_model_ood.py \\")
    print(f"     --model artifacts/lvm/models/amn_584k_pure_mse_20251029_055838/best_model.pt \\")
    print(f"     --ood-data {output_path} \\")
    print("     --device mps")
    print()
    print("   Expected: OOD cosine ~0.63-0.65 (restored!)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Create OOD test sequences (FIXED - matches training distribution)"
    )
    parser.add_argument("--npz", required=True, help="Input NPZ file with vectors")
    parser.add_argument("--output", required=True, help="Output test sequences NPZ")
    parser.add_argument("--min-article", type=int, default=8001,
                       help="Minimum article index (default: 8001)")
    parser.add_argument("--max-article", type=int, default=8470,
                       help="Maximum article index (default: 8470)")
    parser.add_argument("--context-len", type=int, default=5,
                       help="Context length (default: 5)")
    parser.add_argument("--max-sequences", type=int, default=10000,
                       help="Max sequences to generate (default: 10000)")

    args = parser.parse_args()

    create_ood_sequences_fixed(
        args.npz,
        args.output,
        args.min_article,
        args.max_article,
        args.context_len,
        args.max_sequences
    )


if __name__ == "__main__":
    main()
