#!/usr/bin/env python3
"""
Create Training Sequences with Article Metadata
================================================

Creates training sequences with article indices for proper article-based splits.

Usage:
    python tools/create_training_sequences_with_articles.py \
        --npz artifacts/wikipedia_584k_fresh.npz \
        --output artifacts/lvm/training_sequences_ctx5_584k_with_articles.npz \
        --exclude-articles 1500-1999 \
        --context-len 5
"""

import argparse
import numpy as np
from pathlib import Path


def create_training_sequences(
    npz_path: str,
    output_path: str,
    exclude_article_ranges: list = None,
    context_len: int = 5
):
    """
    Create training sequences with article metadata for article-based splits.

    Args:
        npz_path: Source Wikipedia vectors NPZ
        output_path: Output training sequences NPZ
        exclude_article_ranges: List of (min, max) tuples to exclude (for OOD holdout)
        context_len: Context length (default 5)
    """

    print("=" * 80)
    print("CREATE TRAINING SEQUENCES WITH ARTICLE METADATA")
    print("=" * 80)
    print()

    # Load source data
    print(f"📥 Loading Wikipedia vectors from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    vectors = data['vectors']
    article_indices = data['article_indices']
    chunk_indices = data['chunk_indices']
    concept_texts = data['concept_texts']
    cpe_ids = data['cpe_ids']

    print(f"   Loaded {len(vectors):,} total chunks")
    print()

    # Apply exclusions
    if exclude_article_ranges:
        print(f"🚫 Excluding article ranges for OOD holdout:")
        mask = np.ones(len(vectors), dtype=bool)
        for min_art, max_art in exclude_article_ranges:
            exclude_mask = (article_indices >= min_art) & (article_indices <= max_art)
            n_excluded = exclude_mask.sum()
            print(f"   Articles {min_art}-{max_art}: {n_excluded:,} chunks")
            mask &= ~exclude_mask

        valid_indices = np.where(mask)[0]
        print(f"   Total remaining: {len(valid_indices):,} chunks")
        print()
    else:
        valid_indices = np.arange(len(vectors))
        print("   No exclusions - using all chunks")
        print()

    # Create sequences
    print(f"🔨 Creating sequences (context_len={context_len})...")
    print(f"   Method: stride=1 across all valid chunks")
    print()

    num_possible = len(valid_indices) - context_len

    context_sequences = []
    target_vectors = []
    metadata = []

    for i in range(num_possible):
        # Get context and target indices
        context_idxs = valid_indices[i:i+context_len]
        target_idx = valid_indices[i+context_len]

        # Build context sequence [context_len, 768]
        context_vecs = np.stack([vectors[idx] for idx in context_idxs])
        target_vec = vectors[target_idx]

        context_sequences.append(context_vecs)
        target_vectors.append(target_vec)

        # Store metadata
        meta = {
            'article_index': int(article_indices[target_idx]),
            'last_context_article': int(article_indices[context_idxs[-1]]),
            'target_chunk_index': int(chunk_indices[target_idx]),
            'target_cpe_id': str(cpe_ids[target_idx]),
            'crosses_article': int(article_indices[context_idxs[-1]]) != int(article_indices[target_idx])
        }
        metadata.append(meta)

        if (i + 1) % 50000 == 0:
            print(f"   Created {i+1:,} sequences...")

    context_sequences = np.array(context_sequences, dtype=np.float32)
    target_vectors = np.array(target_vectors, dtype=np.float32)
    context_texts_to_save = np.array([concept_texts[i:i+context_len] for i in range(num_possible)], dtype=object)
    target_texts_to_save = np.array([concept_texts[i+context_len] for i in range(num_possible)], dtype=object)
    metadata = np.array(metadata, dtype=object)

    print(f"   ✅ Created {len(context_sequences):,} training sequences")
    print()

    # Calculate coherence
    print("🔍 Verifying coherence...")
    coherences = []
    for i in range(min(4, context_len-1)):
        a = context_sequences[:1000, i, :]
        b = context_sequences[:1000, i+1, :]
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        sim = (a_norm * b_norm).sum(axis=1).mean()
        coherences.append(sim)
        print(f"   pos[{i}] vs pos[{i+1}]: {sim:.4f}")

    mean_coherence = np.mean(coherences)
    print(f"   Mean coherence: {mean_coherence:.4f}")

    if 0.45 <= mean_coherence <= 0.50:
        print(f"   ✅ GOOD: Coherence matches target distribution!")
    else:
        print(f"   ⚠️  WARNING: Coherence {mean_coherence:.4f} outside target range 0.45-0.50")
    print()

    # Save
    print(f"💾 Saving to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        context_sequences=context_sequences,
        target_vectors=target_vectors,
        context_texts=context_texts_to_save,
        target_texts=target_texts_to_save,
        metadata=metadata,
        context_length=context_len,
        num_sequences=len(context_sequences),
        vector_dim=768,
        source_dataset=str(npz_path),
        excluded_articles=str(exclude_article_ranges) if exclude_article_ranges else "none"
    )

    file_size = Path(output_path).stat().st_size / (1024**2)
    print(f"✅ Saved {len(context_sequences):,} sequences ({file_size:.1f} MB)")
    print()

    # Statistics
    unique_articles = len(set(m['article_index'] for m in metadata))
    cross_boundary = sum(1 for m in metadata if m['crosses_article'])

    print("📊 Statistics:")
    print(f"   Training sequences: {len(context_sequences):,}")
    print(f"   Unique articles: {unique_articles}")
    print(f"   Context length: {context_len}")
    print(f"   Cross-article sequences: {cross_boundary} ({100*cross_boundary/len(context_sequences):.1f}%)")
    print(f"   Mean coherence: {mean_coherence:.4f}")
    if exclude_article_ranges:
        print(f"   Excluded articles: {exclude_article_ranges}")
    print()

    print("=" * 80)
    print("✅ TRAINING SEQUENCES WITH ARTICLE METADATA COMPLETE!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Create training sequences with article metadata"
    )
    parser.add_argument("--npz", required=True, help="Input Wikipedia NPZ file")
    parser.add_argument("--output", required=True, help="Output training sequences NPZ")
    parser.add_argument("--exclude-articles",
                       help="Article ranges to exclude (e.g., '1500-1999,7000-7499')")
    parser.add_argument("--context-len", type=int, default=5,
                       help="Context length (default: 5)")

    args = parser.parse_args()

    # Parse exclusion ranges
    exclude_ranges = None
    if args.exclude_articles:
        exclude_ranges = []
        for range_str in args.exclude_articles.split(','):
            min_art, max_art = map(int, range_str.split('-'))
            exclude_ranges.append((min_art, max_art))

    create_training_sequences(
        args.npz,
        args.output,
        exclude_ranges,
        args.context_len
    )


if __name__ == "__main__":
    main()
