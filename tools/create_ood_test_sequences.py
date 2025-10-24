#!/usr/bin/env python3
"""
Create OOD Test Sequences from Wikipedia NPZ
=============================================

Creates out-of-distribution test sequences for LVM evaluation.
Uses articles 8001-8470 as holdout set (not seen during training).

Usage:
    python tools/create_ood_test_sequences.py \
        --npz artifacts/wikipedia_584k_fresh.npz \
        --output artifacts/lvm/wikipedia_ood_test_ctx5_fresh.npz \
        --min-article 8001 \
        --max-article 8470 \
        --context-len 5 \
        --max-sequences 10000
"""

import argparse
import numpy as np
from pathlib import Path


def create_ood_sequences(
    npz_path: str,
    output_path: str,
    min_article: int = 8001,
    max_article: int = 8470,
    context_len: int = 5,
    max_sequences: int = 10000
):
    """Create OOD test sequences from holdout articles."""

    print("=" * 80)
    print("CREATE OOD TEST SEQUENCES")
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

    # Group by article
    print("📚 Grouping chunks by article...")
    articles = {}
    for idx in holdout_indices:
        article_idx = int(article_indices[idx])
        if article_idx not in articles:
            articles[article_idx] = []
        articles[article_idx].append(idx)

    # Sort chunks within each article
    for article_idx in articles:
        articles[article_idx] = sorted(articles[article_idx],
                                      key=lambda i: chunk_indices[i])

    print(f"   Found {len(articles)} unique articles in holdout")
    print()

    # Create sequences
    print(f"🔨 Creating sequences (context_len={context_len})...")
    context_sequences = []
    target_vectors = []
    metadata = []

    total_sequences = 0
    for article_idx, chunk_idxs in articles.items():
        if len(chunk_idxs) < context_len + 1:
            continue  # Need at least context + target

        # Create sliding window sequences
        for i in range(len(chunk_idxs) - context_len):
            context_idxs = chunk_idxs[i:i+context_len]
            target_idx = chunk_idxs[i+context_len]

            # Build context sequence [5, 768]
            context_vecs = np.stack([vectors[idx] for idx in context_idxs])
            target_vec = vectors[target_idx]

            context_sequences.append(context_vecs)
            target_vectors.append(target_vec)

            # Metadata for v2 evaluation
            meta = {
                'article_index': article_idx,
                'last_chunk_index': int(chunk_indices[context_idxs[-1]]),
                'target_chunk_index': int(chunk_indices[target_idx]),
                'last_cpe_id': str(cpe_ids[context_idxs[-1]]),
                'target_cpe_id': str(cpe_ids[target_idx])
            }
            metadata.append(meta)

            total_sequences += 1
            if total_sequences >= max_sequences:
                break

        if total_sequences >= max_sequences:
            break

        if len(context_sequences) % 1000 == 0:
            print(f"   Created {len(context_sequences):,} sequences...")

    context_sequences = np.array(context_sequences, dtype=np.float32)
    target_vectors = np.array(target_vectors, dtype=np.float32)
    metadata = np.array(metadata, dtype=object)

    print(f"   ✅ Created {len(context_sequences):,} test sequences")
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
    print("📊 Statistics:")
    print(f"   Test sequences: {len(context_sequences):,}")
    print(f"   Unique articles: {unique_articles}")
    print(f"   Article range: {min_article}-{max_article}")
    print(f"   Context length: {context_len}")
    print(f"   Avg sequences/article: {len(context_sequences)/unique_articles:.1f}")
    print()

    print("=" * 80)
    print("✅ OOD TEST SET COMPLETE!")
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description="Create OOD test sequences from Wikipedia NPZ")
    parser.add_argument("--npz", required=True, help="Input NPZ file with vectors")
    parser.add_argument("--output", required=True, help="Output test sequences NPZ")
    parser.add_argument("--min-article", type=int, default=8001, help="Minimum article index (default: 8001)")
    parser.add_argument("--max-article", type=int, default=8470, help="Maximum article index (default: 8470)")
    parser.add_argument("--context-len", type=int, default=5, help="Context length (default: 5)")
    parser.add_argument("--max-sequences", type=int, default=10000, help="Max sequences to generate (default: 10000)")

    args = parser.parse_args()

    create_ood_sequences(
        args.npz,
        args.output,
        args.min_article,
        args.max_article,
        args.context_len,
        args.max_sequences
    )


if __name__ == "__main__":
    main()
