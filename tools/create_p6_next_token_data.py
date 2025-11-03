#!/usr/bin/env python3
"""
P6: Create NEXT-token training data

Instead of predicting target (ctx[0..4] → v5), predict target_next (ctx[0..4] → v6).
This removes the identity path - model cannot copy ctx[4] to predict v6!

Input: Standard training sequences (contexts, targets, metadata)
Output: P6 training sequences (contexts, target_next, metadata_next)
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Any


def load_wikipedia_vectors(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Wikipedia article vectors and chunk mappings"""
    print(f"📂 Loading Wikipedia vectors from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    vectors = data['vectors']  # (N, 768)
    article_indices = data['article_indices']  # (N,)
    chunk_indices = data['chunk_indices']  # (N,)
    cpe_ids = data['cpe_ids']  # (N,)

    print(f"   Loaded {len(vectors):,} vectors")
    return vectors, article_indices, chunk_indices, cpe_ids


def build_chunk_lookup(article_indices: np.ndarray, chunk_indices: np.ndarray) -> Dict[tuple, int]:
    """Build lookup: (article_idx, chunk_idx) → vector_position"""
    lookup = {}
    for i in range(len(article_indices)):
        lookup[(article_indices[i], chunk_indices[i])] = i
    return lookup


def create_p6_data(
    train_npz: Path,
    wiki_npz: Path,
    output_path: Path
) -> None:
    """
    Convert standard training data to P6 next-token data

    Args:
        train_npz: Original training sequences (contexts → targets)
        wiki_npz: Full Wikipedia vectors (to look up target_next)
        output_path: Output path for P6 data (contexts → target_next)
    """

    # Load training sequences
    print(f"\n📂 Loading training sequences from {train_npz}...")
    train_data = np.load(train_npz, allow_pickle=True)

    # Handle different key names (context_sequences vs contexts, target_vectors vs targets)
    if 'context_sequences' in train_data:
        contexts = train_data['context_sequences']
        targets = train_data['target_vectors']
        metadata = train_data['metadata']
    elif 'contexts' in train_data:
        contexts = train_data['contexts']
        targets = train_data['targets']
        # Build metadata from article_ids and target_indices if available
        article_ids = train_data.get('article_ids', None)
        target_indices = train_data.get('target_indices', None)
        if article_ids is not None and target_indices is not None:
            metadata = [
                {
                    'article_index': int(art),
                    'target_chunk_index': int(idx),
                    'target_cpe_id': 'unknown',
                    'crosses_article': False
                }
                for art, idx in zip(article_ids, target_indices)
            ]
        else:
            raise ValueError("Cannot build metadata from validation file")
    else:
        raise KeyError(f"Unknown data format in {train_npz}. Keys: {list(train_data.keys())}")

    N = len(contexts)
    print(f"   Loaded {N:,} training sequences")

    # Load Wikipedia vectors for lookup
    wiki_vectors, wiki_article_indices, wiki_chunk_indices, wiki_cpe_ids = load_wikipedia_vectors(wiki_npz)
    chunk_lookup = build_chunk_lookup(wiki_article_indices, wiki_chunk_indices)
    print(f"   Built lookup for {len(chunk_lookup):,} chunks")

    # Create P6 sequences
    print(f"\n🔨 Creating P6 next-token sequences...")
    p6_contexts = []
    p6_target_next = []
    p6_metadata = []

    skipped_end_of_article = 0
    skipped_not_found = 0

    for i, meta in enumerate(metadata):
        if i % 50000 == 0:
            print(f"   Processing {i:,}/{N:,}...")

        article_idx = meta['article_index']
        target_chunk_idx = meta['target_chunk_index']

        # Look up target_next (the chunk AFTER target)
        next_chunk_idx = target_chunk_idx + 1
        next_key = (article_idx, next_chunk_idx)

        if next_key not in chunk_lookup:
            # End of article or missing chunk
            skipped_end_of_article += 1
            continue

        # Get target_next vector
        next_pos = chunk_lookup[next_key]
        target_next_vec = wiki_vectors[next_pos]

        # Keep this sequence
        p6_contexts.append(contexts[i])
        p6_target_next.append(target_next_vec)

        # Update metadata
        next_meta = meta.copy()
        next_meta['target_chunk_index'] = next_chunk_idx
        next_meta['target_cpe_id'] = str(wiki_cpe_ids[next_pos])
        next_meta['p6_offset'] = 1  # Predicting +1 offset
        p6_metadata.append(next_meta)

    # Convert to arrays
    p6_contexts = np.array(p6_contexts)
    p6_target_next = np.array(p6_target_next)

    print(f"\n✅ P6 data created:")
    print(f"   Original sequences: {N:,}")
    print(f"   P6 sequences: {len(p6_contexts):,}")
    print(f"   Skipped (end of article): {skipped_end_of_article:,}")
    print(f"   Skipped (not found): {skipped_not_found:,}")
    print(f"   Retention: {100*len(p6_contexts)/N:.1f}%")

    # Verify no identity path
    print(f"\n🔍 Verifying no identity path...")
    last_ctx = p6_contexts[:, -1, :]  # Last context position (ctx[4])
    cos_last_to_next = np.sum(last_ctx * p6_target_next, axis=1) / (
        np.linalg.norm(last_ctx, axis=1) * np.linalg.norm(p6_target_next, axis=1) + 1e-8
    )
    print(f"   Mean cos(ctx[4], target_next): {cos_last_to_next.mean():.3f}")
    print(f"   (Should be < 0.6, much lower than ~0.8 for target)")

    # Save P6 data
    print(f"\n💾 Saving P6 data to {output_path}...")
    np.savez_compressed(
        output_path,
        context_sequences=p6_contexts,
        target_vectors=p6_target_next,  # Actually target_next, but keep name for compatibility
        metadata=np.array(p6_metadata, dtype=object),
        context_length=5,
        num_sequences=len(p6_contexts),
        vector_dim=768,
        source_dataset='wikipedia_p6_next_token',
        p6_mode=True,
        p6_offset=1
    )
    print(f"✅ Saved {len(p6_contexts):,} P6 sequences")


def main():
    parser = argparse.ArgumentParser(description="Create P6 next-token training data")
    parser.add_argument("--train-npz", type=Path, required=True,
                        help="Original training sequences")
    parser.add_argument("--wiki-npz", type=Path, required=True,
                        help="Full Wikipedia vectors for lookup")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for P6 data")

    args = parser.parse_args()

    create_p6_data(args.train_npz, args.wiki_npz, args.output)


if __name__ == "__main__":
    main()
