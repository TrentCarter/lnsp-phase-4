#!/usr/bin/env python3
"""
Create simple training sequences from arXiv NPZ for Δ measurement.

Creates sequences within each article (no cross-article transitions).
Output format: contexts (N, 5, 768) + targets (N, 768)
"""

import argparse
import numpy as np
from pathlib import Path


def create_sequences(vectors, article_ids, context_size=5):
    """Create training sequences from vectors with article boundaries."""
    contexts = []
    targets = []

    # Group vectors by article
    unique_articles = []
    seen = set()
    for aid in article_ids:
        if aid not in seen:
            unique_articles.append(aid)
            seen.add(aid)

    for article_id in unique_articles:
        # Get all vectors for this article
        article_mask = article_ids == article_id
        article_vectors = vectors[article_mask]

        if len(article_vectors) < context_size + 1:
            continue  # Skip articles with too few chunks

        # Create sequences within this article
        for i in range(len(article_vectors) - context_size):
            context = article_vectors[i:i+context_size]  # (5, 768)
            target = article_vectors[i+context_size]      # (768,)

            contexts.append(context)
            targets.append(target)

    return np.array(contexts, dtype=np.float32), np.array(targets, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Create arXiv training sequences")
    parser.add_argument("--input", required=True, help="Input NPZ with vectors and article_ids")
    parser.add_argument("--output", required=True, help="Output NPZ with contexts and targets")
    parser.add_argument("--context-size", type=int, default=5, help="Context window size")

    args = parser.parse_args()

    # Load data
    print(f"Loading {args.input}...")
    data = np.load(args.input, allow_pickle=True)

    vectors = data['vectors']
    article_ids = data['article_ids']

    print(f"  Vectors: {vectors.shape}")
    print(f"  Articles: {len(set(article_ids))}")

    # Create sequences
    print(f"\nCreating sequences (context_size={args.context_size})...")
    contexts, targets = create_sequences(vectors, article_ids, args.context_size)

    print(f"  Contexts: {contexts.shape}")
    print(f"  Targets: {targets.shape}")

    if len(contexts) == 0:
        print("ERROR: No sequences created!")
        return

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        contexts=contexts,
        targets=targets
    )

    print(f"\nSaved to {args.output}")
    print(f"\n✓ Ready for diagnosis!")
    print(f"\nNext step:")
    print(f"  python tools/tests/diagnose_data_direction.py \\")
    print(f"    {args.output} --n-samples {len(contexts)}")


if __name__ == "__main__":
    main()
