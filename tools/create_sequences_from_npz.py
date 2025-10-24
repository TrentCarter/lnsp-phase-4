#!/usr/bin/env python3
"""
Create LVM Training Sequences from Wikipedia NPZ Vectors

This script creates training sequences from the fresh Wikipedia ingestion data.
It uses the NPZ file with 771k vectors and PostgreSQL metadata to maintain
article ordering.

Usage:
    python tools/create_sequences_from_npz.py \
        --npz artifacts/wikipedia_500k_corrected_vectors.npz \
        --output artifacts/lvm/wikipedia_fresh_sequences_ctx5.npz \
        --context-size 5 \
        --min-article-chunks 7

Output Format (compatible with train_unified.py):
    - context_sequences: [N, 5, 768] - Context vectors
    - target_vectors: [N, 768] - Target vectors
    - val_context_sequences: [N_val, 5, 768] - Validation contexts
    - val_target_vectors: [N_val, 768] - Validation targets
"""

import argparse
import numpy as np
import psycopg2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def load_article_ordering_from_db(dataset_source: str = "wikipedia_500k"):
    """
    Load article ordering from PostgreSQL.
    Returns dict: {cpe_id: (article_index, chunk_index)}
    """
    print("📊 Loading article ordering from PostgreSQL...")

    conn = psycopg2.connect(dbname="lnsp")
    cur = conn.cursor()

    # Query to get article/chunk ordering
    query = """
        SELECT
            cpe_id,
            (chunk_position->>'article_index')::int as article_index,
            (chunk_position->>'chunk_index')::int as chunk_index
        FROM cpe_entry
        WHERE dataset_source = %s
        ORDER BY article_index, chunk_index;
    """

    cur.execute(query, (dataset_source,))
    rows = cur.fetchall()

    ordering = {}
    for cpe_id, article_idx, chunk_idx in rows:
        ordering[str(cpe_id)] = (article_idx, chunk_idx)

    cur.close()
    conn.close()

    print(f"   ✅ Loaded ordering for {len(ordering)} chunks")
    return ordering


def create_sequences_from_npz(
    npz_path: str,
    ordering: dict,
    context_size: int = 5,
    min_article_chunks: int = 7
):
    """
    Create training sequences from NPZ file using article ordering.

    For each article:
    1. Group chunks by article_index
    2. Sort by chunk_index
    3. Create sliding windows: [chunk_0...chunk_4] -> chunk_5

    Args:
        npz_path: Path to NPZ file with vectors
        ordering: Dict mapping cpe_id -> (article_index, chunk_index)
        context_size: Number of context vectors (default 5)
        min_article_chunks: Minimum chunks per article (default 7 = 5 ctx + 1 target + 1 extra)

    Returns:
        contexts: List of context arrays [context_size, 768]
        targets: List of target vectors [768]
        metadata: Dict with article_ids, sequence_ids for debugging
    """
    print(f"📖 Loading vectors from {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    vectors = data['vectors']  # [N, 768]
    cpe_ids = data['cpe_ids']  # [N] strings

    print(f"   Loaded {len(vectors)} vectors")
    print(f"   Vector shape: {vectors.shape}")

    # Group chunks by article
    print("🔗 Grouping chunks by article...")
    articles = defaultdict(list)

    skipped_no_ordering = 0
    for i, cpe_id in enumerate(tqdm(cpe_ids, desc="Processing chunks")):
        cpe_id_str = str(cpe_id)

        if cpe_id_str not in ordering:
            skipped_no_ordering += 1
            continue

        article_idx, chunk_idx = ordering[cpe_id_str]
        articles[article_idx].append({
            'chunk_index': chunk_idx,
            'vector': vectors[i],
            'cpe_id': cpe_id_str
        })

    if skipped_no_ordering > 0:
        print(f"   ⚠️  Skipped {skipped_no_ordering} chunks without ordering info")

    print(f"   ✅ Grouped into {len(articles)} articles")

    # Create sequences
    print(f"🎯 Creating training sequences (context_size={context_size})...")

    contexts = []
    targets = []
    article_ids = []
    sequence_ids = []

    articles_used = 0
    articles_too_short = 0
    sequence_id = 0

    for article_idx in tqdm(sorted(articles.keys()), desc="Creating sequences"):
        chunks = articles[article_idx]

        # Skip articles with too few chunks
        if len(chunks) < min_article_chunks:
            articles_too_short += 1
            continue

        # Sort by chunk_index to maintain temporal order
        chunks = sorted(chunks, key=lambda x: x['chunk_index'])

        # Extract vectors
        article_vectors = np.array([chunk['vector'] for chunk in chunks])

        # Create sliding window sequences
        for i in range(len(article_vectors) - context_size):
            ctx = article_vectors[i:i+context_size]       # [context_size, 768]
            tgt = article_vectors[i+context_size]          # [768]

            contexts.append(ctx)
            targets.append(tgt)
            article_ids.append(article_idx)
            sequence_ids.append(sequence_id)
            sequence_id += 1

        articles_used += 1

    print()
    print(f"✅ Sequence creation complete:")
    print(f"   Articles used: {articles_used}")
    print(f"   Articles too short (skipped): {articles_too_short}")
    print(f"   Training sequences: {len(contexts)}")
    print()

    return {
        'contexts': np.array(contexts),        # [N, context_size, 768]
        'targets': np.array(targets),          # [N, 768]
        'article_ids': np.array(article_ids),  # [N]
        'sequence_ids': np.array(sequence_ids) # [N]
    }


def save_train_val_split(
    data: dict,
    output_path: str,
    val_split: float = 0.1
):
    """
    Split into train/val and save to NPZ format compatible with train_unified.py
    """
    print(f"💾 Saving training data...")

    contexts = data['contexts']
    targets = data['targets']

    print(f"   Context sequences shape: {contexts.shape}")
    print(f"   Target vectors shape: {targets.shape}")

    # Shuffle data
    N = len(contexts)
    indices = np.random.permutation(N)
    contexts = contexts[indices]
    targets = targets[indices]

    # Split train/val
    val_size = int(N * val_split)
    train_size = N - val_size

    train_ctx = contexts[:train_size]
    train_tgt = targets[:train_size]

    val_ctx = contexts[train_size:]
    val_tgt = targets[train_size:]

    # Save in format expected by train_unified.py
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        # Training data (required by train_unified.py)
        context_sequences=train_ctx,
        target_vectors=train_tgt,
        # Validation data
        val_context_sequences=val_ctx,
        val_target_vectors=val_tgt
    )

    print(f"✅ Saved to {output_path}")
    print(f"   Train: {len(train_ctx):,} sequences ({len(train_ctx)/N*100:.1f}%)")
    print(f"   Val:   {len(val_ctx):,} sequences ({len(val_ctx)/N*100:.1f}%)")
    print()

    # File size
    file_size = Path(output_path).stat().st_size / (1024**2)  # MB
    print(f"   File size: {file_size:.1f} MB")
    print()


def main():
    parser = argparse.ArgumentParser(description="Create LVM training sequences from Wikipedia NPZ")
    parser.add_argument(
        "--npz",
        default="artifacts/wikipedia_500k_corrected_vectors.npz",
        help="Input NPZ file with Wikipedia vectors"
    )
    parser.add_argument(
        "--output",
        default="artifacts/lvm/wikipedia_fresh_sequences_ctx5.npz",
        help="Output NPZ file for training sequences"
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=5,
        help="Number of context vectors (default: 5)"
    )
    parser.add_argument(
        "--min-article-chunks",
        type=int,
        default=7,
        help="Minimum chunks per article (default: 7)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--dataset-source",
        default="wikipedia_500k",
        help="Dataset source in PostgreSQL (default: wikipedia_500k)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CREATE LVM TRAINING SEQUENCES FROM FRESH WIKIPEDIA DATA")
    print("=" * 80)
    print()

    # Step 1: Load article ordering from PostgreSQL
    ordering = load_article_ordering_from_db(args.dataset_source)

    # Step 2: Create sequences from NPZ
    data = create_sequences_from_npz(
        args.npz,
        ordering,
        context_size=args.context_size,
        min_article_chunks=args.min_article_chunks
    )

    if len(data['contexts']) == 0:
        print("❌ No training sequences created!")
        return

    # Step 3: Save train/val split
    save_train_val_split(
        data,
        args.output,
        val_split=args.val_split
    )

    print("=" * 80)
    print("✅ TRAINING SEQUENCES READY!")
    print("=" * 80)
    print()
    print("Next steps:")
    print()
    print("1. Train LSTM (recommended for production):")
    print(f"   ./.venv/bin/python app/lvm/train_unified.py \\")
    print(f"       --model-type lstm \\")
    print(f"       --data {args.output} \\")
    print(f"       --epochs 20 \\")
    print(f"       --batch-size 32 \\")
    print(f"       --lambda-mse 1.0")
    print()
    print("2. Train AMN (fastest, low latency):")
    print(f"   ./.venv/bin/python app/lvm/train_unified.py \\")
    print(f"       --model-type amn \\")
    print(f"       --data {args.output} \\")
    print(f"       --epochs 20")
    print()
    print("3. Train Transformer (best accuracy):")
    print(f"   ./.venv/bin/python app/lvm/train_unified.py \\")
    print(f"       --model-type transformer \\")
    print(f"       --data {args.output} \\")
    print(f"       --epochs 20")
    print()


if __name__ == "__main__":
    main()
