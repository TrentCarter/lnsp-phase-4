#!/usr/bin/env python3
"""
Export Wikipedia Vectors from PostgreSQL to NPZ

Exports the fresh Wikipedia ingestion data (584k chunks, 8,447 articles)
from PostgreSQL to NPZ format for LVM training sequence creation.

Usage:
    python tools/export_wikipedia_vectors_from_pg.py \
        --output artifacts/wikipedia_fresh_584k_vectors.npz \
        --dataset-source wikipedia_500k
"""

import argparse
import numpy as np
import psycopg2
from pathlib import Path
from tqdm import tqdm


def export_vectors_from_pg(
    dataset_source: str = "wikipedia_500k",
    output_path: str = "artifacts/wikipedia_fresh_584k_vectors.npz"
):
    """
    Export all vectors and metadata from PostgreSQL.
    """
    print(f"📊 Exporting vectors from PostgreSQL...")
    print(f"   Dataset source: {dataset_source}")
    print()

    conn = psycopg2.connect(dbname="lnsp")
    cur = conn.cursor()

    # Get count
    cur.execute("""
        SELECT COUNT(*)
        FROM cpe_entry e
        JOIN cpe_vectors v ON e.cpe_id = v.cpe_id
        WHERE e.dataset_source = %s;
    """, (dataset_source,))
    total_count = cur.fetchone()[0]
    print(f"   Total chunks: {total_count:,}")

    # Query all data ordered by article and chunk
    query = """
        SELECT
            e.cpe_id,
            e.concept_text,
            (e.chunk_position->>'article_index')::int as article_index,
            (e.chunk_position->>'chunk_index')::int as chunk_index,
            (e.chunk_position->>'article_title') as article_title,
            v.concept_vec
        FROM cpe_entry e
        JOIN cpe_vectors v ON e.cpe_id = v.cpe_id
        WHERE e.dataset_source = %s
        ORDER BY article_index, chunk_index;
    """

    print("   Fetching data from PostgreSQL...")
    cur.execute(query, (dataset_source,))

    # Collect data
    cpe_ids = []
    concept_texts = []
    article_indices = []
    chunk_indices = []
    article_titles = []
    vectors = []

    print("   Processing rows...")
    for row in tqdm(cur, total=total_count, desc="Exporting"):
        cpe_id, concept_text, article_idx, chunk_idx, article_title, vec = row

        cpe_ids.append(str(cpe_id))
        concept_texts.append(concept_text)
        article_indices.append(article_idx)
        chunk_indices.append(chunk_idx)
        article_titles.append(article_title or "")

        # Convert pgvector to numpy
        if isinstance(vec, str):
            # Parse string format: '[0.1, 0.2, ...]'
            vec_str = vec.strip('[]')
            vec_arr = np.array([float(x) for x in vec_str.split(',')])
        else:
            vec_arr = np.array(vec)

        vectors.append(vec_arr)

    cur.close()
    conn.close()

    # Convert to numpy arrays
    print()
    print("   Converting to numpy arrays...")
    vectors = np.array(vectors, dtype=np.float32)
    cpe_ids = np.array(cpe_ids)
    concept_texts = np.array(concept_texts)
    article_indices = np.array(article_indices, dtype=np.int32)
    chunk_indices = np.array(chunk_indices, dtype=np.int32)
    article_titles = np.array(article_titles)

    print(f"   Vectors shape: {vectors.shape}")
    print(f"   Vector dtype: {vectors.dtype}")
    print()

    # Save to NPZ
    print(f"💾 Saving to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        vectors=vectors,
        cpe_ids=cpe_ids,
        concept_texts=concept_texts,
        article_indices=article_indices,
        chunk_indices=chunk_indices,
        article_titles=article_titles
    )

    file_size = Path(output_path).stat().st_size / (1024**2)  # MB
    print(f"✅ Saved {len(vectors):,} vectors ({file_size:.1f} MB)")
    print()

    # Statistics
    unique_articles = len(np.unique(article_indices))
    print("📊 Statistics:")
    print(f"   Total chunks: {len(vectors):,}")
    print(f"   Unique articles: {unique_articles:,}")
    print(f"   Article range: {article_indices.min()} - {article_indices.max()}")
    print(f"   Avg chunks/article: {len(vectors)/unique_articles:.1f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Export Wikipedia vectors from PostgreSQL to NPZ")
    parser.add_argument(
        "--output",
        default="artifacts/wikipedia_fresh_584k_vectors.npz",
        help="Output NPZ file path"
    )
    parser.add_argument(
        "--dataset-source",
        default="wikipedia_500k",
        help="Dataset source in PostgreSQL (default: wikipedia_500k)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("EXPORT WIKIPEDIA VECTORS FROM POSTGRESQL")
    print("=" * 80)
    print()

    export_vectors_from_pg(
        dataset_source=args.dataset_source,
        output_path=args.output
    )

    print("=" * 80)
    print("✅ EXPORT COMPLETE!")
    print("=" * 80)
    print()
    print("Next step: Create training sequences")
    print(f"  ./.venv/bin/python tools/create_sequences_from_npz_simple.py \\")
    print(f"      --npz {args.output} \\")
    print(f"      --output artifacts/lvm/wikipedia_fresh_sequences_ctx5.npz")
    print()


if __name__ == "__main__":
    main()
