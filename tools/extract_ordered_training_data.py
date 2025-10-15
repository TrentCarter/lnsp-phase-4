#!/usr/bin/env python3
"""
Extract ordered training data from PostgreSQL for LVM training.

CRITICAL: Preserves temporal order using created_at timestamp!
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime

def extract_ordered_data(
    db_name: str = "lnsp",
    output_dir: str = "artifacts/lvm",
    dataset_source: str = "user_input"
):
    """
    Extract all concepts and vectors in temporal order.

    Order is CRITICAL for LVM training - concepts must be in the sequence
    they appeared in the original documents.
    """

    print(f"🔄 Extracting ordered training data from {db_name}...")
    print(f"📊 Dataset source: {dataset_source}")

    # Connect to database
    conn = psycopg2.connect(f"dbname={db_name}")
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Get all data in proper article+sequence order (CRITICAL!)
    print("\n⏳ Fetching data ordered by article (batch_id) then sequence (created_at)...")
    cur.execute("""
        SELECT
            e.cpe_id,
            e.concept_text,
            e.tmd_bits,
            e.domain_code,
            e.task_code,
            e.modifier_code,
            e.batch_id,
            e.created_at,
            v.concept_vec,
            ROW_NUMBER() OVER (PARTITION BY e.batch_id ORDER BY e.created_at) as seq_in_article
        FROM cpe_entry e
        JOIN cpe_vectors v ON e.cpe_id = v.cpe_id
        WHERE e.dataset_source = %s
        ORDER BY e.batch_id ASC, e.created_at ASC
    """, (dataset_source,))

    rows = cur.fetchall()
    print(f"✅ Loaded {len(rows):,} concepts in temporal order")

    # Extract arrays
    print("\n📦 Extracting arrays...")
    cpe_ids = [str(row['cpe_id']) for row in rows]
    concept_texts = [row['concept_text'] for row in rows]
    tmd_codes = [row['tmd_bits'] for row in rows]
    domain_codes = [row['domain_code'] for row in rows]
    task_codes = [row['task_code'] for row in rows]
    modifier_codes = [row['modifier_code'] for row in rows]
    batch_ids = [str(row['batch_id']) for row in rows]
    seq_in_article = [row['seq_in_article'] for row in rows]
    timestamps = [row['created_at'].isoformat() for row in rows]

    # Count unique articles
    unique_articles = len(set(batch_ids))
    print(f"   Total articles: {unique_articles:,}")
    print(f"   Avg concepts per article: {len(rows)/unique_articles:.1f}")

    # Convert vectors to numpy array (handle pgvector type)
    print("   Converting pgvector types...")
    vectors_list = []
    for row in rows:
        vec = row['concept_vec']
        # pgvector returns as string, parse it
        if isinstance(vec, str):
            # Remove brackets and split by comma
            vec_str = vec.strip('[]')
            vec_floats = [float(x) for x in vec_str.split(',')]
            vectors_list.append(vec_floats)
        else:
            # Already a list/array
            vectors_list.append(list(vec))

    vectors = np.array(vectors_list, dtype=np.float32)
    print(f"   Vectors shape: {vectors.shape}")
    print(f"   Vector dtype: {vectors.dtype}")

    # Verify order is maintained
    print("\n✅ ORDER VERIFICATION (Article → Sequence):")
    print(f"   First article ID: {batch_ids[0]}")
    print(f"   First concept [seq={seq_in_article[0]}]: {concept_texts[0][:80]}...")
    print(f"   Last article ID: {batch_ids[-1]}")
    print(f"   Last concept [seq={seq_in_article[-1]}]: {concept_texts[-1][:80]}...")

    # Show example of article sequence
    first_article = batch_ids[0]
    article_concepts = [(i, concept_texts[i]) for i in range(len(batch_ids)) if batch_ids[i] == first_article]
    print(f"\n   Example: First article has {len(article_concepts)} concepts:")
    for i, (idx, text) in enumerate(article_concepts[:3], 1):
        print(f"      [{i}] {text[:60]}...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save to NPZ
    npz_file = output_path / f"wikipedia_{len(rows)}_ordered.npz"
    print(f"\n💾 Saving to {npz_file}...")

    np.savez_compressed(
        npz_file,
        cpe_ids=np.array(cpe_ids, dtype=object),
        concept_texts=np.array(concept_texts, dtype=object),
        vectors=vectors,
        tmd_codes=np.array(tmd_codes, dtype=np.int32),
        domain_codes=np.array(domain_codes, dtype=np.int16),
        task_codes=np.array(task_codes, dtype=np.int16),
        modifier_codes=np.array(modifier_codes, dtype=np.int16),
        batch_ids=np.array(batch_ids, dtype=object),
        seq_in_article=np.array(seq_in_article, dtype=np.int32),
        timestamps=np.array(timestamps, dtype=object),
        metadata=np.array([{
            'total_concepts': len(rows),
            'total_articles': unique_articles,
            'avg_concepts_per_article': len(rows) / unique_articles,
            'vector_dim': vectors.shape[1],
            'dataset_source': dataset_source,
            'extraction_date': datetime.now().isoformat(),
            'ordering': 'article-based (batch_id ASC, created_at ASC)'
        }], dtype=object)
    )

    print(f"✅ Saved {len(rows):,} ordered concepts to {npz_file}")

    # Save metadata JSON
    meta_file = output_path / f"wikipedia_{len(rows)}_ordered_metadata.json"
    metadata = {
        'total_concepts': len(rows),
        'total_articles': unique_articles,
        'avg_concepts_per_article': len(rows) / unique_articles,
        'vector_dimensions': int(vectors.shape[1]),
        'unique_tmd_codes': len(set(tmd_codes)),
        'unique_domains': len(set(domain_codes)),
        'unique_tasks': len(set(task_codes)),
        'unique_modifiers': len(set(modifier_codes)),
        'dataset_source': dataset_source,
        'extraction_date': datetime.now().isoformat(),
        'ordering': 'article-based (batch_id ASC, created_at ASC)',
        'first_article_id': batch_ids[0],
        'last_article_id': batch_ids[-1],
        'first_concept': concept_texts[0],
        'last_concept': concept_texts[-1],
        'npz_file': str(npz_file),
        'sample_article_sequences': [
            {
                'article_id': batch_ids[i],
                'seq_in_article': seq_in_article[i],
                'concept': concept_texts[i][:200]
            }
            for i in [0, 1, 2, 100, 1000, 10000, len(rows)-1]
            if i < len(rows)
        ]
    }

    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved metadata to {meta_file}")

    # Close connection
    cur.close()
    conn.close()

    return npz_file, meta_file


def create_training_sequences(
    npz_file: Path,
    context_size: int = 5,
    output_dir: str = "artifacts/lvm"
):
    """
    Create training sequences from ordered concepts.

    Format: [context_vectors] → target_vector

    Example with context_size=5:
        Concepts [0,1,2,3,4] → predict concept 5
        Concepts [1,2,3,4,5] → predict concept 6
        etc.
    """

    print(f"\n🔄 Creating training sequences (context_size={context_size})...")

    # Load ordered data
    data = np.load(npz_file, allow_pickle=True)
    vectors = data['vectors']
    concept_texts = data['concept_texts']
    tmd_codes = data['tmd_codes']
    cpe_ids = data['cpe_ids']

    print(f"📊 Loaded {len(vectors):,} ordered vectors")
    print(f"   Vector shape: {vectors.shape}")

    # Create sequences
    context_sequences = []
    target_vectors = []
    target_texts = []
    target_tmds = []
    target_ids = []
    sequence_positions = []

    print(f"\n⏳ Generating sequences...")
    for i in range(len(vectors) - context_size):
        context = vectors[i:i+context_size]  # Shape: (context_size, 768)
        target = vectors[i+context_size]      # Shape: (768,)

        context_sequences.append(context)
        target_vectors.append(target)
        target_texts.append(concept_texts[i+context_size])
        target_tmds.append(tmd_codes[i+context_size])
        target_ids.append(cpe_ids[i+context_size])
        sequence_positions.append(i)

    # Convert to numpy arrays
    context_sequences = np.array(context_sequences, dtype=np.float32)
    target_vectors = np.array(target_vectors, dtype=np.float32)

    print(f"✅ Created {len(context_sequences):,} training sequences")
    print(f"   Context shape: {context_sequences.shape}")
    print(f"   Target shape:  {target_vectors.shape}")

    # Show examples
    print("\n📝 EXAMPLE SEQUENCES (first 3):")
    for i in range(min(3, len(context_sequences))):
        print(f"\n   Sequence {i+1}:")
        print(f"   Context concepts {i} → {i+context_size-1}:")
        for j in range(context_size):
            print(f"      [{i+j}] {concept_texts[i+j][:60]}...")
        print(f"   ↓ PREDICT:")
        print(f"   Target concept [{i+context_size}]: {target_texts[i][:80]}...")

    # Save training sequences
    output_path = Path(output_dir)
    seq_file = output_path / f"training_sequences_ctx{context_size}.npz"

    print(f"\n💾 Saving training sequences to {seq_file}...")
    np.savez_compressed(
        seq_file,
        context_sequences=context_sequences,
        target_vectors=target_vectors,
        target_texts=np.array(target_texts, dtype=object),
        target_tmds=np.array(target_tmds, dtype=np.int32),
        target_ids=np.array(target_ids, dtype=object),
        sequence_positions=np.array(sequence_positions, dtype=np.int32),
        metadata=np.array([{
            'num_sequences': len(context_sequences),
            'context_size': context_size,
            'vector_dim': vectors.shape[1],
            'source_npz': str(npz_file),
            'creation_date': datetime.now().isoformat()
        }], dtype=object)
    )

    print(f"✅ Saved {len(context_sequences):,} training sequences")

    return seq_file


def main():
    parser = argparse.ArgumentParser(
        description="Extract ordered training data and create sequences for LVM training"
    )
    parser.add_argument(
        '--db', default='lnsp',
        help='Database name (default: lnsp)'
    )
    parser.add_argument(
        '--output-dir', default='artifacts/lvm',
        help='Output directory (default: artifacts/lvm)'
    )
    parser.add_argument(
        '--dataset-source', default='user_input',
        help='Dataset source filter (default: user_input)'
    )
    parser.add_argument(
        '--context-size', type=int, default=5,
        help='Number of context vectors for training sequences (default: 5)'
    )
    parser.add_argument(
        '--skip-sequences', action='store_true',
        help='Skip creating training sequences (only extract data)'
    )

    args = parser.parse_args()

    # Extract ordered data
    npz_file, meta_file = extract_ordered_data(
        db_name=args.db,
        output_dir=args.output_dir,
        dataset_source=args.dataset_source
    )

    # Create training sequences
    if not args.skip_sequences:
        seq_file = create_training_sequences(
            npz_file=npz_file,
            context_size=args.context_size,
            output_dir=args.output_dir
        )

        print("\n" + "="*60)
        print("✅ EXTRACTION COMPLETE")
        print("="*60)
        print(f"📦 Ordered data:       {npz_file}")
        print(f"📝 Metadata:           {meta_file}")
        print(f"🎯 Training sequences: {seq_file}")
        print(f"🔢 Context size:       {args.context_size}")
        print("\n🚀 Ready for LVM training!")
    else:
        print("\n" + "="*60)
        print("✅ DATA EXTRACTION COMPLETE")
        print("="*60)
        print(f"📦 Ordered data: {npz_file}")
        print(f"📝 Metadata:     {meta_file}")


if __name__ == '__main__':
    main()
