#!/usr/bin/env python3
"""
Rebuild FAISS index with CORRECTED 768D vectors from PostgreSQL.

Exports from cpe_vectors.concept_vec (CORRECT encoder) instead of sentence-transformers.
Creates both:
1. NPZ file with vectors + metadata (for LVM training)
2. FAISS IVF index (for retrieval)
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db_postgres import connect as connect_pg

try:
    import faiss
except ImportError:
    print("ERROR: faiss-cpu not installed. Run: ./.venv/bin/pip install faiss-cpu")
    sys.exit(1)

# Configuration
DATASET_SOURCE = 'wikipedia_500k'
OUTPUT_DIR = Path("artifacts")
INDEX_TYPE = "ivf_flat"  # IVF_FLAT for exact distances within clusters
METRIC = "ip"  # Inner Product (cosine similarity when normalized)
NLIST = 512  # Number of clusters (good for 80k vectors)
NPROBE = 16  # Number of clusters to search at query time

def main():
    print("=" * 80)
    print("Rebuild FAISS Index with CORRECTED Vectors")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Dataset source:  {DATASET_SOURCE}")
    print(f"  Output dir:      {OUTPUT_DIR}")
    print(f"  Index type:      {INDEX_TYPE}")
    print(f"  Metric:          {METRIC} (cosine similarity)")
    print(f"  nlist:           {NLIST} clusters")
    print(f"  nprobe:          {NPROBE} clusters to search")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================================
    # Step 1: Export CORRECTED Vectors from PostgreSQL
    # ============================================================================

    print("Step 1: Exporting CORRECTED 768D vectors from PostgreSQL...")
    print()

    conn = connect_pg()
    cur = conn.cursor()

    # Fetch all vectors in order
    query = """
    SELECT
        e.cpe_id::text,
        e.concept_text,
        v.concept_vec,
        e.tmd_lane,
        e.lane_index,
        e.created_at
    FROM cpe_entry e
    JOIN cpe_vectors v USING (cpe_id)
    WHERE e.dataset_source = %s
    ORDER BY e.created_at
    """

    print(f"Fetching vectors for dataset: {DATASET_SOURCE}...")
    cur.execute(query, (DATASET_SOURCE,))
    rows = cur.fetchall()

    print(f"✓ Fetched {len(rows):,} rows")
    print()

    if not rows:
        print("✗ No vectors found! Exiting.")
        sys.exit(1)

    # Parse vectors
    print("Parsing vectors...")
    cpe_ids = []
    concept_texts = []
    vectors = []
    tmd_lanes = []
    lane_indices = []
    created_ats = []

    for idx, row in enumerate(rows):
        cpe_id, concept_text, vec_str, tmd_lane, lane_index, created_at = row

        if vec_str is None:
            print(f"⚠️  Row {idx+1}: Skipping null vector for {cpe_id}")
            continue

        # Parse vector string "[0.123, -0.456, ...]"
        try:
            vec_array = np.array([float(x) for x in vec_str.strip('[]').split(',')], dtype=np.float32)
        except Exception as e:
            print(f"⚠️  Row {idx+1}: Failed to parse vector for {cpe_id}: {e}")
            continue

        if vec_array.shape[0] != 768:
            print(f"⚠️  Row {idx+1}: Wrong dimension {vec_array.shape[0]} for {cpe_id} (expected 768)")
            continue

        cpe_ids.append(cpe_id)
        concept_texts.append(concept_text or "")
        vectors.append(vec_array)
        tmd_lanes.append(tmd_lane or "")
        lane_indices.append(lane_index or 0)
        created_ats.append(str(created_at))

        if (idx + 1) % 10000 == 0:
            print(f"  Parsed {idx+1:,} / {len(rows):,} vectors...")

    cur.close()
    conn.close()

    vectors_array = np.vstack(vectors).astype('float32')

    print()
    print(f"✓ Parsed {len(vectors):,} vectors")
    print(f"  Shape: {vectors_array.shape}")
    print(f"  Dtype: {vectors_array.dtype}")
    print()

    # ============================================================================
    # Step 2: Save NPZ File (for LVM Training)
    # ============================================================================

    print("Step 2: Saving NPZ file with vectors and metadata...")
    print()

    npz_path = OUTPUT_DIR / f"{DATASET_SOURCE}_corrected_vectors.npz"

    print(f"Saving to: {npz_path}")
    np.savez_compressed(
        npz_path,
        vectors=vectors_array,
        cpe_ids=np.array(cpe_ids, dtype=object),
        concept_texts=np.array(concept_texts, dtype=object),
        tmd_lanes=np.array(tmd_lanes, dtype=object),
        lane_indices=np.array(lane_indices, dtype=np.int16),
        created_ats=np.array(created_ats, dtype=object)
    )

    npz_size_mb = npz_path.stat().st_size / 1024 / 1024
    print(f"✓ NPZ file saved: {npz_size_mb:.1f} MB")
    print()

    # ============================================================================
    # Step 3: Build FAISS IVF Index
    # ============================================================================

    print("Step 3: Building FAISS IVF index...")
    print()

    N, D = vectors_array.shape
    print(f"Building IVF index:")
    print(f"  Vectors: {N:,}")
    print(f"  Dimension: {D}")
    print(f"  Clusters: {NLIST}")
    print(f"  Metric: {METRIC.upper()} (Inner Product)")
    print()

    # Normalize for cosine similarity
    print("Normalizing vectors for cosine similarity...")
    faiss.normalize_L2(vectors_array)
    print("✓ Vectors normalized")
    print()

    # Build IVF index
    print(f"Creating IVF quantizer...")
    quantizer = faiss.IndexFlatIP(D)  # Inner Product quantizer

    print(f"Creating IVF index with {NLIST} clusters...")
    index = faiss.IndexIVFFlat(quantizer, D, NLIST, faiss.METRIC_INNER_PRODUCT)

    print("Training index (clustering vectors)...")
    index.train(vectors_array)
    print("✓ Index trained")
    print()

    print("Adding vectors to index...")
    index.add(vectors_array)
    print(f"✓ Added {index.ntotal:,} vectors")
    print()

    # Set search parameters
    index.nprobe = NPROBE
    print(f"Set nprobe={NPROBE} for search time")
    print()

    # ============================================================================
    # Step 4: Save FAISS Index
    # ============================================================================

    print("Step 4: Saving FAISS index...")
    print()

    index_path = OUTPUT_DIR / f"{DATASET_SOURCE}_corrected_{INDEX_TYPE}_{METRIC}.index"

    print(f"Saving to: {index_path}")
    faiss.write_index(index, str(index_path))

    index_size_mb = index_path.stat().st_size / 1024 / 1024
    print(f"✓ FAISS index saved: {index_size_mb:.1f} MB")
    print()

    # ============================================================================
    # Step 5: Save Metadata
    # ============================================================================

    print("Step 5: Saving metadata...")
    print()

    import json

    metadata = {
        "dataset_source": DATASET_SOURCE,
        "num_vectors": int(N),
        "dimension": int(D),
        "index_type": INDEX_TYPE,
        "metric": METRIC,
        "nlist": NLIST,
        "nprobe": NPROBE,
        "npz_path": str(npz_path),
        "index_path": str(index_path),
        "npz_size_mb": npz_size_mb,
        "index_size_mb": index_size_mb,
        "encoder": "IsolatedVecTextVectOrchestrator (CORRECT)",
        "created_at": datetime.now().isoformat()
    }

    meta_path = OUTPUT_DIR / f"{DATASET_SOURCE}_corrected_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {meta_path}")
    print()

    # ============================================================================
    # Step 6: Verify Index Quality
    # ============================================================================

    print("Step 6: Verifying index quality...")
    print()

    # Test search with first vector
    test_vec = vectors_array[0:1]
    D_search, I_search = index.search(test_vec, k=10)

    print(f"Test search (first vector should match itself):")
    print(f"  Top-1 index: {I_search[0][0]} (expected 0)")
    print(f"  Top-1 score: {D_search[0][0]:.6f} (expected ~1.0)")
    print(f"  Top-5 indices: {I_search[0][:5]}")
    print(f"  Top-5 scores: {D_search[0][:5]}")
    print()

    if I_search[0][0] == 0 and D_search[0][0] > 0.99:
        print("✅ Index quality check PASSED!")
    else:
        print("⚠️  Index quality check FAILED - unexpected search results")

    print()

    # ============================================================================
    # Summary
    # ============================================================================

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    print(f"✅ FAISS index rebuilt successfully!")
    print()
    print(f"Dataset:         {DATASET_SOURCE}")
    print(f"Vectors:         {N:,}")
    print(f"Dimension:       {D}D (CORRECTED encoder)")
    print()
    print(f"NPZ file:        {npz_path} ({npz_size_mb:.1f} MB)")
    print(f"FAISS index:     {index_path} ({index_size_mb:.1f} MB)")
    print(f"Metadata:        {meta_path}")
    print()
    print(f"Index config:")
    print(f"  Type:          {INDEX_TYPE.upper()}")
    print(f"  Metric:        {METRIC.upper()} (cosine similarity)")
    print(f"  Clusters:      {NLIST}")
    print(f"  Search probe:  {NPROBE}")
    print()
    print("Next step: Export LVM training data chains!")
    print()
    print("=" * 80)

if __name__ == '__main__':
    main()
