#!/usr/bin/env python3
"""
Build Faiss IVF index from Postgres cpe_vectors table.

Task 1.2 from Sprint 1: Build optimized Faiss IVF (Inverted File) index 
from 9,477×784D fused vectors for sub-second retrieval.

Index Configuration:
- Type: IVF_FLAT (exact distances within clusters)
- Metric: Inner Product (IP) for cosine similarity
- nlist: 128 (number of clusters for 9.5K vectors)
- nprobe: 16 (search 16 clusters at query time)
- Dimension: 784 (768D concept + 16D TMD)

Performance Target:
- Query time: <10ms for top-10 retrieval
- Recall@10: >95%
"""

import os
import sys
import numpy as np
from pathlib import Path

try:
    import faiss
except ImportError:
    print("ERROR: faiss-cpu not installed. Run: pip install faiss-cpu")
    sys.exit(1)

try:
    import psycopg2
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)


def get_pg_dsn():
    """Get Postgres DSN from environment or default."""
    return os.getenv("PG_DSN", "host=localhost dbname=lnsp user=lnsp password=lnsp")


def load_vectors_from_postgres(dsn=None):
    """
    Load fused vectors from Postgres cpe_vectors table.
    
    Returns:
        tuple: (cpe_ids, vectors, metadata)
            - cpe_ids: numpy array of CPE IDs
            - vectors: numpy array of shape [N, 784] float32
            - metadata: dict with additional fields (doc_ids, lane_indices, etc.)
    """
    if dsn is None:
        dsn = get_pg_dsn()
    
    print(f"[FaissBuilder] Connecting to Postgres: {dsn.split('password=')[0]}...")
    
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    
    # Query all vectors ordered by cpe_id for reproducibility
    query = """
        SELECT
            v.cpe_id::text,
            v.fused_vec,
            c.concept_text,
            c.tmd_lane,
            c.lane_index
        FROM cpe_vectors v
        LEFT JOIN cpe_entry c ON v.cpe_id = c.cpe_id
        ORDER BY v.cpe_id;
    """
    
    print(f"[FaissBuilder] Executing query to fetch vectors...")
    cur.execute(query)
    
    cpe_ids = []
    vectors = []
    concept_texts = []
    tmd_lanes = []
    lane_indices = []
    
    for row in cur.fetchall():
        cpe_id, fused_vec, concept_text, tmd_lane, lane_index = row
        
        if fused_vec is None:
            print(f"[FaissBuilder] WARNING: Skipping {cpe_id} - null fused_vec")
            continue

        # Convert pgvector to numpy array
        # pgvector returns as string: "[-0.06,0.06,...]"
        if isinstance(fused_vec, str):
            # Parse pgvector string format
            vec_data = fused_vec.strip('[]{}').split(',')
            vec_array = np.array([float(v.strip()) for v in vec_data], dtype=np.float32)
        elif isinstance(fused_vec, list):
            vec_array = np.array(fused_vec, dtype=np.float32)
        else:
            # Already an array-like object
            vec_array = np.array(fused_vec, dtype=np.float32)
        
        if vec_array.shape[0] != 784:
            print(f"[FaissBuilder] WARNING: Skipping {cpe_id} - wrong dimension {vec_array.shape[0]} (expected 784)")
            continue
        
        cpe_ids.append(cpe_id)
        vectors.append(vec_array)
        concept_texts.append(concept_text or "")
        tmd_lanes.append(tmd_lane or 0)
        lane_indices.append(lane_index or 0)
    
    cur.close()
    conn.close()
    
    if not vectors:
        raise RuntimeError("No valid vectors found in database")
    
    vectors_array = np.vstack(vectors).astype('float32')
    print(f"[FaissBuilder] Loaded {len(vectors)} vectors, shape: {vectors_array.shape}")
    
    metadata = {
        'cpe_ids': np.array(cpe_ids),
        'concept_texts': np.array(concept_texts),
        'tmd_lanes': np.array(tmd_lanes),
        'lane_indices': np.array(lane_indices)
    }
    
    return np.array(cpe_ids), vectors_array, metadata


def build_faiss_ivf_index(vectors, nlist=128, nprobe=16):
    """
    Build Faiss IVF_FLAT index with Inner Product metric.
    
    Args:
        vectors: numpy array of shape [N, 784] float32
        nlist: number of clusters (default 128 for ~10K vectors)
        nprobe: number of clusters to search at query time (default 16)
    
    Returns:
        faiss.IndexIVFFlat: trained index
    """
    N, dim = vectors.shape
    print(f"[FaissBuilder] Building IVF index: N={N}, dim={dim}, nlist={nlist}, nprobe={nprobe}")
    
    # Normalize for cosine similarity (IP = cosine when normalized)
    print(f"[FaissBuilder] Normalizing vectors...")
    faiss.normalize_L2(vectors)
    
    # Build IVF index
    print(f"[FaissBuilder] Creating IVF index with {nlist} clusters...")
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Train on all vectors (IVF requires training)
    print(f"[FaissBuilder] Training index on {N} vectors...")
    index.train(vectors)
    
    # Add vectors to index
    print(f"[FaissBuilder] Adding vectors to index...")
    index.add(vectors)
    
    # Set nprobe for query time
    index.nprobe = nprobe
    
    print(f"[FaissBuilder] Index built successfully: ntotal={index.ntotal}, is_trained={index.is_trained}")
    
    return index


def save_artifacts(index, cpe_ids, metadata, output_dir="artifacts"):
    """
    Save Faiss index and metadata to disk.
    
    Args:
        index: Faiss index
        cpe_ids: numpy array of CPE IDs
        metadata: dict with additional metadata
        output_dir: output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Faiss index
    index_path = output_dir / "fw9k_ivf_flat_ip.index"
    print(f"[FaissBuilder] Saving index to {index_path}...")
    faiss.write_index(index, str(index_path))
    index_size = index_path.stat().st_size
    print(f"[FaissBuilder] Index saved: {index_size:,} bytes ({index_size/1024/1024:.1f} MB)")
    
    # Save CPE ID mapping
    cpe_ids_path = output_dir / "fw9k_cpe_ids.npy"
    print(f"[FaissBuilder] Saving CPE IDs to {cpe_ids_path}...")
    np.save(cpe_ids_path, cpe_ids)
    
    # Save metadata NPZ (for compatibility with existing FaissDB loader)
    meta_path = output_dir / "fw9k_vectors.npz"
    print(f"[FaissBuilder] Saving metadata to {meta_path}...")
    np.savez(
        meta_path,
        cpe_ids=cpe_ids,
        lane_indices=metadata['lane_indices'],
        concept_texts=metadata['concept_texts'],
        tmd_lanes=metadata['tmd_lanes']
    )
    
    print(f"[FaissBuilder] All artifacts saved to {output_dir}")
    
    return {
        'index_path': str(index_path),
        'cpe_ids_path': str(cpe_ids_path),
        'meta_path': str(meta_path),
        'index_size_mb': index_size / 1024 / 1024,
        'num_vectors': len(cpe_ids)
    }


def build_faiss_from_postgres(dsn=None, output_dir="artifacts", nlist=128, nprobe=16):
    """
    Main entry point: Load vectors from Postgres and build Faiss IVF index.
    
    Args:
        dsn: Postgres connection string (default from env)
        output_dir: output directory for artifacts
        nlist: number of IVF clusters (default 128)
        nprobe: number of clusters to probe (default 16)
    
    Returns:
        dict: Summary with paths and metrics
    """
    print("=" * 60)
    print("Faiss IVF Index Builder - Sprint 1 Task 1.2")
    print("=" * 60)
    
    # Step 1: Load vectors from Postgres
    cpe_ids, vectors, metadata = load_vectors_from_postgres(dsn)
    
    # Step 2: Build Faiss index
    index = build_faiss_ivf_index(vectors, nlist=nlist, nprobe=nprobe)
    
    # Step 3: Save artifacts
    artifact_info = save_artifacts(index, cpe_ids, metadata, output_dir)
    
    print("=" * 60)
    print("✅ BUILD COMPLETE")
    print("=" * 60)
    print(f"Vectors indexed: {artifact_info['num_vectors']}")
    print(f"Index size: {artifact_info['index_size_mb']:.1f} MB")
    print(f"Index file: {artifact_info['index_path']}")
    print(f"CPE IDs: {artifact_info['cpe_ids_path']}")
    print(f"Metadata: {artifact_info['meta_path']}")
    print("=" * 60)
    
    return index, cpe_ids, artifact_info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Faiss IVF index from Postgres")
    parser.add_argument("--dsn", help="Postgres DSN (default from PG_DSN env)")
    parser.add_argument("--output-dir", default="artifacts", help="Output directory")
    parser.add_argument("--nlist", type=int, default=128, help="Number of IVF clusters")
    parser.add_argument("--nprobe", type=int, default=16, help="Number of clusters to probe")
    
    args = parser.parse_args()
    
    try:
        build_faiss_from_postgres(
            dsn=args.dsn,
            output_dir=args.output_dir,
            nlist=args.nlist,
            nprobe=args.nprobe
        )
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        sys.exit(1)
