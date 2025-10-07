#!/usr/bin/env python3
"""
Rebuild FAISS index from NPZ file

Usage:
    python tools/rebuild_faiss_index.py artifacts/ontology_13k.npz artifacts/ontology_13k_ivf_flat_ip.index
"""
import sys
import numpy as np
import faiss
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.norms import l2_normalize


def build_faiss_index(vectors, nlist=128, metric='ip'):
    """Build IVF FAISS index from vectors."""
    d = vectors.shape[1]
    n = vectors.shape[0]

    # Normalize vectors for inner product
    vecs = l2_normalize(vectors).astype(np.float32)

    # Build IVF index
    nlist = max(8, min(nlist, n // 10))  # Adaptive nlist
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train on subset
    train_size = min(n, max(nlist * 40, n // 2))
    train_vecs = vecs[:train_size]
    print(f"Training index on {train_size} vectors...")
    index.train(train_vecs)

    # Add all vectors
    print(f"Adding {n} vectors to index...")
    index.add(vecs)

    # Set nprobe
    index.nprobe = 16

    print(f"✓ Built index: {n} vectors, {d}D, nlist={nlist}, nprobe={index.nprobe}")
    return index


def main():
    if len(sys.argv) < 3:
        print("Usage: python rebuild_faiss_index.py <npz_path> <output_index_path>")
        sys.exit(1)

    npz_path = sys.argv[1]
    index_path = sys.argv[2]

    print(f"Loading NPZ from {npz_path}...")
    npz = np.load(npz_path, allow_pickle=True)

    # Try different vector keys
    if 'fused' in npz:
        vectors = npz['fused']
        print(f"Using 'fused' vectors: {vectors.shape}")
    elif 'vectors' in npz:
        vectors = npz['vectors']
        print(f"Using 'vectors': {vectors.shape}")
    elif 'concept_vecs' in npz:
        vectors = npz['concept_vecs']
        print(f"Using 'concept_vecs': {vectors.shape}")
    else:
        print(f"ERROR: No vector array found in NPZ. Keys: {list(npz.keys())}")
        sys.exit(1)

    # Build index
    index = build_faiss_index(vectors, nlist=128)

    # Save index
    print(f"Saving index to {index_path}...")
    faiss.write_index(index, index_path)
    print(f"✓ Index saved: {index_path}")

    # Verify
    print("\nVerifying index...")
    loaded = faiss.read_index(index_path)
    print(f"✓ Loaded index has {loaded.ntotal} vectors")

    # Test search
    query = vectors[0:1]
    query_norm = l2_normalize(query).astype(np.float32)
    D, I = loaded.search(query_norm, 5)
    print(f"✓ Search test passed: found {len(I[0])} results")
    print(f"  Top result: index={I[0][0]}, score={D[0][0]:.4f}")


if __name__ == "__main__":
    main()
