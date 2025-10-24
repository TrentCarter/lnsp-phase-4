#!/usr/bin/env python3
"""
Build FAISS Index from NPZ File
================================

Builds a FAISS IVF index from Wikipedia NPZ export.

Usage:
    python tools/build_faiss_from_npz.py \
        --npz artifacts/wikipedia_584k_fresh.npz \
        --output artifacts/wikipedia_584k_ivf_flat_ip.index \
        --nlist 2048 \
        --nprobe 32
"""

import argparse
import numpy as np
import faiss
from pathlib import Path


def build_faiss_index(npz_path: str, output_path: str, nlist: int = 2048, nprobe: int = 32):
    """Build IVF FAISS index from NPZ file."""

    print("=" * 80)
    print("BUILD FAISS INDEX FROM NPZ")
    print("=" * 80)
    print()

    # Load vectors
    print(f"📥 Loading vectors from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    vectors = data['vectors']

    print(f"   Loaded {len(vectors):,} vectors")
    print(f"   Vector shape: {vectors.shape}")
    print(f"   Vector dtype: {vectors.dtype}")
    print()

    # Normalize vectors for cosine similarity (inner product on normalized = cosine)
    print("🔄 Normalizing vectors for cosine similarity...")
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    vectors_norm = vectors_norm.astype(np.float32)
    print(f"   ✅ Normalized {len(vectors_norm):,} vectors")
    print()

    # Build FAISS index
    dim = vectors_norm.shape[1]
    print(f"🔨 Building FAISS IVF index...")
    print(f"   Dimension: {dim}")
    print(f"   nlist: {nlist}")
    print(f"   nprobe: {nprobe}")
    print()

    # Create quantizer and IVF index
    quantizer = faiss.IndexFlatIP(dim)  # Inner product quantizer
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train index
    print("   Training index...")
    index.train(vectors_norm)
    print("   ✅ Training complete")

    # Add vectors
    print("   Adding vectors...")
    index.add(vectors_norm)
    print(f"   ✅ Added {index.ntotal:,} vectors")
    print()

    # Set default nprobe
    index.nprobe = nprobe
    print(f"   Set nprobe={nprobe}")
    print()

    # Save index
    print(f"💾 Saving index to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, output_path)

    file_size = Path(output_path).stat().st_size / (1024**2)  # MB
    print(f"✅ Saved index ({file_size:.1f} MB)")
    print()

    # Test search
    print("🧪 Testing index with sample query...")
    test_query = vectors_norm[:1]  # Use first vector as test
    D, I = index.search(test_query, 5)

    print(f"   Top-5 results:")
    for i, (dist, idx) in enumerate(zip(D[0], I[0])):
        print(f"     {i+1}. Index {idx}, Distance {dist:.4f}")
    print()

    print("=" * 80)
    print("✅ FAISS INDEX BUILD COMPLETE!")
    print("=" * 80)
    print()
    print(f"Index stats:")
    print(f"  - Total vectors: {index.ntotal:,}")
    print(f"  - nlist (clusters): {nlist}")
    print(f"  - nprobe (search clusters): {nprobe}")
    print(f"  - Metric: Inner Product (cosine on normalized vectors)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from NPZ file")
    parser.add_argument("--npz", required=True, help="Input NPZ file with vectors")
    parser.add_argument("--output", required=True, help="Output FAISS index file")
    parser.add_argument("--nlist", type=int, default=2048, help="Number of clusters (default: 2048)")
    parser.add_argument("--nprobe", type=int, default=32, help="Number of clusters to search (default: 32)")

    args = parser.parse_args()

    build_faiss_index(args.npz, args.output, args.nlist, args.nprobe)


if __name__ == "__main__":
    main()
