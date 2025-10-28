#!/usr/bin/env python3
"""
Build Per-Article Shard Indexes
================================

Creates small FAISS indexes for each article to enable fast local search
for continuation queries.

Outputs: artifacts/article_shards.pkl (dict of article_index -> local data)
"""

import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import faiss


def main():
    print("=" * 80)
    print("BUILDING PER-ARTICLE SHARD INDEXES")
    print("=" * 80)
    print()

    # Load payload
    print("Loading payload...")
    payload_path = Path("artifacts/wikipedia_584k_payload.npy")
    payload = np.load(payload_path, allow_pickle=True).item()
    print(f"  Loaded {len(payload):,} entries\n")

    # Group by article
    print("Grouping chunks by article...")
    article_chunks = defaultdict(list)

    for idx, (text, meta, vec) in payload.items():
        article_idx = int(meta["article_index"])
        chunk_idx = int(meta["chunk_index"])
        article_chunks[article_idx].append((chunk_idx, idx, text, meta, vec))

    print(f"  Found {len(article_chunks):,} unique articles\n")

    # Build shard for each article
    print("Building per-article shards...")
    article_shards = {}

    chunk_counts = []
    for article_idx, chunks in sorted(article_chunks.items()):
        chunk_counts.append(len(chunks))

        # Sort by chunk index for locality
        chunks.sort(key=lambda x: x[0])

        # Extract vectors
        vectors = np.stack([c[4] for c in chunks], axis=0).astype(np.float32)

        # Build small flat index (exact search, fast for <1000 vectors)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        # Build ID-to-payload mapping for this article
        local_payload = {}
        for i, (chunk_idx, global_idx, text, meta, vec) in enumerate(chunks):
            local_payload[i] = (text, meta, vec, global_idx)

        article_shards[article_idx] = {
            "index": index,
            "payload": local_payload,
            "n_chunks": len(chunks),
        }

        if (len(article_shards) % 1000 == 0):
            print(f"  Progress: {len(article_shards):,} / {len(article_chunks):,} articles")

    print(f"\n✓ Built {len(article_shards):,} article shards\n")

    # Statistics
    chunk_counts = np.array(chunk_counts)
    print("Shard statistics:")
    print(f"  Chunks per article:")
    print(f"    Min:    {chunk_counts.min():,}")
    print(f"    P50:    {int(np.percentile(chunk_counts, 50)):,}")
    print(f"    P95:    {int(np.percentile(chunk_counts, 95)):,}")
    print(f"    Max:    {chunk_counts.max():,}")
    print(f"    Mean:   {chunk_counts.mean():.1f}")
    print()

    # Save shards
    output_path = Path("artifacts/article_shards.pkl")
    print(f"Saving to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(article_shards, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = output_path.stat().st_size / (1024**2)
    print(f"✓ Saved {file_size:.1f} MB\n")

    print("=" * 80)
    print("✓ ARTICLE SHARDS READY!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
