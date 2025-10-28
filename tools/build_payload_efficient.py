#!/usr/bin/env python3
"""
Efficient Payload Builder
=========================

Builds payload dictionary efficiently by loading arrays once
and using list comprehension (much faster than loop indexing).
"""

import numpy as np
from pathlib import Path


def main():
    print("=" * 80)
    print("EFFICIENT PAYLOAD BUILDER")
    print("=" * 80)
    print()

    print("Loading Wikipedia vectors...")
    data = np.load("artifacts/wikipedia_584k_fresh.npz", allow_pickle=True)

    # Convert ALL arrays to lists ONCE (bulk operation - fast!)
    print("Converting arrays to lists (bulk operation)...")
    texts = list(data['concept_texts'])
    vectors = list(data['vectors'])
    article_indices = list(data['article_indices'])
    chunk_indices = list(data['chunk_indices'])
    cpe_ids = list(data['cpe_ids'])

    print(f"✅ Loaded {len(texts):,} entries\n")

    # Build payload using zip (much faster than indexing!)
    print("Building payload dictionary...")
    payload = {}

    for idx, (text, vec, art_idx, chunk_idx, cpe_id) in enumerate(
        zip(texts, vectors, article_indices, chunk_indices, cpe_ids)
    ):
        payload[idx] = (
            text,
            {
                "article_index": int(art_idx),
                "chunk_index": int(chunk_idx),
                "cpe_id": str(cpe_id)
            },
            vec
        )

        if (idx + 1) % 100000 == 0:
            print(f"  Progress: {idx + 1:,} / {len(texts):,}")

    print(f"\n✅ Payload built with {len(payload):,} entries\n")

    # Save
    print("Saving payload...")
    Path("artifacts").mkdir(exist_ok=True)
    np.save("artifacts/wikipedia_584k_payload.npy", payload, allow_pickle=True)

    file_size = Path("artifacts/wikipedia_584k_payload.npy").stat().st_size / (1024**3)
    print(f"✅ Saved payload ({file_size:.2f} GB)\n")

    print("=" * 80)
    print("✅ PAYLOAD READY!")
    print("=" * 80)
    print("\nNow you can run evaluations:")
    print("  python tools/eval_retrieval_v2.py --npz ... --payload artifacts/wikipedia_584k_payload.npy ...")
    print()


if __name__ == "__main__":
    main()
