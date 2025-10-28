#!/usr/bin/env python3
"""
Build a proper eval set with DISJOINT articles from training.

This fixes the data leak where 74% of eval chunks overlapped with training.

Strategy:
1. Load payload and identify all article indices
2. Load training data to get training article set
3. Select eval articles that DON'T appear in training
4. Extract sequences from these held-out articles
5. Target: 5k sequences from 100+ held-out articles

Usage:
    python tools/build_disjoint_eval.py \
        --payload artifacts/wikipedia_584k_payload.npy \
        --train-npz artifacts/lvm/train_payload_aligned.npz \
        --out artifacts/lvm/eval_v3_disjoint.npz \
        --n-sequences 5244 \
        --context-length 5
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Build disjoint eval set")
    ap.add_argument("--payload", type=Path, required=True)
    ap.add_argument("--train-npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-sequences", type=int, default=5244)
    ap.add_argument("--context-length", type=int, default=5)
    ap.add_argument("--min-article-chunks", type=int, default=10)
    args = ap.parse_args()

    print("=" * 80)
    print("BUILDING DISJOINT EVAL SET")
    print("=" * 80)
    print(f"Payload: {args.payload}")
    print(f"Train data: {args.train_npz}")
    print(f"Target sequences: {args.n_sequences}")
    print(f"Context length: {args.context_length}")
    print("=" * 80)
    print()

    # Load training data to get training article set
    print("Loading training data...")
    train_data = np.load(args.train_npz, allow_pickle=True)
    train_keys = train_data['truth_keys']
    train_articles = set(int(k[0]) for k in train_keys)

    print(f"  Training articles: {len(train_articles)}")
    print(f"  Training sequences: {len(train_keys)}")
    print()

    # Load payload
    print("Loading payload...")
    payload = np.load(args.payload, allow_pickle=True).item()
    print(f"  Payload vectors: {len(payload)}")
    print()

    # Group payload by article
    print("Grouping payload by article...")
    articles = defaultdict(list)
    for payload_id, (text, meta, vec) in payload.items():
        art_idx = meta['article_index']
        chunk_idx = meta['chunk_index']
        articles[art_idx].append((chunk_idx, payload_id, vec))

    # Sort chunks within each article
    for art_idx in articles:
        articles[art_idx].sort(key=lambda x: x[0])

    print(f"  Total articles in payload: {len(articles)}")
    print()

    # Find held-out articles (NOT in training)
    print("Finding held-out articles...")
    heldout_articles = []

    for art_idx, chunks in articles.items():
        if art_idx not in train_articles and len(chunks) >= args.min_article_chunks:
            heldout_articles.append(art_idx)

    heldout_articles.sort()

    print(f"  Held-out articles: {len(heldout_articles)}")
    print(f"  (articles with ≥{args.min_article_chunks} chunks, not in training)")
    print()

    if len(heldout_articles) == 0:
        print("❌ ERROR: No held-out articles found!")
        print("   All articles in payload are in training set")
        print("   Need to ingest more Wikipedia articles for eval")
        return

    # Extract sequences from held-out articles
    print("Extracting sequences from held-out articles...")

    contexts = []
    targets = []
    truth_keys = []

    for art_idx in heldout_articles:
        chunks = articles[art_idx]

        for i in range(len(chunks) - args.context_length):
            # Check for continuity
            chunk_indices = [chunks[j][0] for j in range(i, i + args.context_length + 1)]
            expected_indices = list(range(chunk_indices[0], chunk_indices[0] + args.context_length + 1))

            if chunk_indices != expected_indices:
                continue  # Skip if not continuous

            # Extract context and target
            context_vecs = [chunks[j][2] for j in range(i, i + args.context_length)]
            target_chunk_idx, target_payload_id, target_vec = chunks[i + args.context_length]

            contexts.append(np.stack(context_vecs, axis=0))
            targets.append(target_vec)
            truth_keys.append([art_idx, target_chunk_idx])

            # Stop if we have enough
            if len(contexts) >= args.n_sequences:
                break

        if len(contexts) >= args.n_sequences:
            break

    contexts = np.stack(contexts[:args.n_sequences], axis=0)
    targets = np.stack(targets[:args.n_sequences], axis=0)
    truth_keys = np.array(truth_keys[:args.n_sequences])

    print(f"  Extracted: {len(contexts)} sequences")
    print(f"  Context shape: {contexts.shape}")
    print(f"  Target shape: {targets.shape}")
    print(f"  Truth keys shape: {truth_keys.shape}")
    print()

    # Verify disjointness
    eval_articles = set(int(k[0]) for k in truth_keys)
    overlap = eval_articles & train_articles

    print("Verifying disjointness...")
    print(f"  Eval articles: {len(eval_articles)}")
    print(f"  Train articles: {len(train_articles)}")
    print(f"  Overlap: {len(overlap)}")

    if overlap:
        print(f"  ❌ ERROR: {len(overlap)} articles still overlap!")
        return
    else:
        print(f"  ✅ PERFECT: Completely disjoint!")

    # Check chunk disjointness
    eval_chunks = set((int(k[0]), int(k[1])) for k in truth_keys)
    train_chunks = set((int(k[0]), int(k[1])) for k in train_keys)
    chunk_overlap = eval_chunks & train_chunks

    print(f"  Eval chunks: {len(eval_chunks)}")
    print(f"  Train chunks: {len(train_chunks)}")
    print(f"  Chunk overlap: {len(chunk_overlap)}")

    if chunk_overlap:
        print(f"  ❌ ERROR: {len(chunk_overlap)} chunks overlap!")
        return
    else:
        print(f"  ✅ PERFECT: No chunk overlap!")

    print()

    # Provenance
    provenance = {
        'embedder_id': 'GTR-T5-base-768',
        'norm': 'l2_once',
        'metric': 'ip',
        'payload_build_id': 'payload584k_2025-10-24@sha256:12cfd8d7d92dca99',
        'source': 'wikipedia_disjoint_eval_v3',
        'context_length': args.context_length,
        'min_article_chunks': args.min_article_chunks,
        'train_articles': len(train_articles),
        'eval_articles': len(eval_articles),
        'article_overlap': 0,
        'chunk_overlap': 0,
    }

    # Save
    print("Saving...")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.out,
        contexts=contexts,
        targets=targets,
        truth_keys=truth_keys,
        provenance=np.array([json.dumps(provenance)], dtype=object),
    )

    print(f"  ✅ Saved to: {args.out}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Created disjoint eval set with {len(contexts)} sequences")
    print(f"✅ From {len(eval_articles)} held-out articles (not in training)")
    print(f"✅ Zero article overlap (was 91.5% before)")
    print(f"✅ Zero chunk overlap (was 74% before)")
    print()
    print("This eval set tests REAL generalization to unseen articles!")
    print("=" * 80)


if __name__ == "__main__":
    main()
