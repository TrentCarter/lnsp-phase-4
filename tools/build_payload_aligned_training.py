#!/usr/bin/env python3
"""
Build Payload-Aligned Training Data
====================================

Extracts training sequences directly from Wikipedia payload to ensure
training targets match FAISS retrieval vectors exactly.

This fixes the root cause of 0% retrieval: previous training data came from
a different source than the payload, resulting in ~0.14 cosine similarity.

Strategy:
1. Load payload (584k Wikipedia chunks with GTR-T5 vectors)
2. Sort by (article_index, chunk_index) for temporal order
3. Extract sliding windows: [t-4, t-3, t-2, t-1, t-0] → predict t+1
4. Preserve truth_keys for traceability
5. Add provenance metadata

Gates:
- All vectors must have norm = 1.0 (L2 normalized)
- All truth_keys must exist in payload (100% coverage)
- Context-to-target cosine should be reasonable (0.3-0.9)
- Train/val split: 80/20

Usage:
    python tools/build_payload_aligned_training.py \
        --payload artifacts/wikipedia_584k_payload.npy \
        --out-train artifacts/lvm/train_payload_aligned.npz \
        --out-val artifacts/lvm/val_payload_aligned.npz \
        --context-length 5 \
        --min-article-chunks 10
"""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def compute_provenance(payload_path: Path, embedder="GTR-T5-base-768"):
    """Compute provenance metadata for reproducibility."""
    with open(payload_path, 'rb') as f:
        payload_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    return {
        "embedder_id": embedder,
        "payload_build_id": f"payload584k_2025-10-24@sha256:{payload_hash}",
        "norm": "l2_once",
        "metric": "ip",
        "created_at": datetime.now().isoformat(),
        "creation_tool": "build_payload_aligned_training.py",
        "data_source": "wikipedia_584k_payload.npy",
    }


def extract_sequences(payload, context_length=5, min_article_chunks=10):
    """
    Extract training sequences from payload.

    Strategy:
    1. Group chunks by article_index
    2. Sort chunks within each article by chunk_index
    3. For each article with ≥ min_article_chunks chunks:
       - Create sliding windows: [t-4, t-3, t-2, t-1, t-0] → target=t+1
       - Skip if any chunk is missing (ensure continuity)

    Args:
        payload: Dict[int, (text, meta, vec)]
        context_length: Number of context vectors (default: 5)
        min_article_chunks: Minimum chunks per article (default: 10)

    Returns:
        contexts: [N, context_length, 768]
        targets: [N, 768]
        truth_keys: [N, 2] - (article_idx, chunk_idx) for target
    """
    print("\n1. Grouping chunks by article...")

    # Group by article
    articles = defaultdict(list)
    for payload_id, (text, meta, vec) in payload.items():
        art_idx = meta['article_index']
        chunk_idx = meta['chunk_index']
        articles[art_idx].append((chunk_idx, payload_id, vec))

    print(f"  Found {len(articles)} articles")

    # Sort chunks within each article
    for art_idx in articles:
        articles[art_idx].sort(key=lambda x: x[0])  # Sort by chunk_index

    print("\n2. Extracting sequences...")

    contexts = []
    targets = []
    truth_keys = []

    articles_used = 0
    sequences_extracted = 0
    articles_skipped_short = 0

    for art_idx, chunks in articles.items():
        # Skip articles with too few chunks
        if len(chunks) < min_article_chunks:
            articles_skipped_short += 1
            continue

        articles_used += 1

        # Extract sliding windows
        for i in range(len(chunks) - context_length):
            # Get context window: [i, i+1, ..., i+context_length-1]
            # Target: i+context_length

            # Check for continuity (no missing chunks)
            chunk_indices = [chunks[j][0] for j in range(i, i + context_length + 1)]
            expected_indices = list(range(chunk_indices[0], chunk_indices[0] + context_length + 1))

            if chunk_indices != expected_indices:
                # Missing chunk, skip this window
                continue

            # Extract context vectors
            context_vecs = [chunks[j][2] for j in range(i, i + context_length)]

            # Extract target vector and its truth key
            target_chunk_idx, target_payload_id, target_vec = chunks[i + context_length]

            contexts.append(np.stack(context_vecs, axis=0))
            targets.append(target_vec)
            truth_keys.append([art_idx, target_chunk_idx])

            sequences_extracted += 1

        if articles_used % 100 == 0:
            print(f"  Progress: {articles_used} articles, {sequences_extracted} sequences")

    print(f"\n  Articles used: {articles_used}")
    print(f"  Articles skipped (too short): {articles_skipped_short}")
    print(f"  Total sequences: {sequences_extracted}")

    return (
        np.array(contexts, dtype=np.float32),
        np.array(targets, dtype=np.float32),
        np.array(truth_keys, dtype=np.int32),
    )


def validate_data(contexts, targets, truth_keys, payload):
    """
    Validate extracted data quality.

    Checks:
    1. All vectors are L2 normalized (norm = 1.0)
    2. All truth_keys exist in payload
    3. Targets match payload vectors exactly
    4. Context-to-target cosines are reasonable (not degenerate)
    """
    print("\n3. Validating data quality...")

    # Check norms
    context_norms = np.linalg.norm(contexts, axis=2)  # [N, context_length]
    target_norms = np.linalg.norm(targets, axis=1)     # [N]

    mean_context_norm = np.mean(context_norms)
    mean_target_norm = np.mean(target_norms)

    print(f"  Context vector norms: mean={mean_context_norm:.6f}, std={np.std(context_norms):.6f}")
    print(f"  Target vector norms: mean={mean_target_norm:.6f}, std={np.std(target_norms):.6f}")

    if not (0.99 < mean_context_norm < 1.01 and 0.99 < mean_target_norm < 1.01):
        print(f"  ⚠️  WARNING: Vectors may not be properly normalized!")

    # Build reverse index
    article_chunk_to_id = {}
    for payload_id, (text, meta, vec) in payload.items():
        key = (meta['article_index'], meta['chunk_index'])
        article_chunk_to_id[key] = payload_id

    # Check truth_keys coverage
    print(f"\n  Checking truth_keys coverage...")
    found = 0
    missing = []

    for i, (art_idx, chunk_idx) in enumerate(truth_keys):
        key = (int(art_idx), int(chunk_idx))
        if key in article_chunk_to_id:
            found += 1
        else:
            missing.append((i, art_idx, chunk_idx))

    coverage = found / len(truth_keys)
    print(f"  Coverage: {found}/{len(truth_keys)} ({100*coverage:.2f}%)")

    if coverage < 1.0:
        print(f"  ❌ FAILED: Coverage {coverage:.4f} < 1.0")
        print(f"  Missing keys (first 10): {missing[:10]}")
        return False

    # Verify targets match payload exactly
    print(f"\n  Verifying targets match payload...")
    cosines = []

    for i in range(min(1000, len(targets))):
        art_idx, chunk_idx = truth_keys[i]
        key = (int(art_idx), int(chunk_idx))
        payload_id = article_chunk_to_id[key]
        _, _, payload_vec = payload[payload_id]

        cos = np.dot(targets[i], payload_vec)
        cosines.append(cos)

    mean_cos = np.mean(cosines)
    print(f"  Mean cosine (target vs payload): {mean_cos:.6f}")

    if mean_cos < 0.999:
        print(f"  ⚠️  WARNING: Targets don't match payload perfectly! ({mean_cos:.6f} < 0.999)")
        return False

    # Check context-to-target similarity (should be reasonable, not degenerate)
    print(f"\n  Checking context-to-target similarity...")
    ctx_to_tgt_cosines = []

    for i in range(min(1000, len(contexts))):
        # Compute cosine between last context vector and target
        last_ctx = contexts[i, -1, :]
        target = targets[i]
        cos = np.dot(last_ctx, target)
        ctx_to_tgt_cosines.append(cos)

    mean_ctx_cos = np.mean(ctx_to_tgt_cosines)
    print(f"  Mean cosine (last_context vs target): {mean_ctx_cos:.4f}")
    print(f"  Range: [{np.min(ctx_to_tgt_cosines):.4f}, {np.max(ctx_to_tgt_cosines):.4f}]")

    if not (0.2 < mean_ctx_cos < 0.9):
        print(f"  ⚠️  WARNING: Context-to-target cosine outside expected range (0.2-0.9)!")

    print(f"\n  ✅ All validation checks passed!")
    return True


def train_val_split(contexts, targets, truth_keys, val_ratio=0.2, seed=42):
    """
    Split data into train and validation sets.

    Strategy: Random shuffle, then split by ratio.
    """
    print(f"\n4. Creating train/val split ({int(100*(1-val_ratio))}%/{int(100*val_ratio)})...")

    n = len(contexts)
    np.random.seed(seed)
    indices = np.random.permutation(n)

    val_size = int(n * val_ratio)
    train_size = n - val_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_contexts = contexts[train_idx]
    train_targets = targets[train_idx]
    train_truth_keys = truth_keys[train_idx]

    val_contexts = contexts[val_idx]
    val_targets = targets[val_idx]
    val_truth_keys = truth_keys[val_idx]

    print(f"  Train: {len(train_contexts)} sequences")
    print(f"  Val: {len(val_contexts)} sequences")

    return (
        (train_contexts, train_targets, train_truth_keys),
        (val_contexts, val_targets, val_truth_keys),
    )


def main():
    ap = argparse.ArgumentParser(description="Build payload-aligned training data")
    ap.add_argument("--payload", type=Path, required=True,
                    help="Path to payload NPY file")
    ap.add_argument("--out-train", type=Path, required=True,
                    help="Output NPZ file for training set")
    ap.add_argument("--out-val", type=Path, required=True,
                    help="Output NPZ file for validation set")
    ap.add_argument("--context-length", type=int, default=5,
                    help="Number of context vectors (default: 5)")
    ap.add_argument("--min-article-chunks", type=int, default=10,
                    help="Minimum chunks per article (default: 10)")
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="Validation set ratio (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for train/val split (default: 42)")

    args = ap.parse_args()

    print("=" * 80)
    print("Building Payload-Aligned Training Data")
    print("=" * 80)
    print(f"Payload: {args.payload}")
    print(f"Context length: {args.context_length}")
    print(f"Min article chunks: {args.min_article_chunks}")
    print(f"Val ratio: {args.val_ratio}")
    print("=" * 80)

    # Load payload
    print("\nLoading payload...")
    payload = np.load(args.payload, allow_pickle=True).item()
    print(f"  Loaded {len(payload)} vectors")

    # Extract sequences
    contexts, targets, truth_keys = extract_sequences(
        payload,
        context_length=args.context_length,
        min_article_chunks=args.min_article_chunks,
    )

    # Validate
    if not validate_data(contexts, targets, truth_keys, payload):
        print("\n❌ Validation failed! Aborting.")
        return 1

    # Train/val split
    (train_ctx, train_tgt, train_keys), (val_ctx, val_tgt, val_keys) = train_val_split(
        contexts, targets, truth_keys,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Compute provenance
    print("\n5. Computing provenance...")
    provenance = compute_provenance(args.payload)
    for k, v in provenance.items():
        print(f"  {k}: {v}")

    # Save training set
    print(f"\n6. Saving training set to: {args.out_train}")
    args.out_train.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.out_train,
        context_sequences=train_ctx,
        target_vectors=train_tgt,
        truth_keys=train_keys,
        context_length=args.context_length,
        num_sequences=len(train_ctx),
        vector_dim=768,
        provenance=np.array([json.dumps(provenance)]),
    )

    print(f"  Saved {len(train_ctx)} training sequences")

    # Save validation set
    print(f"\n7. Saving validation set to: {args.out_val}")
    np.savez(
        args.out_val,
        context_sequences=val_ctx,
        target_vectors=val_tgt,
        truth_keys=val_keys,
        context_length=args.context_length,
        num_sequences=len(val_ctx),
        vector_dim=768,
        provenance=np.array([json.dumps(provenance)]),
    )

    print(f"  Saved {len(val_ctx)} validation sequences")

    # Summary
    print("\n" + "=" * 80)
    print("✅ SUCCESS! Payload-aligned training data created")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Training sequences: {len(train_ctx)}")
    print(f"  Validation sequences: {len(val_ctx)}")
    print(f"  Total: {len(contexts)}")
    print(f"  Context length: {args.context_length}")
    print(f"  Vector dim: 768")
    print(f"\nFiles:")
    print(f"  Train: {args.out_train}")
    print(f"  Val: {args.out_val}")
    print(f"\nNext steps:")
    print(f"  1. Train models using {args.out_train}")
    print(f"  2. Validate on {args.out_val}")
    print(f"  3. Evaluate with aligned eval data")
    print(f"  4. Expect Contain@50: 60-75%, R@5: 40-55%")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
