#!/usr/bin/env python3
"""
v4 Synthetic Pair Generator

Improvements over v3:
- Per-article caps (--max-per-article) to prevent long-article dominance
- Rolling deduplication window to avoid near-duplicates
- Lane-aware balancing (if lanes present)
- Sharding support for large datasets
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json


def cosine_similarity(a, b):
    """Compute cosine similarity between vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def generate_pairs_from_sequences(sequences, context_len=100, stride=50,
                                   max_per_sequence=20, dedup_threshold=0.999,
                                   dedup_window=2000):
    """
    Generate (context, target) pairs from sequences with controls.

    Args:
        sequences: List of (seq_vectors, metadata) tuples
        context_len: Context window length
        stride: Sliding window stride
        max_per_sequence: Maximum pairs per sequence (cap)
        dedup_threshold: Cosine threshold for duplicate detection
        dedup_window: Size of rolling window for dedup
    """
    pairs = []
    metadata = []
    recent_targets = []  # Rolling window for dedup

    for seq_idx, (seq_vecs, seq_meta) in enumerate(tqdm(sequences, desc="Generating pairs")):
        T, D = seq_vecs.shape

        if T < context_len + 1:
            continue

        seq_pairs = []

        # Slide window
        for i in range(0, T - context_len, stride):
            if len(seq_pairs) >= max_per_sequence:
                break

            context = seq_vecs[i:i+context_len]  # (context_len, D)
            target = seq_vecs[i+context_len]     # (D,)

            # Dedup check against recent targets
            is_dup = False
            for recent_target in recent_targets[-dedup_window:]:
                if cosine_similarity(target, recent_target) > dedup_threshold:
                    is_dup = True
                    break

            if is_dup:
                continue

            seq_pairs.append((context, target))
            recent_targets.append(target)

        # Add to global collection
        pairs.extend(seq_pairs)
        metadata.extend([seq_meta] * len(seq_pairs))

    return pairs, metadata


def balance_lanes(pairs, metadata, tolerance=2.0):
    """
    Balance pairs across lanes (if lane info present).

    Args:
        pairs: List of (context, target) tuples
        metadata: List of metadata dicts (should have 'lane' key if applicable)
        tolerance: Max ratio between largest and smallest lane

    Returns:
        Balanced pairs and metadata
    """
    # Check if lanes present
    if not metadata or 'lane' not in metadata[0]:
        print("  No lane info found, skipping balance")
        return pairs, metadata

    # Count by lane
    lane_counts = {}
    lane_indices = {}
    for i, meta in enumerate(metadata):
        lane = meta.get('lane', 'unknown')
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
        if lane not in lane_indices:
            lane_indices[lane] = []
        lane_indices[lane].append(i)

    print(f"\n  Lane distribution before balancing:")
    for lane, count in sorted(lane_counts.items()):
        print(f"    {lane}: {count}")

    # Find min count
    min_count = min(lane_counts.values())
    max_allowed = int(min_count * tolerance)

    # Subsample each lane
    balanced_indices = []
    for lane, indices in lane_indices.items():
        if len(indices) <= max_allowed:
            balanced_indices.extend(indices)
        else:
            # Random sample
            sampled = np.random.choice(indices, size=max_allowed, replace=False)
            balanced_indices.extend(sampled.tolist())

    # Shuffle
    np.random.shuffle(balanced_indices)

    # Extract balanced pairs
    balanced_pairs = [pairs[i] for i in balanced_indices]
    balanced_metadata = [metadata[i] for i in balanced_indices]

    # Report
    lane_counts_after = {}
    for meta in balanced_metadata:
        lane = meta.get('lane', 'unknown')
        lane_counts_after[lane] = lane_counts_after.get(lane, 0) + 1

    print(f"\n  Lane distribution after balancing:")
    for lane, count in sorted(lane_counts_after.items()):
        print(f"    {lane}: {count}")

    return balanced_pairs, balanced_metadata


def main():
    parser = argparse.ArgumentParser(description="Generate v4 synthetic pairs")
    parser.add_argument('--sequences', type=str, required=True,
                        help='Path to sequences NPZ (with vectors and metadata)')
    parser.add_argument('--out', type=str, required=True,
                        help='Output NPZ path (or shard pattern like "shard_{:02d}.npz")')
    parser.add_argument('--context-len', type=int, default=100,
                        help='Context length')
    parser.add_argument('--stride', type=int, default=50,
                        help='Sliding window stride')
    parser.add_argument('--max-per-sequence', type=int, default=20,
                        help='Maximum pairs per sequence (prevents dominance)')
    parser.add_argument('--dedup-threshold', type=float, default=0.999,
                        help='Cosine threshold for duplicate detection')
    parser.add_argument('--dedup-window', type=int, default=2000,
                        help='Rolling window size for dedup')
    parser.add_argument('--balance-lanes', action='store_true',
                        help='Balance across lanes (if metadata has lane info)')
    parser.add_argument('--train-split', type=float, default=0.9,
                        help='Train/val split ratio')
    parser.add_argument('--shard-size', type=int, default=None,
                        help='If set, split output into shards of this size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("============================================================")
    print("V4 SYNTHETIC PAIR GENERATOR")
    print("============================================================")
    print(f"Sequences: {args.sequences}")
    print(f"Context length: {args.context_len}")
    print(f"Stride: {args.stride}")
    print(f"Max per sequence: {args.max_per_sequence}")
    print(f"Dedup threshold: {args.dedup_threshold}")
    print(f"Dedup window: {args.dedup_window}")
    print(f"Balance lanes: {args.balance_lanes}")
    print(f"Train/val split: {args.train_split}")
    print()

    # Load sequences
    print("Loading sequences...")
    data = np.load(args.sequences, allow_pickle=True)

    # Expecting: 'sequences' array where each element is (vectors, metadata)
    # For now, assume we have 'training_sequences' from Phase 3
    # Format: List of (T, 768) arrays with optional metadata

    # TODO: Adapt to actual sequence format
    # For now, create simple sequences from existing training data
    if 'sequences' in data:
        sequences = data['sequences']
    else:
        # Fallback: use existing training chains
        print("  ⚠️  No 'sequences' key found. Looking for alternative format...")
        # Try to reconstruct from existing data
        raise NotImplementedError("Need to implement sequence loading from your specific format")

    print(f"  Loaded {len(sequences)} sequences")

    # Generate pairs
    print("\nGenerating pairs...")
    pairs, metadata = generate_pairs_from_sequences(
        sequences,
        context_len=args.context_len,
        stride=args.stride,
        max_per_sequence=args.max_per_sequence,
        dedup_threshold=args.dedup_threshold,
        dedup_window=args.dedup_window
    )
    print(f"  Generated {len(pairs)} pairs")

    # Balance lanes if requested
    if args.balance_lanes:
        print("\nBalancing lanes...")
        pairs, metadata = balance_lanes(pairs, metadata, tolerance=2.0)
        print(f"  Balanced to {len(pairs)} pairs")

    # Convert to arrays
    print("\nConverting to arrays...")
    X = np.array([p[0] for p in pairs], dtype=np.float32)  # (N, context_len, 768)
    Y = np.array([p[1] for p in pairs], dtype=np.float32)  # (N, 768)
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")

    # Train/val split
    print("\nSplitting train/val...")
    n_train = int(len(pairs) * args.train_split)
    indices = np.random.permutation(len(pairs))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    print(f"  Train: {len(X_train)} pairs")
    print(f"  Val: {len(X_val)} pairs")

    # Save
    print("\nSaving...")
    if args.shard_size:
        # Shard output
        n_shards = (len(X_train) + args.shard_size - 1) // args.shard_size
        print(f"  Sharding into {n_shards} files...")

        for shard_idx in range(n_shards):
            start = shard_idx * args.shard_size
            end = min((shard_idx + 1) * args.shard_size, len(X_train))

            shard_path = args.out.replace('{shard}', f'{shard_idx:02d}')
            np.savez_compressed(
                shard_path,
                X_train=X_train[start:end],
                Y_train=Y_train[start:end],
                X_val=X_val if shard_idx == 0 else np.array([]),  # Val only in first shard
                Y_val=Y_val if shard_idx == 0 else np.array([])
            )
            print(f"    Shard {shard_idx}: {shard_path} ({end-start} pairs)")
    else:
        # Single file
        np.savez_compressed(
            args.out,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val
        )
        print(f"  Saved to: {args.out}")
        print(f"  Size: {Path(args.out).stat().st_size / 1e9:.2f} GB")

    print("\n============================================================")
    print("GENERATION COMPLETE")
    print("============================================================")
    print(f"Total pairs: {len(pairs)}")
    print(f"Train: {len(X_train)}")
    print(f"Val: {len(X_val)}")


if __name__ == '__main__':
    main()
