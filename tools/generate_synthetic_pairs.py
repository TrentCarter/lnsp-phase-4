#!/usr/bin/env python3
"""
Generate synthetic training pairs from Phase-3 sequences using sliding windows.

Strategy:
- Use existing Phase-3 sequences (1,386 seqs × 1000 vectors)
- Extract overlapping (context, target) pairs with stride
- De-duplicate near-identical targets (cos > 0.999)
- Shuffle and split (90/10)

Expected output: 15k-20k pairs

Usage:
    python tools/generate_synthetic_pairs.py \
      --input artifacts/lvm/data_phase3_tmd/training_sequences_ctx100.npz \
      --out artifacts/twotower/pairs_v3_synth.npz \
      --context 100 --stride 50 --max-per-seq 15
"""

import argparse
import json
import os
import numpy as np
from numpy.linalg import norm
from pathlib import Path
from tqdm import tqdm


def cos_sim(a, b):
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))


def extract_pairs_from_sequence(
    seq,
    context_len=100,
    stride=50,
    max_pairs=15
):
    """
    Extract (context, target) pairs from a single sequence using sliding window.

    Args:
        seq: (T, 768) - Full sequence
        context_len: Number of vectors in context (default: 100)
        stride: Stride for sliding window (default: 50 = 50% overlap)
        max_pairs: Maximum pairs to extract per sequence (default: 15)

    Returns:
        pairs: List of (context, target) tuples
    """
    T = len(seq)
    pairs = []

    # Need at least context_len + 1 vectors (context + target)
    if T <= context_len:
        return pairs

    # Sliding window: i to i+context_len → target at i+context_len
    for i in range(0, T - context_len, stride):
        context = seq[i:i+context_len]  # (100, 768)
        target = seq[i+context_len]     # (768,)

        pairs.append((context, target))

        # Cap pairs per sequence
        if len(pairs) >= max_pairs:
            break

    return pairs


def deduplicate_targets(pairs, threshold=0.999, window=1000):
    """
    Remove pairs with near-duplicate targets (cos > threshold).

    Uses a sliding window to avoid O(N^2) comparisons.

    Args:
        pairs: List of (context, target) tuples
        threshold: Cosine similarity threshold (default: 0.999)
        window: Size of sliding window for dedup (default: 1000)

    Returns:
        deduped_pairs: List of unique pairs
    """
    if not pairs:
        return []

    kept_pairs = []
    kept_targets = []

    for ctx, tgt in tqdm(pairs, desc="  Deduplicating", leave=False):
        # Check against recent kept targets (sliding window)
        is_duplicate = False
        check_start = max(0, len(kept_targets) - window)

        for prev_tgt in kept_targets[check_start:]:
            if cos_sim(tgt, prev_tgt) > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_pairs.append((ctx, tgt))
            kept_targets.append(tgt)

    return kept_pairs


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training pairs")
    parser.add_argument('--input', required=True, help='Phase-3 sequences NPZ')
    parser.add_argument('--out', required=True, help='Output pairs NPZ')
    parser.add_argument('--context', type=int, default=100, help='Context length')
    parser.add_argument('--stride', type=int, default=50, help='Sliding window stride')
    parser.add_argument('--max-per-seq', type=int, default=15, help='Max pairs per sequence')
    parser.add_argument('--train-split', type=float, default=0.9, help='Train/val split')
    parser.add_argument('--dedup-threshold', type=float, default=0.999, help='Dedup cosine threshold')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    print("="*60)
    print("SYNTHETIC PAIR GENERATION")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Context length: {args.context}")
    print(f"Stride: {args.stride}")
    print(f"Max pairs/seq: {args.max_per_seq}")
    print(f"Dedup threshold: {args.dedup_threshold}")
    print("="*60)

    # Load Phase-3 sequences
    print("\nLoading sequences...")
    data = np.load(args.input, allow_pickle=True)

    # Handle different formats
    if 'context_sequences' in data:
        # Phase-3 TMD format: (N, 1000, 768)
        sequences = data['context_sequences']
        print(f"  Found {len(sequences):,} sequences (Phase-3 TMD format)")
    elif 'train_seqs' in data:
        # Alternative format
        sequences = data['train_seqs']
        print(f"  Found {len(sequences):,} sequences")
    else:
        raise ValueError(f"Unknown format. Keys: {list(data.keys())}")

    print(f"  Sequence shape: {sequences.shape}")

    # Extract pairs from all sequences
    print("\nExtracting pairs...")
    all_pairs = []

    for seq in tqdm(sequences, desc="  Processing seqs"):
        seq_pairs = extract_pairs_from_sequence(
            seq,
            context_len=args.context,
            stride=args.stride,
            max_pairs=args.max_per_seq
        )
        all_pairs.extend(seq_pairs)

    print(f"  Total pairs extracted: {len(all_pairs):,}")

    # Deduplicate
    print("\nDeduplicating near-identical targets...")
    unique_pairs = deduplicate_targets(
        all_pairs,
        threshold=args.dedup_threshold,
        window=1000
    )

    removed = len(all_pairs) - len(unique_pairs)
    print(f"  Removed {removed:,} duplicates ({100*removed/len(all_pairs):.1f}%)")
    print(f"  Unique pairs: {len(unique_pairs):,}")

    # Convert to arrays
    print("\nConverting to arrays...")
    X = np.stack([ctx for ctx, _ in unique_pairs], axis=0).astype(np.float32)
    Y = np.stack([tgt for _, tgt in unique_pairs], axis=0).astype(np.float32)

    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")

    # Shuffle and split
    print("\nShuffling and splitting...")
    n_total = len(X)
    n_train = int(n_total * args.train_split)

    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_val = X[val_idx]
    Y_val = Y[val_idx]

    print(f"  Train: {len(X_train):,} pairs")
    print(f"  Val:   {len(X_val):,} pairs")

    # Save
    print("\nSaving...")
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    np.savez_compressed(
        out_path,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val
    )

    # Save manifest
    manifest = {
        'total_pairs': int(n_total),
        'train_pairs': int(len(X_train)),
        'val_pairs': int(len(X_val)),
        'context_len': int(args.context),
        'stride': int(args.stride),
        'max_per_seq': int(args.max_per_seq),
        'dedup_threshold': float(args.dedup_threshold),
        'train_split': float(args.train_split),
        'seed': int(args.seed),
        'input_file': str(args.input),
        'output_file': str(args.out)
    }

    manifest_path = out_path.parent / f"{out_path.stem}_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"  Pairs: {out_path}")
    print(f"  Manifest: {manifest_path}")

    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(json.dumps({
        'total_pairs': int(n_total),
        'train_pairs': int(len(X_train)),
        'val_pairs': int(len(X_val)),
        'removed_duplicates': int(removed)
    }, indent=2))
    print("="*60)

    # Success check
    if n_total >= 15000:
        print(f"\n✓ SUCCESS: Generated {n_total:,} pairs (≥15k threshold)")
    elif n_total >= 10000:
        print(f"\n⚠️  Generated {n_total:,} pairs (10-15k range, acceptable)")
    else:
        print(f"\n❌ WARNING: Only {n_total:,} pairs (<10k, may be insufficient)")


if __name__ == '__main__':
    main()
