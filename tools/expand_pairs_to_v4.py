#!/usr/bin/env python3
"""
Expand v3 pairs to v4 (50k-100k pairs)

Reuses the same TMD training sequences but with:
- Lower stride (32 instead of 50)
- Higher max-per-sequence cap (30 instead of 15)
- Better deduplication
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys


def cosine_sim(a, b):
    """Fast cosine similarity"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def load_tmd_training_sequences():
    """
    Load TMD training sequences from artifacts.

    This is the same source used for v3, but we'll extract more pairs.
    """
    # Use the same source as v3
    path = 'artifacts/lvm/data_phase3_tmd/training_sequences_ctx100.npz'

    if not Path(path).exists():
        raise FileNotFoundError(f"TMD training sequences not found: {path}")

    print(f"  Loading from: {path}")
    data = np.load(path, allow_pickle=True)

    # Phase3 TMD format: (N, long_seq_len, 768)
    if 'context_sequences' in data:
        sequences = data['context_sequences']
        print(f"  Found {len(sequences)} sequences")
        print(f"  Each sequence length: {sequences.shape[1]} vectors")
        return sequences
    else:
        raise ValueError(f"Unknown format. Keys: {list(data.keys())}")


def generate_pairs_with_controls(sequences, context_len=100, stride=32,
                                  max_per_seq=30, dedup_threshold=0.999):
    """
    Generate pairs from sequences with aggressive extraction.

    Args:
        sequences: List of (T, 768) arrays
        context_len: Context window length
        stride: Sliding window stride (lower = more pairs)
        max_per_seq: Max pairs per sequence
        dedup_threshold: Cosine threshold for dedup
    """
    X_pairs = []
    Y_targets = []
    dedup_window = []
    window_size = 2000

    for seq_idx, seq in enumerate(tqdm(sequences, desc="Generating")):
        if isinstance(seq, np.ndarray):
            vectors = seq
        else:
            vectors = seq  # Already an array

        T = len(vectors)

        if T < context_len + 1:
            continue

        seq_count = 0

        # Slide window with specified stride
        for i in range(0, T - context_len, stride):
            if seq_count >= max_per_seq:
                break

            context = vectors[i:i+context_len]  # (100, 768)
            target = vectors[i+context_len]     # (768,)

            # Dedup check
            is_dup = False
            for prev_target in dedup_window[-window_size:]:
                if cosine_sim(target, prev_target) > dedup_threshold:
                    is_dup = True
                    break

            if is_dup:
                continue

            X_pairs.append(context)
            Y_targets.append(target)
            dedup_window.append(target)
            seq_count += 1

    return np.array(X_pairs, dtype=np.float32), np.array(Y_targets, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Expand v3 pairs to v4")
    parser.add_argument('--out', type=str, default='artifacts/twotower/pairs_v4_synth.npz',
                        help='Output path')
    parser.add_argument('--context-len', type=int, default=100)
    parser.add_argument('--stride', type=int, default=32,
                        help='Lower stride = more pairs (v3 used 50)')
    parser.add_argument('--max-per-seq', type=int, default=30,
                        help='Higher cap = more pairs (v3 used 15)')
    parser.add_argument('--target-pairs', type=int, default=50000,
                        help='Target number of training pairs')
    parser.add_argument('--train-split', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("============================================================")
    print("EXPAND V3 → V4 PAIRS")
    print("============================================================")
    print(f"Target pairs: {args.target_pairs:,}")
    print(f"Context length: {args.context_len}")
    print(f"Stride: {args.stride} (v3 used 50)")
    print(f"Max per sequence: {args.max_per_seq} (v3 used 15)")
    print()

    # Load sequences
    print("Loading TMD training sequences...")
    sequences = load_tmd_training_sequences()
    print(f"  Found {len(sequences)} sequences")

    # Generate pairs
    print(f"\nGenerating pairs...")
    X, Y = generate_pairs_with_controls(
        sequences,
        context_len=args.context_len,
        stride=args.stride,
        max_per_seq=args.max_per_seq
    )

    print(f"  Generated: {len(X):,} pairs")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")

    # Check if we need another pass with even lower stride
    if len(X) < args.target_pairs:
        shortage = args.target_pairs - len(X)
        print(f"\n⚠️  Short by {shortage:,} pairs!")
        print(f"  Consider:")
        print(f"    - Lower stride (try {args.stride - 10})")
        print(f"    - Higher max-per-seq (try {args.max_per_seq + 10})")
        print(f"    - Ingest more Wikipedia articles")
        print(f"\n  Proceeding with {len(X):,} pairs...")
    else:
        print(f"  ✓ Target reached!")

    # Train/val split
    print(f"\nSplitting train/val ({args.train_split:.0%}/{1-args.train_split:.0%})...")
    n = len(X)
    indices = np.random.permutation(n)
    n_train = int(n * args.train_split)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    print(f"  Train: {len(X_train):,} pairs")
    print(f"  Val: {len(X_val):,} pairs")

    # Save
    print(f"\nSaving to: {args.out}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.out,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val
    )

    size_gb = Path(args.out).stat().st_size / 1e9
    print(f"  Size: {size_gb:.2f} GB")

    print("\n============================================================")
    print("EXPANSION COMPLETE")
    print("============================================================")
    print(f"v3 had: 18,109 training pairs")
    print(f"v4 has: {len(X_train):,} training pairs")
    print(f"Increase: {len(X_train)/18109:.1f}x")
    print()
    print("Next: Train v4 with curriculum learning")
    print(f"  python tools/train_twotower_v4.py --pairs {args.out}")


if __name__ == '__main__':
    main()
