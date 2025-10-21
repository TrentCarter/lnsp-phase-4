#!/usr/bin/env python3
"""
Build training pairs for two-tower retriever.

Extracts (context, target) pairs from Phase-3 sequence data:
- Context: Last N vectors from a sequence
- Target: The next vector after context

Usage:
    python tools/build_twotower_pairs.py \
      --inputs artifacts/lvm/data_phase3_tmd/validation_sequences_ctx100.npz \
      --out artifacts/twotower/pairs_v1.npz \
      --train-split 0.90 \
      --context-len 100 \
      --target-offset 1
"""

import argparse
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple


def load_sequences(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load sequences from NPZ file.
    
    Returns:
        context_sequences: (N, seq_len, 768) - Context vectors
        target_vectors: (N, 768) - Target vectors (what comes next)
    """
    print(f"Loading sequences from: {path}")
    data = np.load(path, allow_pickle=True)
    
    # Phase-3 TMD format
    if 'context_sequences' in data:
        contexts = data['context_sequences']
        targets = data['target_vectors']
        print(f"  Found Phase-3 TMD format: {contexts.shape[0]} sequences")
        return contexts, targets
    
    # Fallback: older formats
    elif 'val_seqs' in data:
        contexts = data['val_seqs']
        targets = data['val_next']
        print(f"  Found older format: {contexts.shape[0]} sequences")
        return contexts, targets
    
    else:
        raise ValueError(f"Unknown NPZ format. Keys: {list(data.keys())}")


def extract_pairs(
    contexts: np.ndarray,
    targets: np.ndarray,
    context_len: int = 100,
    target_offset: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (context, target) pairs from sequences.
    
    Args:
        contexts: (N, full_seq_len, 768)
        targets: (N, 768)
        context_len: How many vectors to use as context
        target_offset: Offset from end of context to target (usually 1)
    
    Returns:
        X: (N, context_len, 768) - Context sequences
        Y: (N, 768) - Target vectors
    """
    N, full_len, D = contexts.shape
    
    print(f"\nExtracting pairs:")
    print(f"  Full sequence length: {full_len}")
    print(f"  Using last {context_len} as context")
    print(f"  Target offset: {target_offset}")
    
    # Use last context_len vectors as context
    if full_len >= context_len:
        X = contexts[:, -context_len:, :]  # Last N vectors
    else:
        # Pad if sequence is shorter than context_len
        print(f"  Warning: Sequences shorter than context_len, padding...")
        X = np.zeros((N, context_len, D), dtype=np.float32)
        X[:, -full_len:, :] = contexts
    
    # Target is already provided
    Y = targets
    
    print(f"  Output shapes: X={X.shape}, Y={Y.shape}")
    
    return X.astype(np.float32), Y.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Build two-tower training pairs")
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='Input NPZ files with sequences'
    )
    parser.add_argument(
        '--out',
        required=True,
        help='Output NPZ file for pairs'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.90,
        help='Fraction of data for training (default: 0.90)'
    )
    parser.add_argument(
        '--context-len',
        type=int,
        default=100,
        help='Number of context vectors (default: 100)'
    )
    parser.add_argument(
        '--target-offset',
        type=int,
        default=1,
        help='Offset from context end to target (default: 1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/val split (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("TWO-TOWER PAIR BUILDER")
    print("="*60)
    
    # Load all input sequences
    all_contexts = []
    all_targets = []
    
    for input_path in args.inputs:
        contexts, targets = load_sequences(input_path)
        all_contexts.append(contexts)
        all_targets.append(targets)
    
    # Concatenate
    contexts = np.concatenate(all_contexts, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    print(f"\nTotal sequences loaded: {len(contexts):,}")
    
    # Extract pairs
    X, Y = extract_pairs(contexts, targets, args.context_len, args.target_offset)
    
    # Train/val split
    print(f"\nSplitting data:")
    print(f"  Train split: {args.train_split:.1%}")
    print(f"  Random seed: {args.seed}")
    
    n = len(X)
    n_train = int(n * args.train_split)
    
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(n)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_val = X[val_idx]
    Y_val = Y[val_idx]
    
    print(f"  Train: {len(X_train):,} pairs")
    print(f"  Val:   {len(X_val):,} pairs")
    
    # Save
    output_path = Path(args.out)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    print(f"\nSaving pairs to: {output_path}")
    
    np.savez_compressed(
        output_path,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        metadata=np.array([{
            'total_pairs': n,
            'train_pairs': len(X_train),
            'val_pairs': len(X_val),
            'context_len': args.context_len,
            'target_offset': args.target_offset,
            'train_split': args.train_split,
            'seed': args.seed,
            'input_files': args.inputs
        }], dtype=object)
    )
    
    # Save manifest JSON
    manifest_path = output_path.parent / 'manifest.json'
    manifest = {
        'total_pairs': int(n),
        'train_pairs': int(len(X_train)),
        'val_pairs': int(len(X_val)),
        'context_len': int(args.context_len),
        'vector_dim': int(X.shape[2]),
        'train_split': float(args.train_split),
        'seed': int(args.seed),
        'input_files': args.inputs,
        'output_file': str(output_path)
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Manifest saved to: {manifest_path}")
    print("\n" + "="*60)
    print("PAIR BUILDING COMPLETE")
    print("="*60)
    print(f"Train pairs: {len(X_train):,}")
    print(f"Val pairs:   {len(X_val):,}")
    print(f"Context len: {args.context_len}")
    print(f"Vector dim:  {X.shape[2]}")
    print("="*60)


if __name__ == '__main__':
    main()
