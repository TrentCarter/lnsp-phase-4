#!/usr/bin/env python3
"""
Create 10k Training Subset for GraphMERT-LVM Benchmark
=======================================================

Extract first 10k sequences from full 80k training data for quick benchmark.
"""

import numpy as np
from pathlib import Path

def create_10k_subset():
    """Create 10k subset from full training data"""

    # Load full training data
    print("Loading full training data...")
    full_data = np.load('artifacts/lvm/training_sequences_ctx5.npz')

    print(f"Full dataset size: {full_data['context_sequences'].shape[0]}")

    # Extract first 10k sequences
    n_subset = 10000
    contexts_10k = full_data['context_sequences'][:n_subset]
    targets_10k = full_data['target_vectors'][:n_subset]

    print(f"Subset size: {contexts_10k.shape[0]}")
    print(f"Context shape: {contexts_10k.shape}")
    print(f"Target shape: {targets_10k.shape}")

    # Save 10k subset
    output_path = Path('artifacts/lvm/training_sequences_ctx5_10k.npz')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        context_sequences=contexts_10k,
        target_vectors=targets_10k,
        context_length=5,
        num_sequences=n_subset,
        vector_dim=768
    )

    print(f"\n✓ Saved 10k subset to: {output_path}")
    print(f"  Size: {output_path.stat().st_size / (1024*1024):.1f} MB")

if __name__ == '__main__':
    create_10k_subset()
