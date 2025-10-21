#!/usr/bin/env python3
"""
Recreate Phase-3 Validation Split

Reproduces the exact validation split used during Phase-3 training
by replicating the train_final.py split logic with seed=42.
"""

import numpy as np
from pathlib import Path

# Load Phase-3 training data
train_path = Path('artifacts/lvm/data_phase3/training_sequences_ctx100.npz')
print(f"Loading training data from {train_path}...")
data = np.load(train_path, allow_pickle=True)

contexts = data['context_sequences']  # [1273, 1000, 768]
targets = data['target_vectors']  # [1273, 768]

n_sequences = len(contexts)
print(f"Total sequences: {n_sequences}")

# Recreate train_final.py's split logic (lines 98, 131-136)
chain_ids = np.arange(n_sequences)  # [0, 1, 2, ..., 1272]
unique_chains = np.unique(chain_ids)  # Same as chain_ids since all unique

# Shuffle with seed=42 (same as training)
rng = np.random.RandomState(42)
rng.shuffle(unique_chains)

# Split 90/10
train_ratio = 0.9
n_train = int(train_ratio * len(unique_chains))
train_chains = set(unique_chains[:n_train])
val_chains = set(unique_chains[n_train:])

# Create masks
train_mask = np.array([cid in train_chains for cid in chain_ids])
val_mask = np.array([cid in val_chains for cid in chain_ids])

# Extract validation data
val_contexts = contexts[val_mask]
val_targets = targets[val_mask]

print(f"\n=== Recreated Split ===")
print(f"Train sequences: {train_mask.sum()} (expected: 1146)")
print(f"Val sequences: {val_mask.sum()} (expected: 127)")
print(f"Overlap: {len(set(np.where(train_mask)[0]) & set(np.where(val_mask)[0]))} (MUST BE 0)")

# Save validation data
output_path = Path('artifacts/lvm/data_phase3/validation_phase3_exact.npz')
np.savez_compressed(
    output_path,
    context_sequences=val_contexts,
    target_vectors=val_targets,
    val_indices=np.where(val_mask)[0]  # Save which indices were used
)

print(f"\nSaved exact Phase-3 validation data to: {output_path}")
print(f"  Contexts: {val_contexts.shape}")
print(f"  Targets: {val_targets.shape}")
print(f"\nValidation sequence indices (first 10): {np.where(val_mask)[0][:10].tolist()}")
