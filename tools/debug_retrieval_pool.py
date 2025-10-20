#!/usr/bin/env python3
"""
Debug script to check if validation targets exist in retrieval pool.

The Hit@k results (0.39%) are suspiciously low. This script checks:
1. Are validation targets actually in the retrieval pool?
2. If we just search for the target itself, do we find it?

This will tell us if the evaluation metric is broken or if performance is really this bad.
"""

import numpy as np
from pathlib import Path

print("Loading validation data...")
val_data = np.load('artifacts/lvm/data_extended/validation_sequences_ctx100.npz')
val_targets = val_data['target_vectors']  # [1275, 768]

print("Loading vector pool...")
pool_data = np.load('artifacts/wikipedia_500k_corrected_vectors.npz', allow_pickle=True)
pool_vectors = pool_data['vectors']  # [637997, 768] or [637997, 784]

# If 784D, extract 768D
if pool_vectors.shape[1] == 784:
    pool_vectors = pool_vectors[:, :768]
    print(f"Extracted 768D from 784D vectors")

print(f"\nValidation targets: {val_targets.shape}")
print(f"Pool vectors: {pool_vectors.shape}")

# Sample pool (same as eval script)
sample_size = 10000
np.random.seed(42)  # Same seed as eval
indices = np.random.choice(len(pool_vectors), sample_size, replace=False)
pool_sample = pool_vectors[indices]

# L2 normalize
pool_sample = pool_sample / (np.linalg.norm(pool_sample, axis=1, keepdims=True) + 1e-8)

print(f"\nSampled pool: {pool_sample.shape}")

# Check: For each validation target, can we find it in the pool?
print("\nChecking if targets exist in pool...")
found_exact = 0
found_0_99 = 0
found_0_95 = 0

for i, target in enumerate(val_targets[:10]):  # Check first 10
    target_norm = target / (np.linalg.norm(target) + 1e-8)

    # Compute cosine similarity with pool
    sims = np.dot(pool_sample, target_norm)
    max_sim = sims.max()

    if max_sim > 0.999:
        found_exact += 1
    if max_sim > 0.99:
        found_0_99 += 1
    if max_sim > 0.95:
        found_0_95 += 1

    if i < 3:
        print(f"  Target {i}: max similarity = {max_sim:.6f}")

print(f"\nOut of first 10 targets:")
print(f"  Found with >0.999 sim: {found_exact}")
print(f"  Found with >0.99 sim: {found_0_99}")
print(f"  Found with >0.95 sim: {found_0_95}")

print("\n" + "="*60)
print("DIAGNOSIS:")
if found_0_99 < 5:
    print("❌ PROBLEM: Validation targets NOT in sampled pool!")
    print("   → This explains 0% Hit@k")
    print("   → Validation set might be from different data source")
    print("   → OR pool sampling is excluding validation vectors")
else:
    print("✓ Targets exist in pool - low Hit@k is real model problem")
