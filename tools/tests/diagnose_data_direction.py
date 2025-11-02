#!/usr/bin/env python3
"""
Diagnose Training Data Direction
=================================

Verifies if training sequences are in correct temporal order.

For a sequence: [v1, v2, v3, v4, v5] → v6

Expected behavior:
- v5 (last context) should be MOST similar to v6 (target)
- v1 (first context) should be LEAST similar to v6 (target)

If v1 is MORE similar than v5, sequences are reversed!
"""

import numpy as np
import argparse


def cosine(a, b):
    """Compute cosine similarity"""
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return (a_norm * b_norm).sum(axis=-1).mean()


def diagnose_data(npz_path: str, n_samples: int = 1000):
    """Diagnose if training sequences are in correct temporal order"""

    print("=" * 80)
    print("TRAINING DATA DIRECTION DIAGNOSIS")
    print("=" * 80)
    print()

    # Load data
    print(f"📥 Loading {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    # Get keys
    keys = sorted(data.keys())
    print(f"   Keys: {keys}")

    # Try different key names
    if 'context_sequences' in data:
        contexts = data['context_sequences']
        targets = data['target_vectors']
    elif 'contexts' in data:
        contexts = data['contexts']
        targets = data['targets']
    elif 'train_context_sequences' in data:
        contexts = data['train_context_sequences']
        targets = data['train_target_vectors']
    else:
        print(f"❌ ERROR: Cannot find context/target keys in NPZ")
        print(f"   Available keys: {sorted(data.keys())}")
        return

    print(f"   Contexts: {contexts.shape}")
    print(f"   Targets: {targets.shape}")
    print()

    # Limit samples
    n_samples = min(n_samples, len(contexts))
    contexts = contexts[:n_samples]
    targets = targets[:n_samples]

    # Test 1: Position-to-Target Similarity
    print("🔍 Test 1: Position-to-Target Similarity")
    print("   Expected: pos[4] > pos[3] > pos[2] > pos[1] > pos[0]")
    print()

    position_sims = []
    for i in range(5):
        sim = cosine(contexts[:, i, :], targets)
        position_sims.append(sim)
        print(f"   pos[{i}] → target: {sim:.4f}")

    print()

    # Check if increasing
    is_increasing = all(position_sims[i] < position_sims[i+1] for i in range(4))
    is_decreasing = all(position_sims[i] > position_sims[i+1] for i in range(4))

    if is_increasing:
        print("   ✅ CORRECT: Similarity increases toward target (forward sequence)")
    elif is_decreasing:
        print("   ❌ REVERSED: Similarity decreases toward target (backward sequence!)")
    else:
        print("   ⚠️  WARNING: Non-monotonic pattern (data may be shuffled)")

    print()

    # Test 2: Compare first vs last
    print("🔍 Test 2: First vs Last Position")
    first_sim = position_sims[0]
    last_sim = position_sims[4]

    print(f"   pos[0] (first) → target: {first_sim:.4f}")
    print(f"   pos[4] (last)  → target: {last_sim:.4f}")
    print(f"   Difference: {last_sim - first_sim:+.4f}")
    print()

    if last_sim > first_sim + 0.05:
        print("   ✅ CORRECT: Last position much closer to target")
    elif first_sim > last_sim + 0.05:
        print("   ❌ REVERSED: First position closer to target!")
    else:
        print("   ⚠️  WARNING: First and last positions similar (possible issue)")

    print()

    # Test 3: Internal coherence
    print("🔍 Test 3: Internal Context Coherence")
    print("   Expected: adjacent positions should be similar")
    print()

    coherences = []
    for i in range(4):
        coh = cosine(contexts[:, i, :], contexts[:, i+1, :])
        coherences.append(coh)
        print(f"   pos[{i}] ↔ pos[{i+1}]: {coh:.4f}")

    mean_coherence = np.mean(coherences)
    print(f"   Mean coherence: {mean_coherence:.4f}")
    print()

    if mean_coherence > 0.40:
        print("   ✅ GOOD: Context is coherent (adjacent chunks are similar)")
    else:
        print("   ⚠️  WARNING: Low coherence (context may be from different articles)")

    print()

    # Test 4: Sample Inspection
    print("🔍 Test 4: Sample Inspection (First 5 Sequences)")
    print()

    for s in range(min(5, n_samples)):
        ctx = contexts[s]
        tgt = targets[s]

        print(f"   Sample {s}:")
        for i in range(5):
            sim = cosine(ctx[i:i+1], tgt[None, :])
            print(f"      pos[{i}]: {sim:.4f}")
        print()

    # Final verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()

    if is_increasing and last_sim > first_sim + 0.05:
        print("✅ DATA IS CORRECT: Sequences are in forward temporal order")
        print("   - Position similarity increases toward target")
        print("   - Last context position is closest to target")
        print()
        print("❓ If models still predict backward, check:")
        print("   1. Model architecture (are inputs reversed?)")
        print("   2. Training loop (are contexts/targets swapped?)")
        print("   3. Inference code (is prediction compared to wrong target?)")
    elif is_decreasing or first_sim > last_sim + 0.05:
        print("❌ DATA IS REVERSED: Sequences are in BACKWARD order!")
        print("   - Position similarity DECREASES toward target")
        print("   - First context position is closest to target")
        print()
        print("🔧 FIXES:")
        print("   1. Reverse context sequences: contexts = contexts[:, ::-1, :]")
        print("   2. Or swap in sequence creation: target = vectors[i-1] instead of vectors[i+5]")
    else:
        print("⚠️  INCONCLUSIVE: Data pattern is unclear")
        print("   - Check for data corruption or mixed sources")

    print()


def main():
    parser = argparse.ArgumentParser(description="Diagnose training data direction")
    parser.add_argument("npz_path", help="Path to training sequences NPZ")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of samples to test (default: 1000)")

    args = parser.parse_args()
    diagnose_data(args.npz_path, args.n_samples)


if __name__ == "__main__":
    main()
