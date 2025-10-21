#!/usr/bin/env python3
"""
Diagnostic script to verify FAISS index integrity and oracle recall.

This script performs sanity checks to determine if the 0% Hit@5 in cascade
experiments is due to:
1. FAISS index corruption/misconfiguration
2. Data alignment issues (target IDs not in bank)
3. Vector normalization problems

Oracle Recall Test:
- For each validation pair (x_t → y_t), search FAISS with y_t itself
- If Recall@K is not near 100%, the index is broken

Bank Alignment Test:
- Verify all target_next_ids exist in the bank
- Check self-similarity: cos(bank[id], bank[id]) ≈ 1.0
- Confirm vector L2-normalization
"""

import argparse
import numpy as np
import faiss
from pathlib import Path
import json
from typing import Dict, List, Tuple


def load_faiss_index(index_path: str) -> faiss.Index:
    """Load FAISS index from disk."""
    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    print(f"  Index type: {type(index).__name__}")
    print(f"  Vectors in index: {index.ntotal:,}")
    print(f"  Dimension: {index.d}")
    return index


def load_vector_bank(npz_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load vector bank with metadata."""
    print(f"\nLoading vector bank from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    vectors = data['vectors']
    concept_texts = data['concept_texts']
    cpe_ids = data['cpe_ids']

    print(f"  Vectors: {vectors.shape}")
    print(f"  Concept texts: {len(concept_texts):,}")
    print(f"  CPE IDs: {len(cpe_ids):,}")

    return vectors, concept_texts, cpe_ids


def load_validation_data(npz_path: str) -> Dict[str, np.ndarray]:
    """Load validation sequences."""
    print(f"\nLoading validation data from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    # Try both key name formats
    if 'context_sequences' in data:
        val_seqs = data['context_sequences']
        val_next = data['target_vectors']
    else:
        val_seqs = data['val_seqs']
        val_next = data['val_next']

    # Check if target_indices exist
    if 'target_indices' in data:
        target_indices = data['target_indices']
        print(f"  ✓ target_indices found: {target_indices.shape}")
    else:
        print(f"  ✗ target_indices NOT found - will compute from vectors")
        target_indices = None

    print(f"  Validation sequences: {val_seqs.shape[0]:,}")
    print(f"  Context length: {val_seqs.shape[1]}")

    return {
        'val_seqs': val_seqs,
        'val_next': val_next,
        'target_indices': target_indices
    }


def check_vector_normalization(vectors: np.ndarray, sample_size: int = 1000):
    """Check if vectors are L2-normalized."""
    print(f"\n{'='*60}")
    print("NORMALIZATION CHECK")
    print(f"{'='*60}")

    # Sample random vectors
    indices = np.random.choice(len(vectors), min(sample_size, len(vectors)), replace=False)
    sample = vectors[indices]

    # Compute L2 norms
    norms = np.linalg.norm(sample, axis=1)

    print(f"L2 norms (sample of {len(sample):,}):")
    print(f"  Min:    {norms.min():.6f}")
    print(f"  Max:    {norms.max():.6f}")
    print(f"  Mean:   {norms.mean():.6f}")
    print(f"  Median: {np.median(norms):.6f}")
    print(f"  Std:    {norms.std():.6f}")

    # Check if normalized
    normalized_count = np.sum(np.abs(norms - 1.0) < 0.001)
    normalized_pct = 100 * normalized_count / len(norms)

    print(f"\nNormalized (within 0.001 of 1.0): {normalized_count:,} / {len(norms):,} ({normalized_pct:.1f}%)")

    if normalized_pct > 95:
        print("  ✓ Vectors appear L2-normalized")
        return True
    else:
        print("  ✗ WARNING: Vectors may not be normalized!")
        return False


def check_self_similarity(vectors: np.ndarray, sample_size: int = 1000):
    """Check that cos(v, v) ≈ 1.0 for all vectors."""
    print(f"\n{'='*60}")
    print("SELF-SIMILARITY CHECK")
    print(f"{'='*60}")

    # Sample random vectors
    indices = np.random.choice(len(vectors), min(sample_size, len(vectors)), replace=False)
    sample = vectors[indices]

    # Normalize (in case they aren't)
    norms = np.linalg.norm(sample, axis=1, keepdims=True)
    sample_norm = sample / (norms + 1e-8)

    # Self dot product
    self_sim = np.sum(sample_norm * sample_norm, axis=1)

    print(f"Self-similarity cos(v, v) (sample of {len(sample):,}):")
    print(f"  Min:    {self_sim.min():.6f}")
    print(f"  Max:    {self_sim.max():.6f}")
    print(f"  Mean:   {self_sim.mean():.6f}")
    print(f"  Median: {np.median(self_sim):.6f}")

    # Should be very close to 1.0
    perfect_count = np.sum(np.abs(self_sim - 1.0) < 0.001)
    perfect_pct = 100 * perfect_count / len(self_sim)

    print(f"\nPerfect self-similarity (≥0.999): {perfect_count:,} / {len(self_sim):,} ({perfect_pct:.1f}%)")

    if perfect_pct > 95:
        print("  ✓ Self-similarity is correct")
        return True
    else:
        print("  ✗ WARNING: Self-similarity issues detected!")
        return False


def check_target_indices_in_bank(target_indices: np.ndarray, bank_size: int):
    """Verify all target indices are valid."""
    print(f"\n{'='*60}")
    print("TARGET INDEX VALIDITY CHECK")
    print(f"{'='*60}")

    print(f"Target indices: {len(target_indices):,}")
    print(f"Bank size: {bank_size:,}")

    # Check range
    min_idx = target_indices.min()
    max_idx = target_indices.max()

    print(f"\nTarget index range:")
    print(f"  Min: {min_idx:,}")
    print(f"  Max: {max_idx:,}")

    # Check validity
    invalid = (target_indices < 0) | (target_indices >= bank_size)
    invalid_count = invalid.sum()

    if invalid_count > 0:
        print(f"\n  ✗ ERROR: {invalid_count:,} invalid target indices (out of range)!")
        print(f"     First few invalid: {target_indices[invalid][:10]}")
        return False
    else:
        print(f"\n  ✓ All target indices are valid (0 ≤ idx < {bank_size:,})")
        return True


def oracle_recall_test(
    index: faiss.Index,
    val_next: np.ndarray,
    target_indices: np.ndarray,
    k_values: List[int] = [1, 5, 10, 50, 100, 500, 1000],
    nprobe: int = 32
):
    """
    Oracle recall test: Search FAISS with the TRUE target vectors.

    If FAISS can't find y_t when we search for y_t itself, the index is broken.
    """
    print(f"\n{'='*60}")
    print("ORACLE RECALL TEST")
    print(f"{'='*60}")
    print(f"Testing: Can FAISS find the exact target when we search with it?")
    print(f"Queries: {len(val_next):,} validation targets")
    print(f"nprobe: {nprobe}")

    # Set nprobe if index supports it
    if hasattr(index, 'nprobe'):
        index.nprobe = nprobe
        print(f"  ✓ Set nprobe={nprobe}")

    # Normalize queries
    val_next_norm = val_next.copy()
    norms = np.linalg.norm(val_next_norm, axis=1, keepdims=True)
    val_next_norm = val_next_norm / (norms + 1e-8)

    # Search FAISS with max K
    max_k = max(k_values)
    print(f"\nSearching FAISS for top-{max_k}...")
    distances, indices = index.search(val_next_norm.astype(np.float32), max_k)

    # Compute recall at various K
    print(f"\nOracle Recall Results:")
    print(f"{'K':>6} | {'Recall@K':>10} | {'Found':>8} / {'Total':>8}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*8}---{'-'*8}")

    results = {}
    for k in k_values:
        if k > max_k:
            continue

        # Check if target index is in top-k
        found = 0
        for i, target_idx in enumerate(target_indices):
            if target_idx in indices[i, :k]:
                found += 1

        recall = 100 * found / len(target_indices)
        results[k] = recall

        print(f"{k:6d} | {recall:9.2f}% | {found:8,} / {len(target_indices):8,}")

    # Verdict
    print(f"\n{'='*60}")
    if results.get(10, 0) > 90:
        print("✓ VERDICT: Oracle recall is EXCELLENT")
        print("  → FAISS index is working correctly")
        print("  → The 0% cascade Hit@5 is a MODEL limitation, not index bug")
    elif results.get(100, 0) > 50:
        print("⚠ VERDICT: Oracle recall is MODERATE")
        print("  → FAISS index may need tuning (increase nprobe, use HNSW)")
        print("  → The 0% cascade Hit@5 is partly an index recall problem")
    else:
        print("✗ VERDICT: Oracle recall is POOR")
        print("  → FAISS index has serious issues")
        print("  → Fix index before testing cascade!")
    print(f"{'='*60}")

    return results


def find_target_indices_by_vector_match(
    val_next: np.ndarray,
    bank_vectors: np.ndarray,
    threshold: float = 0.9999
) -> np.ndarray:
    """
    Fallback: Find target indices by matching vectors with high cosine similarity.

    This is slow but accurate for verification.
    """
    print(f"\nComputing target indices by vector matching...")
    print(f"  Threshold: cosine ≥ {threshold}")

    # Normalize
    val_next_norm = val_next / (np.linalg.norm(val_next, axis=1, keepdims=True) + 1e-8)
    bank_norm = bank_vectors / (np.linalg.norm(bank_vectors, axis=1, keepdims=True) + 1e-8)

    target_indices = []
    not_found = 0

    for i, query in enumerate(val_next_norm):
        # Dot product (cosine similarity)
        sims = np.dot(bank_norm, query)

        # Find best match
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]

        if best_sim >= threshold:
            target_indices.append(best_idx)
        else:
            target_indices.append(-1)  # Not found
            not_found += 1
            if not_found <= 5:  # Show first few
                print(f"  ✗ Target {i}: best similarity {best_sim:.4f} < {threshold}")

    target_indices = np.array(target_indices)

    valid = (target_indices >= 0).sum()
    print(f"  Found: {valid:,} / {len(val_next):,} ({100*valid/len(val_next):.1f}%)")

    return target_indices


def main():
    parser = argparse.ArgumentParser(description="FAISS Oracle Recall Diagnostic")
    parser.add_argument(
        '--index',
        default='artifacts/wikipedia_500k_corrected_ivf_flat_ip.index',
        help='FAISS index path'
    )
    parser.add_argument(
        '--vectors',
        default='artifacts/wikipedia_500k_corrected_vectors.npz',
        help='Vector bank NPZ path'
    )
    parser.add_argument(
        '--validation',
        default='artifacts/lvm/data_phase3_tmd/validation_sequences_ctx100.npz',
        help='Validation data NPZ path'
    )
    parser.add_argument(
        '--nprobe',
        type=int,
        default=32,
        help='nprobe for IVF index (default: 32)'
    )
    parser.add_argument(
        '--k-values',
        nargs='+',
        type=int,
        default=[1, 5, 10, 50, 100, 500, 1000],
        help='K values for recall@K (default: 1 5 10 50 100 500 1000)'
    )

    args = parser.parse_args()

    print("="*60)
    print("FAISS ORACLE RECALL DIAGNOSTIC")
    print("="*60)
    print(f"Purpose: Verify FAISS index integrity before cascade tuning")
    print()

    # Load data
    index = load_faiss_index(args.index)
    bank_vectors, concept_texts, cpe_ids = load_vector_bank(args.vectors)
    val_data = load_validation_data(args.validation)

    val_next = val_data['val_next']
    target_indices = val_data['target_indices']

    # Sanity checks
    print(f"\n{'='*60}")
    print("DATA SANITY CHECKS")
    print(f"{'='*60}")

    # 1. Normalization
    bank_normalized = check_vector_normalization(bank_vectors)

    # 2. Self-similarity
    self_sim_ok = check_self_similarity(bank_vectors)

    # 3. Target indices
    if target_indices is None:
        print("\n⚠ No target_indices in validation data - computing from vectors...")
        target_indices = find_target_indices_by_vector_match(val_next, bank_vectors)

        # Save for future use
        output_path = args.validation.replace('.npz', '_with_indices.npz')
        print(f"\nSaving updated validation data to: {output_path}")
        np.savez_compressed(
            output_path,
            val_seqs=val_data['val_seqs'],
            val_next=val_next,
            target_indices=target_indices
        )

    indices_valid = check_target_indices_in_bank(target_indices, len(bank_vectors))

    # 4. Oracle recall test
    if indices_valid:
        oracle_results = oracle_recall_test(
            index=index,
            val_next=val_next,
            target_indices=target_indices,
            k_values=args.k_values,
            nprobe=args.nprobe
        )

        # Save results
        output_dir = Path('artifacts/evals')
        output_dir.mkdir(exist_ok=True, parents=True)

        results_file = output_dir / 'oracle_recall_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'oracle_recall': oracle_results,
                'config': {
                    'index': args.index,
                    'vectors': args.vectors,
                    'validation': args.validation,
                    'nprobe': args.nprobe,
                    'num_queries': len(val_next),
                    'bank_size': len(bank_vectors)
                },
                'sanity_checks': {
                    'bank_normalized': bank_normalized,
                    'self_similarity_ok': self_sim_ok,
                    'indices_valid': indices_valid
                }
            }, f, indent=2)

        print(f"\n✓ Results saved to: {results_file}")
    else:
        print("\n✗ Cannot run oracle recall test - target indices are invalid!")
        print("   Fix data alignment issues first.")


if __name__ == '__main__':
    main()
