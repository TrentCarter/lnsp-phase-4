#!/usr/bin/env python3
"""
Export LVM training data from corrected vectors.

Creates sliding-window sequences for autoregressive training:
- Context: [vec1, vec2, vec3, vec4, vec5]
- Target: vec6

Format compatible with app/lvm/train_transformer.py
"""

import argparse
import numpy as np
from pathlib import Path

def create_training_sequences(vectors, context_length=5):
    """
    Create training sequences using sliding window.

    Args:
        vectors: numpy array [N, 768]
        context_length: number of context vectors

    Returns:
        tuple: (context_sequences, target_vectors)
            - context_sequences: [N_samples, context_length, 768]
            - target_vectors: [N_samples, 768]
    """
    N, D = vectors.shape

    if N < context_length + 1:
        raise ValueError(f"Need at least {context_length + 1} vectors, got {N}")

    # Calculate number of sequences
    num_sequences = N - context_length

    print(f"Creating sequences:")
    print(f"  Total vectors:     {N:,}")
    print(f"  Context length:    {context_length}")
    print(f"  Training samples:  {num_sequences:,}")
    print()

    # Preallocate arrays
    context_sequences = np.zeros((num_sequences, context_length, D), dtype=np.float32)
    target_vectors = np.zeros((num_sequences, D), dtype=np.float32)

    # Fill sequences with sliding window
    for i in range(num_sequences):
        context_sequences[i] = vectors[i:i+context_length]
        target_vectors[i] = vectors[i+context_length]

        if (i + 1) % 10000 == 0:
            print(f"  Created {i+1:,} / {num_sequences:,} sequences...")

    print(f"✓ Created {num_sequences:,} training sequences")
    print()

    return context_sequences, target_vectors


def main():
    parser = argparse.ArgumentParser(description="Export LVM training data")
    parser.add_argument('--input', default='artifacts/wikipedia_500k_corrected_vectors.npz',
                       help='Input NPZ file with vectors')
    parser.add_argument('--output-dir', default='artifacts/lvm',
                       help='Output directory')
    parser.add_argument('--context-length', type=int, default=5,
                       help='Number of context vectors')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of sequences (for testing)')

    args = parser.parse_args()

    print("=" * 80)
    print("Export LVM Training Data")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Input:          {args.input}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Context length: {args.context_length}")
    if args.max_samples:
        print(f"  Max samples:    {args.max_samples:,}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================================
    # Step 1: Load Vectors
    # ============================================================================

    print("Step 1: Loading vectors from NPZ...")
    print()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
        print()
        print("Run this first:")
        print("  ./.venv/bin/python tools/rebuild_faiss_with_corrected_vectors.py")
        return

    data = np.load(input_path, allow_pickle=True)
    vectors = data['vectors']
    cpe_ids = data['cpe_ids']
    concept_texts = data['concept_texts']

    print(f"✓ Loaded from: {input_path}")
    print(f"  Vectors shape: {vectors.shape}")
    print(f"  CPE IDs:       {len(cpe_ids):,}")
    print(f"  Texts:         {len(concept_texts):,}")
    print()

    # Verify vectors are normalized (GTR-T5 should produce L2-normalized vectors)
    norms = np.linalg.norm(vectors, axis=1)
    print(f"Vector norms: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}")
    if not (0.99 < norms.mean() < 1.01):
        print("⚠️  Vectors are not L2-normalized! Normalizing now...")
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        print("✓ Vectors normalized")
    print()

    # ============================================================================
    # Step 2: Create Training Sequences
    # ============================================================================

    print("Step 2: Creating training sequences...")
    print()

    context_sequences, target_vectors = create_training_sequences(
        vectors,
        context_length=args.context_length
    )

    # Limit samples if requested (for testing)
    if args.max_samples and args.max_samples < len(context_sequences):
        print(f"Limiting to {args.max_samples:,} samples (for testing)...")
        indices = np.random.choice(len(context_sequences), args.max_samples, replace=False)
        context_sequences = context_sequences[indices]
        target_vectors = target_vectors[indices]
        print()

    # ============================================================================
    # Step 3: Save Training Data
    # ============================================================================

    print("Step 3: Saving training data...")
    print()

    output_filename = f"training_sequences_ctx{args.context_length}.npz"
    if args.max_samples:
        output_filename = f"training_sequences_ctx{args.context_length}_n{args.max_samples}.npz"

    output_path = output_dir / output_filename

    print(f"Saving to: {output_path}")
    np.savez_compressed(
        output_path,
        context_sequences=context_sequences,
        target_vectors=target_vectors,
        context_length=args.context_length,
        num_sequences=len(context_sequences),
        vector_dim=vectors.shape[1]
    )

    output_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✓ Training data saved: {output_size_mb:.1f} MB")
    print()

    # ============================================================================
    # Step 4: Verify Quality
    # ============================================================================

    print("Step 4: Verifying training data quality...")
    print()

    # Check for NaN/Inf
    has_nan = np.isnan(context_sequences).any() or np.isnan(target_vectors).any()
    has_inf = np.isinf(context_sequences).any() or np.isinf(target_vectors).any()

    if has_nan:
        print("✗ WARNING: Found NaN values in training data!")
    elif has_inf:
        print("✗ WARNING: Found Inf values in training data!")
    else:
        print("✓ No NaN/Inf values found")

    # Check shape consistency
    if context_sequences.shape[0] == target_vectors.shape[0]:
        print(f"✓ Shape consistency OK ({context_sequences.shape[0]:,} samples)")
    else:
        print(f"✗ Shape mismatch: contexts={context_sequences.shape[0]}, targets={target_vectors.shape[0]}")

    # Sample data check
    print()
    print("Sample training pair:")
    print(f"  Context shape:  {context_sequences[0].shape}")
    print(f"  Target shape:   {target_vectors[0].shape}")
    print(f"  Context[0] norm: {np.linalg.norm(context_sequences[0][0]):.6f}")
    print(f"  Target norm:     {np.linalg.norm(target_vectors[0]):.6f}")
    print()

    # ============================================================================
    # Summary
    # ============================================================================

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    print(f"✅ Training data exported successfully!")
    print()
    print(f"Input vectors:       {len(vectors):,}")
    print(f"Training sequences:  {len(context_sequences):,}")
    print(f"Context length:      {args.context_length}")
    print(f"Vector dimension:    {vectors.shape[1]}D")
    print()
    print(f"Output file:         {output_path} ({output_size_mb:.1f} MB)")
    print()
    print("Next step: Train LVM-T model!")
    print()
    print("Training command:")
    print(f"  ./.venv/bin/python app/lvm/train_transformer.py \\")
    print(f"    --data {output_path} \\")
    print(f"    --epochs 20 \\")
    print(f"    --batch-size 32 \\")
    print(f"    --device {'mps' if Path('/dev/null').exists() else 'cpu'}")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
