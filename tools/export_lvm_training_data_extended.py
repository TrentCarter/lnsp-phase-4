#!/usr/bin/env python3
"""
Export LVM Training Data with Extended Context

Creates training sequences with 100-vector context windows (vs current 5-vector).
This extends our effective context from ~100 tokens to ~2,000 tokens.

Context calculation:
- 1 concept vector ≈ 20 tokens (avg concept: "machine learning" = ~2 words)
- Current: 5 vectors = 100 tokens (tiny!)
- Extended: 100 vectors = 2,000 tokens (competitive with small LLMs)

Usage:
    python tools/export_lvm_training_data_extended.py \
        --input artifacts/fw600k_vectors_tmd.npz \
        --context-length 100 \
        --output-dir artifacts/lvm/data_extended/

Output:
    - training_sequences_ctx100.npz (90% of data)
    - validation_sequences_ctx100.npz (10% of data)
    - metadata.json (dataset stats)

Created: 2025-10-19 (for extended context experiments)
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_vectors_with_metadata(npz_path: Path):
    """Load vector NPZ with metadata (CPE IDs, concept texts)."""
    logger.info(f"Loading vectors from {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    vectors = data['vectors']  # [N, 768] or [N, 784]

    # Optional metadata (may not exist in older NPZ files)
    cpe_ids = data.get('cpe_ids', None)
    concept_texts = data.get('concept_texts', None)

    logger.info(f"Loaded {len(vectors)} vectors with shape {vectors.shape}")
    logger.info(f"  CPE IDs: {'✓' if cpe_ids is not None else '✗'}")
    logger.info(f"  Concept texts: {'✓' if concept_texts is not None else '✗'}")

    return {
        'vectors': vectors,
        'cpe_ids': cpe_ids,
        'concept_texts': concept_texts,
        'vector_dim': vectors.shape[1]
    }


def calculate_sequence_coherence(sequence):
    """
    Calculate coherence of a sequence as average cosine similarity between consecutive vectors.

    Args:
        sequence: [context_length, D] array of vectors

    Returns:
        coherence: float, average cosine similarity (range: -1 to 1, typically 0.4-0.9)
    """
    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(sequence, axis=1, keepdims=True)
    normalized = sequence / (norms + 1e-8)

    # Calculate cosine similarity between consecutive pairs
    similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)

    # Average similarity across the sequence
    coherence = np.mean(similarities)

    return coherence


def create_extended_sequences(vectors, context_length=100, overlap=50, min_coherence=None):
    """
    Create training sequences with extended context windows.

    Args:
        vectors: [N, D] array of concept vectors
        context_length: Number of vectors in context window (default 100)
        overlap: Number of overlapping vectors between sequences (default 50)
        min_coherence: Minimum sequence coherence (0.0-1.0). If set, filters out
                      sequences with avg cosine similarity < threshold. Recommended:
                      0.65-0.75 for long contexts (1000-2000 vectors).

    Returns:
        sequences: [num_sequences, context_length, D]
        targets: [num_sequences, D]
        coherence_scores: [num_sequences] array of coherence scores
        target_indices: [num_sequences] array of target bank indices (for TMD eval)
    """
    N, D = vectors.shape

    # Calculate number of sequences we can create
    stride = context_length - overlap
    num_sequences = (N - context_length - 1) // stride

    logger.info(f"Creating sequences:")
    logger.info(f"  Context length: {context_length} vectors (~{context_length * 20} tokens)")
    logger.info(f"  Overlap: {overlap} vectors")
    logger.info(f"  Stride: {stride} vectors")
    logger.info(f"  Total vectors: {N}")
    logger.info(f"  Potential sequences: {num_sequences}")
    if min_coherence is not None:
        logger.info(f"  Coherence filter: ≥{min_coherence:.2f} (filters noisy/off-topic sequences)")

    # First pass: create all sequences and calculate coherence
    sequences_list = []
    targets_list = []
    coherence_list = []
    target_indices_list = []  # NEW: Track source indices for TMD evaluation

    for i in tqdm(range(num_sequences), desc="Creating sequences"):
        start_idx = i * stride
        end_idx = start_idx + context_length
        target_idx = end_idx  # Next vector after context

        seq = vectors[start_idx:end_idx]
        target = vectors[target_idx]

        # Calculate coherence
        coherence = calculate_sequence_coherence(seq)

        # Apply coherence filter if specified
        if min_coherence is None or coherence >= min_coherence:
            sequences_list.append(seq)
            targets_list.append(target)
            coherence_list.append(coherence)
            target_indices_list.append(target_idx)  # NEW: Save target index

    # Convert to arrays
    num_kept = len(sequences_list)
    sequences = np.array(sequences_list, dtype=np.float32)
    targets = np.array(targets_list, dtype=np.float32)
    coherence_scores = np.array(coherence_list, dtype=np.float32)
    target_indices = np.array(target_indices_list, dtype=np.int64)  # NEW

    logger.info(f"\nSequence filtering results:")
    logger.info(f"  Total created: {num_sequences}")
    logger.info(f"  Kept: {num_kept} ({num_kept/num_sequences*100:.1f}%)")
    if min_coherence is not None:
        logger.info(f"  Filtered: {num_sequences - num_kept} ({(num_sequences - num_kept)/num_sequences*100:.1f}%)")
        logger.info(f"  Coherence stats (kept sequences):")
        logger.info(f"    Min: {coherence_scores.min():.3f}")
        logger.info(f"    Mean: {coherence_scores.mean():.3f}")
        logger.info(f"    Max: {coherence_scores.max():.3f}")

    return sequences, targets, coherence_scores, target_indices


def split_train_val(sequences, targets, target_indices, val_split=0.1, seed=42):
    """Split sequences into train/val with deterministic shuffle."""
    np.random.seed(seed)

    N = len(sequences)
    indices = np.arange(N)
    np.random.shuffle(indices)

    val_size = int(N * val_split)
    train_size = N - val_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_seqs = sequences[train_idx]
    train_targets = targets[train_idx]
    train_target_ids = target_indices[train_idx]  # NEW: Split indices

    val_seqs = sequences[val_idx]
    val_targets = targets[val_idx]
    val_target_ids = target_indices[val_idx]  # NEW: Split indices

    logger.info(f"Split: {train_size} train, {val_size} val ({val_split*100:.1f}%)")

    return (train_seqs, train_targets, train_target_ids), (val_seqs, val_targets, val_target_ids)


def save_dataset(output_dir: Path, train_data, val_data, metadata):
    """Save train/val NPZ files and metadata JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_seqs, train_targets, train_target_ids = train_data
    val_seqs, val_targets, val_target_ids = val_data

    # Save training data (with target indices for TMD evaluation)
    train_path = output_dir / "training_sequences_ctx100.npz"
    np.savez_compressed(
        train_path,
        context_sequences=train_seqs,
        target_vectors=train_targets,
        target_indices=train_target_ids  # NEW: Save target bank indices
    )
    logger.info(f"Saved training data: {train_path}")
    logger.info(f"  Sequences: {train_seqs.shape}")
    logger.info(f"  Targets: {train_targets.shape}")
    logger.info(f"  Target indices: {train_target_ids.shape}")

    # Save validation data (with target indices for TMD evaluation)
    val_path = output_dir / "validation_sequences_ctx100.npz"
    np.savez_compressed(
        val_path,
        context_sequences=val_seqs,
        target_vectors=val_targets,
        target_indices=val_target_ids  # NEW: Save target bank indices
    )
    logger.info(f"Saved validation data: {val_path}")
    logger.info(f"  Sequences: {val_seqs.shape}")
    logger.info(f"  Targets: {val_targets.shape}")
    logger.info(f"  Target indices: {val_target_ids.shape}")

    # Save metadata
    meta_path = output_dir / "metadata_ctx100.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export LVM training data with extended context (100 vectors)"
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input NPZ file with vectors (e.g., artifacts/fw600k_vectors_tmd.npz)'
    )
    parser.add_argument(
        '--context-length',
        type=int,
        default=100,
        help='Context window size in vectors (default: 100 ≈ 2k tokens)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=50,
        help='Overlap between sequences (default: 50 vectors)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1 = 10%%)'
    )
    parser.add_argument(
        '--min-coherence',
        type=float,
        default=None,
        help='Minimum sequence coherence (0.0-1.0). Filters sequences with avg cosine '
             'similarity < threshold. Recommended: 0.70 for 2000-context to remove noisy '
             'off-topic sequences. Use None for no filtering.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('artifacts/lvm/data_extended'),
        help='Output directory for training data'
    )

    args = parser.parse_args()

    # Load vectors
    data = load_vectors_with_metadata(args.input)
    vectors = data['vectors']

    # Create sequences (with optional coherence filtering)
    sequences, targets, coherence_scores, target_indices = create_extended_sequences(
        vectors,
        context_length=args.context_length,
        overlap=args.overlap,
        min_coherence=args.min_coherence
    )

    # Split train/val
    train_data, val_data = split_train_val(
        sequences,
        targets,
        target_indices,  # NEW: Pass target indices through
        val_split=args.val_split
    )

    # Metadata
    metadata = {
        'source_file': str(args.input),
        'context_length': args.context_length,
        'overlap': args.overlap,
        'vector_dim': data['vector_dim'],
        'total_vectors': len(vectors),
        'total_sequences': len(sequences),
        'train_sequences': len(train_data[0]),
        'val_sequences': len(val_data[0]),
        'val_split': args.val_split,
        'effective_tokens': args.context_length * 20,  # 1 vector ≈ 20 tokens
        'min_coherence': args.min_coherence,
        'coherence_stats': {
            'min': float(coherence_scores.min()),
            'mean': float(coherence_scores.mean()),
            'max': float(coherence_scores.max()),
            'std': float(coherence_scores.std())
        } if len(coherence_scores) > 0 else None,
        'comparison': {
            'previous_context': 5,
            'previous_tokens': 100,
            'new_context': args.context_length,
            'new_tokens': args.context_length * 20,
            'improvement': f"{args.context_length / 5}x context expansion"
        }
    }

    # Save everything
    save_dataset(args.output_dir, train_data, val_data, metadata)

    logger.info("\n" + "="*60)
    logger.info("Extended Context Export Complete!")
    logger.info("="*60)
    logger.info(f"Context window: {args.context_length} vectors (~{args.context_length * 20} tokens)")
    logger.info(f"Total sequences: {metadata['total_sequences']}")
    logger.info(f"Train: {metadata['train_sequences']}, Val: {metadata['val_sequences']}")
    logger.info(f"\nReady for training:")
    logger.info(f"  python app/lvm/train_unified.py \\")
    logger.info(f"    --model-type hierarchical_gru \\")
    logger.info(f"    --data {args.output_dir}/training_sequences_ctx100.npz \\")
    logger.info(f"    --val-data {args.output_dir}/validation_sequences_ctx100.npz")


if __name__ == '__main__':
    main()
