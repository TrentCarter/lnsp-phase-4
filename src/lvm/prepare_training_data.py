"""
Prepare training data for LVM from ordered ontology chains.

This script loads:
1. Ordered chains from JSONL (already ingested)
2. Vectors from NPZ (already computed)

Creates training sequences: predict vector[i+1] from vector[0:i]

NO NEO4J! NO GRAPH WALKS! Just use the data we already have!
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple


def load_chains_and_vectors(
    chains_path: str = "artifacts/ontology_chains/wordnet_chains.jsonl",
    npz_path: str = "artifacts/fw10k_vectors.npz"
) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    """
    Load chains and their vectors.

    Returns:
        chains: List of chain dicts
        concept_texts: Array of concept strings (for matching)
        vectors: Array of vectors (768D or 784D)
    """
    # Load chains
    chains = []
    with open(chains_path) as f:
        for line in f:
            chain = json.loads(line)
            chains.append(chain)

    print(f"✅ Loaded {len(chains)} chains from {chains_path}")

    # Load vectors
    npz = np.load(npz_path)
    concept_texts = npz['concept_texts']
    vectors = npz['vectors']

    print(f"✅ Loaded {len(vectors)} vectors @ {vectors.shape[1]}D from {npz_path}")

    return chains, concept_texts, vectors


def create_training_sequences(
    chains: List[dict],
    concept_texts: np.ndarray,
    vectors: np.ndarray
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Create training sequences from chains.

    For each chain: [c0, c1, c2, c3]
    Create sequences:
      - context: [c0], target: c1
      - context: [c0, c1], target: c2
      - context: [c0, c1, c2], target: c3

    Returns:
        context_sequences: List of variable-length contexts
        target_vectors: Array of target vectors
    """
    # Build concept → index lookup
    concept_to_idx = {text: i for i, text in enumerate(concept_texts)}

    all_contexts = []
    all_targets = []
    missing_count = 0

    for chain in chains:
        concepts = chain['concepts']

        # Match concepts to vector indices
        indices = []
        for concept in concepts:
            if concept in concept_to_idx:
                indices.append(concept_to_idx[concept])
            else:
                missing_count += 1

        if len(indices) < 2:
            continue  # Need at least 2 concepts for training

        # Create training pairs
        chain_vecs = vectors[indices]  # Shape: (chain_len, 768/784)

        for i in range(1, len(chain_vecs)):
            context = chain_vecs[:i]  # All vectors up to i
            target = chain_vecs[i]    # Next vector

            all_contexts.append(context)
            all_targets.append(target)

    if missing_count > 0:
        print(f"⚠️  {missing_count} concepts not found in NPZ (skipped)")

    print(f"✅ Created {len(all_contexts)} training sequences")

    return all_contexts, np.array(all_targets)


def pad_and_save(
    contexts: List[np.ndarray],
    targets: np.ndarray,
    output_path: str,
    train_split: float = 0.7,
    val_split: float = 0.15
):
    """
    Pad sequences and split into train/val/test.
    """
    # Pad contexts to max length
    max_len = max(len(ctx) for ctx in contexts)
    print(f"Max context length: {max_len}")

    padded_contexts = []
    masks = []

    for ctx in contexts:
        # Pad
        pad_len = max_len - len(ctx)
        if pad_len > 0:
            padding = np.zeros((pad_len, ctx.shape[1]))
            padded = np.vstack([ctx, padding])
        else:
            padded = ctx

        # Mask (1 = valid, 0 = padding)
        mask = np.array([1] * len(ctx) + [0] * pad_len)

        padded_contexts.append(padded)
        masks.append(mask)

    contexts = np.array(padded_contexts)  # Shape: (N, max_len, 768/784)
    masks = np.array(masks)               # Shape: (N, max_len)

    # Split
    N = len(contexts)
    train_end = int(N * train_split)
    val_end = int(N * (train_split + val_split))

    train_ctx = contexts[:train_end]
    train_tgt = targets[:train_end]
    train_mask = masks[:train_end]

    val_ctx = contexts[train_end:val_end]
    val_tgt = targets[train_end:val_end]
    val_mask = masks[train_end:val_end]

    test_ctx = contexts[val_end:]
    test_tgt = targets[val_end:]
    test_mask = masks[val_end:]

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        train_contexts=train_ctx,
        train_targets=train_tgt,
        train_masks=train_mask,
        val_contexts=val_ctx,
        val_targets=val_tgt,
        val_masks=val_mask,
        test_contexts=test_ctx,
        test_targets=test_tgt,
        test_masks=test_mask
    )

    print(f"✅ Saved training data to {output_path}")
    print(f"   Train: {len(train_ctx)} sequences")
    print(f"   Val: {len(val_ctx)} sequences")
    print(f"   Test: {len(test_ctx)} sequences")
    print(f"   Context shape: {train_ctx.shape}")
    print(f"   Target shape: {train_tgt.shape}")


if __name__ == "__main__":
    # Load data
    chains, concept_texts, vectors = load_chains_and_vectors()

    # Create sequences
    contexts, targets = create_training_sequences(chains, concept_texts, vectors)

    # Pad and split
    pad_and_save(
        contexts,
        targets,
        "artifacts/lvm/wordnet_training_sequences.npz",
        train_split=0.7,
        val_split=0.15
    )

    print("✅ Training data preparation complete!")
