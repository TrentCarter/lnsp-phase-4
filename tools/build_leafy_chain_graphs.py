#!/usr/bin/env python3
"""
Build Leafy Chain Graphs for GraphMERT-LVM

Implements vector-space entity linking (no Vec2Text needed):
1. Load 80k Wikipedia concepts as entity pool
2. For each training sequence, link entities using cosine similarity
3. Apply α-filtering (threshold 0.55)
4. Build leafy chain graphs: roots (5 context vectors) + leaves (linked entities)
5. Save in GraphMERT-LVM training format
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
ALPHA_THRESHOLD = 0.55  # Minimum cosine similarity for entity linking
TOP_K_PER_VECTOR = 3    # Top-k entities per context vector
MAX_LEAVES = 15         # Maximum leaves per sequence (5 vectors × 3 entities)

# Relation types (simplified for Phase 1)
RELATION_TYPES = {
    'related_to': 0,       # Generic similarity
    'sequential': 1,       # From same document sequence
    'topical': 2,          # High similarity (>0.7)
    'contextual': 3,       # Medium similarity (0.55-0.7)
}

def load_entity_pool_from_npz(npz_path='artifacts/wikipedia_500k_corrected_vectors.npz'):
    """Load entity pool from NPZ file (simpler than PostgreSQL pgvector parsing)"""
    print(f"Loading entity pool from {npz_path}...")

    data = np.load(npz_path, allow_pickle=True)

    entity_vectors = data['vectors']        # (N, 768)
    entity_texts = data['concept_texts']    # (N,)
    entity_ids = data['cpe_ids']            # (N,)

    print(f"Loaded {len(entity_ids)} entities from Wikipedia NPZ")

    return {
        'ids': entity_ids.tolist(),
        'texts': entity_texts.tolist(),
        'vectors': entity_vectors
    }


def cosine_similarity(a, b):
    """Compute cosine similarity between vectors a and b"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_similarity_matrix(vectors_a, vectors_b):
    """Compute cosine similarity matrix between two sets of vectors"""
    # Normalize
    vectors_a_norm = vectors_a / np.linalg.norm(vectors_a, axis=1, keepdims=True)
    vectors_b_norm = vectors_b / np.linalg.norm(vectors_b, axis=1, keepdims=True)

    # Compute dot product (cosine similarity for normalized vectors)
    return np.dot(vectors_a_norm, vectors_b_norm.T)


def link_entities_to_sequence(context_vectors, entity_pool, top_k=3, alpha=0.55):
    """
    Link entities to a sequence using vector-space cosine similarity

    Args:
        context_vectors: (5, 768) context window
        entity_pool: dict with 'vectors' (N, 768)
        top_k: number of entities to link per context vector
        alpha: minimum cosine similarity threshold

    Returns:
        leaves: list of (head_idx, entity_idx, relation_type, cosine_score)
    """
    leaves = []

    # Compute similarity: (5, N)
    sim_matrix = cosine_similarity_matrix(context_vectors, entity_pool['vectors'])

    for head_idx in range(5):
        # Get top-k entities for this context vector
        similarities = sim_matrix[head_idx]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        for entity_idx in top_indices:
            cosine_score = similarities[entity_idx]

            # Apply α-filtering
            if cosine_score >= alpha:
                # Determine relation type based on similarity
                if cosine_score >= 0.7:
                    relation_type = RELATION_TYPES['topical']
                else:
                    relation_type = RELATION_TYPES['contextual']

                leaves.append({
                    'head_idx': head_idx,
                    'entity_idx': int(entity_idx),
                    'entity_id': entity_pool['ids'][entity_idx],
                    'relation_type': relation_type,
                    'cosine_score': float(cosine_score)
                })

    return leaves


def build_leafy_chain_graphs(training_data_path, entity_pool, output_path,
                              limit=None, alpha=0.55, top_k=3):
    """
    Build leafy chain graphs for all training sequences

    Args:
        training_data_path: path to training_sequences_ctx5.npz
        entity_pool: dict with entity data
        output_path: where to save leafy chain graphs
        limit: optional limit on number of sequences
        alpha: cosine similarity threshold
        top_k: entities per context vector
    """
    print(f"\nLoading training sequences from {training_data_path}...")
    data = np.load(training_data_path, allow_pickle=True)

    context_sequences = data['context_sequences']  # (N, 5, 768)
    target_vectors = data['target_vectors']        # (N, 768)

    if limit:
        context_sequences = context_sequences[:limit]
        target_vectors = target_vectors[:limit]

    n_sequences = len(context_sequences)
    print(f"Building leafy chain graphs for {n_sequences} sequences...")

    # Storage for leafy chain graphs
    root_vectors_list = []
    leaf_tails_list = []
    leaf_relations_list = []
    leaf_heads_list = []
    leaf_scores_list = []
    entity_ids_list = []

    for idx in range(n_sequences):
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{n_sequences} sequences...")

        context_vectors = context_sequences[idx]  # (5, 768)

        # Link entities
        leaves = link_entities_to_sequence(
            context_vectors,
            entity_pool,
            top_k=top_k,
            alpha=alpha
        )

        # Build graph structure
        root_vectors_list.append(context_vectors)

        # Pad leaves to max_leaves
        n_leaves = len(leaves)
        max_leaves = 15  # 5 vectors × 3 entities

        leaf_tails = np.full(max_leaves, -1, dtype=np.int32)
        leaf_relations = np.full(max_leaves, -1, dtype=np.int32)
        leaf_heads = np.full(max_leaves, -1, dtype=np.int32)
        leaf_scores = np.zeros(max_leaves, dtype=np.float32)
        entity_ids = [''] * max_leaves

        for i, leaf in enumerate(leaves[:max_leaves]):
            leaf_tails[i] = leaf['entity_idx']
            leaf_relations[i] = leaf['relation_type']
            leaf_heads[i] = leaf['head_idx']
            leaf_scores[i] = leaf['cosine_score']
            entity_ids[i] = leaf['entity_id']

        leaf_tails_list.append(leaf_tails)
        leaf_relations_list.append(leaf_relations)
        leaf_heads_list.append(leaf_heads)
        leaf_scores_list.append(leaf_scores)
        entity_ids_list.append(entity_ids)

    # Convert to arrays
    root_vectors = np.array(root_vectors_list)       # (N, 5, 768)
    leaf_tails = np.array(leaf_tails_list)           # (N, max_leaves)
    leaf_relations = np.array(leaf_relations_list)   # (N, max_leaves)
    leaf_heads = np.array(leaf_heads_list)           # (N, max_leaves)
    leaf_scores = np.array(leaf_scores_list)         # (N, max_leaves)
    entity_ids = np.array(entity_ids_list)           # (N, max_leaves)

    # Save
    print(f"\nSaving leafy chain graphs to {output_path}...")
    np.savez_compressed(
        output_path,
        root_vectors=root_vectors,
        target_vectors=target_vectors,
        leaf_tails=leaf_tails,
        leaf_relations=leaf_relations,
        leaf_heads=leaf_heads,
        leaf_scores=leaf_scores,
        entity_ids=entity_ids,
        # Metadata
        num_sequences=n_sequences,
        max_leaves=max_leaves,
        alpha_threshold=alpha,
        top_k=top_k,
        relation_types=RELATION_TYPES,
        created_at=str(datetime.now())
    )

    print(f"✓ Saved {n_sequences} leafy chain graphs")

    # Statistics
    avg_leaves = np.mean(np.sum(leaf_tails >= 0, axis=1))
    print(f"\nStatistics:")
    print(f"  Total sequences: {n_sequences}")
    print(f"  Average leaves per sequence: {avg_leaves:.2f}")
    print(f"  Max leaves: {max_leaves}")
    print(f"  Alpha threshold: {alpha}")
    print(f"  Top-k per vector: {top_k}")

    return output_path


def validate_leafy_chain_graphs(graph_path):
    """Validate leafy chain graph structure"""
    print(f"\n{'='*80}")
    print("Validating Leafy Chain Graphs")
    print(f"{'='*80}")

    data = np.load(graph_path, allow_pickle=True)

    print(f"\nKeys: {list(data.keys())}")
    print(f"\nShapes:")
    print(f"  root_vectors: {data['root_vectors'].shape}")
    print(f"  target_vectors: {data['target_vectors'].shape}")
    print(f"  leaf_tails: {data['leaf_tails'].shape}")
    print(f"  leaf_relations: {data['leaf_relations'].shape}")
    print(f"  leaf_heads: {data['leaf_heads'].shape}")
    print(f"  leaf_scores: {data['leaf_scores'].shape}")

    print(f"\nMetadata:")
    print(f"  Num sequences: {data['num_sequences']}")
    print(f"  Max leaves: {data['max_leaves']}")
    print(f"  Alpha threshold: {data['alpha_threshold']}")
    print(f"  Top-k: {data['top_k']}")
    print(f"  Created at: {data['created_at']}")

    # Sample
    print(f"\nSample leafy chain graph (sequence 0):")
    print(f"  Root vectors shape: {data['root_vectors'][0].shape}")
    print(f"  Leaves: {np.sum(data['leaf_tails'][0] >= 0)} linked entities")
    print(f"  Leaf tails: {data['leaf_tails'][0][:5]}")
    print(f"  Leaf relations: {data['leaf_relations'][0][:5]}")
    print(f"  Leaf heads: {data['leaf_heads'][0][:5]}")
    print(f"  Leaf scores: {data['leaf_scores'][0][:5]}")

    print(f"\n✓ Validation complete!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build Leafy Chain Graphs for GraphMERT-LVM')
    parser.add_argument('--training-data', default='artifacts/lvm/training_sequences_ctx5.npz')
    parser.add_argument('--entity-pool', default='artifacts/wikipedia_500k_corrected_vectors.npz')
    parser.add_argument('--output', default='artifacts/graphmert_lvm/leafy_chain_graphs_80k.npz')
    parser.add_argument('--alpha', type=float, default=0.55, help='Cosine similarity threshold')
    parser.add_argument('--top-k', type=int, default=3, help='Entities per context vector')
    parser.add_argument('--limit', type=int, default=None, help='Limit sequences (for testing)')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing graphs')
    args = parser.parse_args()

    if args.validate_only:
        validate_leafy_chain_graphs(args.output)
        return

    # Load entity pool from NPZ
    entity_pool = load_entity_pool_from_npz(args.entity_pool)

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Build leafy chain graphs
    output_path = build_leafy_chain_graphs(
        training_data_path=args.training_data,
        entity_pool=entity_pool,
        output_path=args.output,
        limit=args.limit,
        alpha=args.alpha,
        top_k=args.top_k
    )

    # Validate
    validate_leafy_chain_graphs(output_path)


if __name__ == '__main__':
    main()
