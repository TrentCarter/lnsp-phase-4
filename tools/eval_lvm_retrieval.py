#!/usr/bin/env python3
"""
LVM Retrieval Evaluation: Hit@k Metrics

Evaluates LVM models using retrieval metrics (Hit@1/5/10) instead of just cosine similarity.

Critical evaluation approach:
1. Load trained model
2. For each validation sequence:
   - Predict next vector
   - Retrieve top-k nearest neighbors from validation pool
   - Check if true next concept is in top-k
3. Compute Hit@1, Hit@5, Hit@10

This gives us a more realistic measure of "can the model actually predict the next concept?"
vs just "how similar is the predicted vector to the target vector?"

Consultant claims:
- Need Hit@1 ≥30%, Hit@5 ≥55%, Hit@10 ≥70% for production
- Current models likely underperform on retrieval despite decent cosine scores

Let's validate these claims with DATA.

Usage:
    python tools/eval_lvm_retrieval.py \
        --model artifacts/lvm/models_extended_context/memory_gru_ctx100/best_model.pt \
        --model-type memory_gru \
        --val-data artifacts/lvm/data_extended/validation_sequences_ctx100.npz \
        --vectors artifacts/wikipedia_500k_corrected_vectors.npz

Created: 2025-10-19 (Critical evaluation of consultant suggestions)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.lvm.models import create_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: Path, model_type: str, device: str):
    """Load trained LVM model."""
    logger.info(f"Loading model: {model_path}")

    # Create model with default config
    model = create_model(model_type)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)

    # Handle both formats: direct state_dict or wrapped checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    logger.info(f"Model loaded: {model_type} ({model.count_parameters():,} params)")
    return model


def load_validation_data(val_path: Path):
    """Load validation sequences."""
    logger.info(f"Loading validation data: {val_path}")

    data = np.load(val_path)
    contexts = data['context_sequences']  # [N, 100, 768]
    targets = data['target_vectors']      # [N, 768]

    logger.info(f"Loaded {len(contexts)} validation sequences")
    logger.info(f"  Context shape: {contexts.shape}")
    logger.info(f"  Target shape: {targets.shape}")

    return contexts, targets


def load_vector_pool(vectors_path: Path, max_pool_size: int = 50000):
    """
    Load vector pool for retrieval.

    In production, this would be the full FAISS index.
    For evaluation, we use a subset of vectors as the retrieval pool.
    """
    logger.info(f"Loading vector pool: {vectors_path}")

    data = np.load(vectors_path, allow_pickle=True)
    vectors = data['vectors']  # [N, 768] or [N, 784]

    # If 784D (TMD+semantic), extract semantic 768D only
    if vectors.shape[1] == 784:
        vectors = vectors[:, :768]
        logger.info("Extracted 768D semantic vectors from 784D (TMD+semantic)")

    # Use subset if pool is too large
    if len(vectors) > max_pool_size:
        logger.info(f"Sampling {max_pool_size} vectors from {len(vectors)} total")
        indices = np.random.choice(len(vectors), max_pool_size, replace=False)
        vectors = vectors[indices]

    # L2 normalize for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-8)

    logger.info(f"Vector pool: {vectors.shape}")
    return vectors


def compute_hit_at_k(predicted_vector, target_vector, pool_vectors, k_values=[1, 5, 10]):
    """
    Compute Hit@k metrics.

    Args:
        predicted_vector: [768] predicted next vector
        target_vector: [768] ground truth next vector
        pool_vectors: [N, 768] retrieval pool (L2 normalized)
        k_values: List of k values to evaluate

    Returns:
        dict: {k: bool} - whether target is in top-k
    """
    # Normalize predicted vector
    pred_norm = predicted_vector / (np.linalg.norm(predicted_vector) + 1e-8)
    target_norm = target_vector / (np.linalg.norm(target_vector) + 1e-8)

    # Compute cosine similarity with all pool vectors
    similarities = np.dot(pool_vectors, pred_norm)  # [N]

    # Get top-k indices
    top_k_indices = np.argsort(similarities)[::-1][:max(k_values)]

    # Find rank of target vector
    # Compare target with each top-k vector
    target_rank = None
    for rank, idx in enumerate(top_k_indices):
        # Check if this pool vector matches the target (cosine > 0.99 = same vector)
        if np.dot(pool_vectors[idx], target_norm) > 0.99:
            target_rank = rank
            break

    # Compute Hit@k for each k
    hits = {}
    for k in k_values:
        hits[k] = (target_rank is not None and target_rank < k)

    return hits


@torch.no_grad()
def evaluate_model(model, val_contexts, val_targets, pool_vectors, device='cpu', k_values=[1, 5, 10]):
    """
    Evaluate model using Hit@k retrieval metrics.

    Returns:
        dict: Evaluation metrics
    """
    logger.info(f"Evaluating model on {len(val_contexts)} sequences...")

    model.eval()

    hit_counts = {k: 0 for k in k_values}
    cosine_sims = []

    for i in tqdm(range(len(val_contexts)), desc="Evaluating"):
        context = val_contexts[i]  # [100, 768]
        target = val_targets[i]    # [768]

        # Prepare input
        context_tensor = torch.FloatTensor(context).unsqueeze(0).to(device)  # [1, 100, 768]

        # Predict
        prediction = model(context_tensor)  # [1, 768]
        prediction = prediction.cpu().numpy()[0]  # [768]

        # Compute Hit@k
        hits = compute_hit_at_k(prediction, target, pool_vectors, k_values)
        for k in k_values:
            if hits[k]:
                hit_counts[k] += 1

        # Also compute cosine similarity for comparison
        pred_norm = prediction / (np.linalg.norm(prediction) + 1e-8)
        target_norm = target / (np.linalg.norm(target) + 1e-8)
        cosine = np.dot(pred_norm, target_norm)
        cosine_sims.append(cosine)

    # Compute final metrics
    n = len(val_contexts)
    results = {
        'num_samples': n,
        'mean_cosine': float(np.mean(cosine_sims)),
        'std_cosine': float(np.std(cosine_sims)),
    }

    for k in k_values:
        results[f'hit@{k}'] = hit_counts[k] / n
        results[f'hit@{k}_count'] = hit_counts[k]

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LVM models using Hit@k retrieval metrics"
    )
    parser.add_argument(
        '--model',
        type=Path,
        required=True,
        help='Path to trained model checkpoint (best_model.pt)'
    )
    parser.add_argument(
        '--model-type',
        required=True,
        choices=['gru', 'lstm', 'transformer', 'amn', 'hierarchical_gru', 'memory_gru'],
        help='Model architecture type'
    )
    parser.add_argument(
        '--val-data',
        type=Path,
        required=True,
        help='Validation data NPZ file (validation_sequences_ctx100.npz)'
    )
    parser.add_argument(
        '--vectors',
        type=Path,
        required=True,
        help='Vector pool NPZ file (for retrieval)'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'mps', 'cuda'],
        help='Device to run evaluation on'
    )
    parser.add_argument(
        '--k-values',
        nargs='+',
        type=int,
        default=[1, 5, 10],
        help='K values for Hit@k evaluation'
    )
    parser.add_argument(
        '--pool-size',
        type=int,
        default=50000,
        help='Maximum size of retrieval pool (for memory constraints)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON file for results (optional)'
    )

    args = parser.parse_args()

    # Load components
    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        device = 'cpu'

    model = load_model(args.model, args.model_type, device)
    val_contexts, val_targets = load_validation_data(args.val_data)
    pool_vectors = load_vector_pool(args.vectors, args.pool_size)

    # Evaluate
    results = evaluate_model(
        model,
        val_contexts,
        val_targets,
        pool_vectors,
        device=device,
        k_values=args.k_values
    )

    # Print results
    logger.info("\n" + "="*60)
    logger.info("RETRIEVAL EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"Model: {args.model_type}")
    logger.info(f"Checkpoint: {args.model}")
    logger.info(f"Validation samples: {results['num_samples']}")
    logger.info(f"Retrieval pool size: {len(pool_vectors)}")
    logger.info("")
    logger.info("Cosine Similarity Metrics:")
    logger.info(f"  Mean: {results['mean_cosine']:.4f} ± {results['std_cosine']:.4f}")
    logger.info("")
    logger.info("Retrieval Metrics (Hit@k):")
    for k in args.k_values:
        hit_rate = results[f'hit@{k}']
        hit_count = results[f'hit@{k}_count']
        logger.info(f"  Hit@{k:2d}: {hit_rate*100:5.2f}% ({hit_count}/{results['num_samples']})")
    logger.info("")

    # Consultant's thresholds
    logger.info("Consultant's Production Thresholds:")
    logger.info(f"  Hit@1:  ≥30%  → Current: {results['hit@1']*100:.2f}% {'✓' if results['hit@1'] >= 0.30 else '✗'}")
    logger.info(f"  Hit@5:  ≥55%  → Current: {results['hit@5']*100:.2f}% {'✓' if results['hit@5'] >= 0.55 else '✗'}")
    logger.info(f"  Hit@10: ≥70%  → Current: {results['hit@10']*100:.2f}% {'✓' if results['hit@10'] >= 0.70 else '✗'}")
    logger.info("="*60)

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {args.output}")

    # Exit code based on thresholds
    passes_threshold = (
        results['hit@1'] >= 0.30 and
        results['hit@5'] >= 0.55 and
        results['hit@10'] >= 0.70
    )

    if passes_threshold:
        logger.info("\n✅ Model PASSES production thresholds!")
        return 0
    else:
        logger.info("\n⚠️  Model does NOT pass production thresholds (yet)")
        return 1


if __name__ == '__main__':
    sys.exit(main())
