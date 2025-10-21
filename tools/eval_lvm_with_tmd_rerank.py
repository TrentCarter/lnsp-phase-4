#!/usr/bin/env python3
"""
Evaluate LVM with TMD Re-ranking

Tests if re-ranking LVM candidates using TMD lane similarity improves Hit@K.

Strategy:
1. LVM predicts next vector from context
2. Find top-K candidates (K=20 or 50) using cosine similarity
3. Boost candidates that match the query's TMD lane
4. Re-rank and measure Hit@1, Hit@5, Hit@10

Expected gain: +2-4% Hit@5 (consultant estimate)

Usage:
    python tools/eval_lvm_with_tmd_rerank.py \
        --model artifacts/lvm/models_phase3/run_1000ctx_pilot/best_val_hit5.pt \
        --data artifacts/lvm/data_phase3/validation_sequences_ctx100.npz \
        --vectors artifacts/wikipedia_500k_corrected_vectors.npz \
        --top-k 30 \
        --lane-boost 0.10
"""

import argparse
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Import LVM model
from app.lvm.models import MemoryAugmentedGRU

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: str = 'mps'):
    """Load trained LVM model."""
    logger.info(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Model config from checkpoint
    config = checkpoint.get('config', {})
    d_model = config.get('input_dim', 768)  # input_dim in checkpoint = d_model in model
    hidden_dim = config.get('hidden_dim', 512)
    num_layers = config.get('num_layers', 4)
    memory_slots = config.get('memory_slots', 2048)

    model = MemoryAugmentedGRU(
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        memory_slots=memory_slots
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded: {d_model}D → {hidden_dim}D hidden ({num_layers} layers, {memory_slots} memory slots)")
    logger.info(f"Best validation Hit@5: {checkpoint.get('best_val_hit5', 'N/A')}")

    return model


def load_validation_data(data_path: Path):
    """Load validation sequences with target indices."""
    logger.info(f"Loading validation data from {data_path}")

    data = np.load(data_path, allow_pickle=True)
    contexts = torch.from_numpy(data['context_sequences']).float()
    targets = torch.from_numpy(data['target_vectors']).float()

    # NEW: Load target indices for correct evaluation
    if 'target_indices' in data:
        target_indices = data['target_indices']  # Bank row IDs for each target
        logger.info(f"✓ Target indices loaded (enables correct Hit@K evaluation)")
    else:
        logger.warning("⚠️  No target_indices in validation data! Re-export with updated script.")
        target_indices = None

    logger.info(f"Loaded {len(contexts)} validation sequences")
    logger.info(f"Context shape: {contexts.shape}")

    return contexts, targets, target_indices


def load_vector_bank_with_lanes(vectors_path: Path):
    """Load full vector bank with TMD lane metadata."""
    logger.info(f"Loading vector bank from {vectors_path}")

    data = np.load(vectors_path, allow_pickle=True)
    vectors = data['vectors']  # [N, 768]
    tmd_lanes = data['tmd_lanes']  # [N] array of lane strings (e.g., "lane_0")

    # Convert lane strings to indices
    lane_to_idx = {f"lane_{i}": i for i in range(16)}
    lane_indices = np.array([lane_to_idx.get(lane, 0) for lane in tmd_lanes])

    logger.info(f"Loaded {len(vectors)} vectors with TMD lanes")
    logger.info(f"Lane distribution:")
    lane_counts = Counter(tmd_lanes)
    for lane, count in sorted(lane_counts.items(), key=lambda x: -x[1])[:5]:
        logger.info(f"  {lane}: {count:,} ({count/len(vectors)*100:.1f}%)")

    return torch.from_numpy(vectors).float(), lane_indices


def get_context_lane(context_vectors, vector_bank, lane_indices):
    """
    Determine the TMD lane of a context sequence.

    Uses majority vote: find which vectors in the bank match the context,
    and return the most common lane.

    Args:
        context_vectors: [context_len, D] tensor
        vector_bank: [N, D] tensor of all vectors
        lane_indices: [N] array of lane IDs

    Returns:
        lane_id: int, the dominant lane for this context
    """
    # Find nearest neighbor for each context vector
    # This is expensive but accurate
    context_norm = F.normalize(context_vectors, dim=1)
    bank_norm = F.normalize(vector_bank, dim=1)

    # Compute similarity for all context vectors at once
    similarities = torch.mm(context_norm, bank_norm.t())  # [context_len, N]
    nearest_indices = similarities.argmax(dim=1).cpu().numpy()

    # Get lanes of nearest neighbors
    context_lanes = lane_indices[nearest_indices]

    # Return most common lane (majority vote)
    lane_counts = Counter(context_lanes)
    dominant_lane = lane_counts.most_common(1)[0][0]

    return dominant_lane


@torch.no_grad()
def evaluate_with_tmd_rerank(
    model,
    val_contexts,
    val_targets,
    val_target_indices,  # True bank indices for targets (computed on-the-fly if None)
    vector_bank,
    lane_indices,
    device='mps',
    top_k=30,
    lane_boost=0.10,
    batch_size=32
):
    """
    Evaluate LVM with TMD re-ranking.

    Args:
        model: Trained LVM model
        val_contexts: [N, context_len, D] validation contexts
        val_targets: [N, D] validation targets
        val_target_indices: [N] bank row indices for targets (computed if None)
        vector_bank: [M, D] full vector bank
        lane_indices: [M] lane ID for each vector
        device: 'mps' or 'cpu'
        top_k: Number of candidates to retrieve before re-ranking
        lane_boost: Score boost for same-lane candidates (e.g., 0.10 = +10%)
        batch_size: Evaluation batch size

    Returns:
        metrics: Dict with baseline and re-ranked Hit@K
    """
    # Compute target_indices on the fly if not provided
    if val_target_indices is None:
        logger.info("Computing target_indices on-the-fly by matching targets to vector bank...")
        targets_on_device = val_targets.to(device)
        bank_on_device = vector_bank.to(device)

        target_norm = F.normalize(targets_on_device, dim=1)
        bank_norm = F.normalize(bank_on_device, dim=1)

        similarities = torch.mm(target_norm, bank_norm.t())  # [N, M]
        val_target_indices = similarities.argmax(dim=1).cpu().numpy()

        logger.info(f"✓ Computed {len(val_target_indices)} target indices from vector bank")

    model.eval()
    vector_bank = vector_bank.to(device)
    bank_norm = F.normalize(vector_bank, dim=1)

    num_samples = len(val_contexts)
    num_batches = (num_samples + batch_size - 1) // batch_size

    baseline_hit1 = []
    baseline_hit5 = []
    baseline_hit10 = []

    rerank_hit1 = []
    rerank_hit5 = []
    rerank_hit10 = []

    logger.info(f"Evaluating {num_samples} samples with TMD re-ranking...")
    logger.info(f"  Top-K candidates: {top_k}")
    logger.info(f"  Lane boost: {lane_boost:.2f} ({lane_boost*100:.0f}%)")

    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        batch_contexts = val_contexts[start_idx:end_idx].to(device)
        batch_targets = val_targets[start_idx:end_idx].to(device)
        batch_target_ids = val_target_indices[start_idx:end_idx]  # NEW: True target IDs

        # LVM prediction
        predictions = model(batch_contexts)
        pred_norm = F.normalize(predictions, dim=1)

        # Find top-K candidates
        similarities = torch.mm(pred_norm, bank_norm.t())  # [batch, M]
        topk_scores, topk_indices = similarities.topk(top_k, dim=1)  # [batch, top_k]

        # For each sample in batch
        for i in range(len(batch_contexts)):
            # NEW: Use true target index (no heuristic matching needed!)
            target_idx = batch_target_ids[i]

            # Baseline Hit@K (no re-ranking)
            candidate_indices = topk_indices[i].cpu().numpy()
            baseline_hit1.append(target_idx == candidate_indices[0])
            baseline_hit5.append(target_idx in candidate_indices[:5])
            baseline_hit10.append(target_idx in candidate_indices[:10])

            # TMD re-ranking
            # Get context lane
            context_lane = get_context_lane(
                batch_contexts[i].cpu(),
                vector_bank.cpu(),
                lane_indices
            )

            # Boost scores for same-lane candidates
            candidate_lanes = lane_indices[candidate_indices]
            lane_matches = (candidate_lanes == context_lane).astype(float)
            boosted_scores = topk_scores[i].cpu().numpy() * (1 + lane_boost * lane_matches)

            # Re-rank
            reranked_order = np.argsort(-boosted_scores)  # Descending order
            reranked_indices = candidate_indices[reranked_order]

            # Re-ranked Hit@K
            rerank_hit1.append(target_idx == reranked_indices[0])
            rerank_hit5.append(target_idx in reranked_indices[:5])
            rerank_hit10.append(target_idx in reranked_indices[:10])

    # Calculate metrics
    metrics = {
        'baseline_hit1': np.mean(baseline_hit1) * 100,
        'baseline_hit5': np.mean(baseline_hit5) * 100,
        'baseline_hit10': np.mean(baseline_hit10) * 100,
        'rerank_hit1': np.mean(rerank_hit1) * 100,
        'rerank_hit5': np.mean(rerank_hit5) * 100,
        'rerank_hit10': np.mean(rerank_hit10) * 100,
    }

    # Calculate gains
    metrics['gain_hit1'] = metrics['rerank_hit1'] - metrics['baseline_hit1']
    metrics['gain_hit5'] = metrics['rerank_hit5'] - metrics['baseline_hit5']
    metrics['gain_hit10'] = metrics['rerank_hit10'] - metrics['baseline_hit10']

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LVM with TMD re-ranking"
    )
    parser.add_argument(
        '--model',
        type=Path,
        required=True,
        help='Path to trained model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='Path to validation data (.npz file)'
    )
    parser.add_argument(
        '--vectors',
        type=Path,
        required=True,
        help='Path to full vector bank with TMD lanes'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=30,
        help='Number of candidates to retrieve (default: 30)'
    )
    parser.add_argument(
        '--lane-boost',
        type=float,
        default=0.10,
        help='Score boost for same-lane candidates (default: 0.10 = +10%%)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['mps', 'cpu', 'cuda'],
        help='Device to use (default: mps)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Evaluation batch size (default: 32)'
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.model, device=args.device)

    # Load validation data
    val_contexts, val_targets, val_target_indices = load_validation_data(args.data)

    # Load vector bank with lanes
    vector_bank, lane_indices = load_vector_bank_with_lanes(args.vectors)

    # Evaluate with TMD re-ranking
    metrics = evaluate_with_tmd_rerank(
        model=model,
        val_contexts=val_contexts,
        val_targets=val_targets,
        val_target_indices=val_target_indices,  # NEW: Pass target indices
        vector_bank=vector_bank,
        lane_indices=lane_indices,
        device=args.device,
        top_k=args.top_k,
        lane_boost=args.lane_boost,
        batch_size=args.batch_size
    )

    # Print results
    logger.info("\n" + "="*60)
    logger.info("TMD RE-RANKING RESULTS")
    logger.info("="*60)
    logger.info(f"\nBaseline (LVM only):")
    logger.info(f"  Hit@1:  {metrics['baseline_hit1']:.2f}%")
    logger.info(f"  Hit@5:  {metrics['baseline_hit5']:.2f}%")
    logger.info(f"  Hit@10: {metrics['baseline_hit10']:.2f}%")

    logger.info(f"\nWith TMD Re-ranking (top-K={args.top_k}, boost={args.lane_boost:.2f}):")
    logger.info(f"  Hit@1:  {metrics['rerank_hit1']:.2f}% ({metrics['gain_hit1']:+.2f}%)")
    logger.info(f"  Hit@5:  {metrics['rerank_hit5']:.2f}% ({metrics['gain_hit5']:+.2f}%)")
    logger.info(f"  Hit@10: {metrics['rerank_hit10']:.2f}% ({metrics['gain_hit10']:+.2f}%)")

    logger.info(f"\nNet Improvement:")
    logger.info(f"  Hit@1:  {metrics['gain_hit1']:+.2f}%")
    logger.info(f"  Hit@5:  {metrics['gain_hit5']:+.2f}%")
    logger.info(f"  Hit@10: {metrics['gain_hit10']:+.2f}%")

    if metrics['gain_hit5'] >= 2.0:
        logger.info("\n✅ SUCCESS! TMD re-ranking achieved +2% Hit@5 target!")
    elif metrics['gain_hit5'] > 0:
        logger.info(f"\n⚠️ Modest gain (+{metrics['gain_hit5']:.2f}%), but below +2% target")
    else:
        logger.info("\n❌ TMD re-ranking did not improve Hit@5")

    logger.info("="*60)


if __name__ == '__main__':
    main()
