#!/usr/bin/env python3
"""
Cascade Retrieval Evaluation: FAISS → LVM → TMD

Three-stage pipeline:
1. Stage-1 (FAISS): Fast recall engine, retrieve top-K₀ candidates
2. Stage-2 (LVM): Neural re-ranker, score K₀ → keep top-K₁
3. Stage-3 (TMD): Semantic control, boost same-lane candidates

This aligns with Phase-3's training regime (small candidate set re-ranking).
"""

import argparse
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from app.lvm.models import MemoryAugmentedGRU

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: str = 'cpu'):
    """Load trained LVM model."""
    logger.info(f"Loading LVM model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Model config
    config = checkpoint.get('config', {})
    d_model = config.get('input_dim', 768)
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

    logger.info(f"✓ LVM loaded: {d_model}D → {hidden_dim}D hidden ({num_layers} layers)")

    return model


def load_validation_data(data_path: Path):
    """Load validation sequences."""
    logger.info(f"Loading validation data from {data_path}")

    data = np.load(data_path, allow_pickle=True)
    contexts = torch.from_numpy(data['context_sequences']).float()
    targets = torch.from_numpy(data['target_vectors']).float()

    logger.info(f"✓ Loaded {len(contexts)} validation sequences")

    return contexts, targets


def load_vector_bank(vectors_path: Path):
    """Load vector bank with TMD lanes."""
    logger.info(f"Loading vector bank from {vectors_path}")

    data = np.load(vectors_path, allow_pickle=True)
    vectors = torch.from_numpy(data['vectors']).float()
    tmd_lanes = data['tmd_lanes']

    # Convert lane strings to indices
    lane_to_idx = {f"lane_{i}": i for i in range(16)}
    lane_indices = np.array([lane_to_idx.get(lane, 0) for lane in tmd_lanes])

    logger.info(f"✓ Loaded {len(vectors)} vectors")
    logger.info(f"  Lane distribution: {Counter(tmd_lanes).most_common(3)}")

    return vectors, lane_indices


def get_context_lane(context_vectors, vector_bank, lane_indices):
    """Determine TMD lane of context (majority vote of nearest neighbors)."""
    context_norm = F.normalize(context_vectors, dim=1)
    bank_norm = F.normalize(vector_bank, dim=1)

    similarities = torch.mm(context_norm, bank_norm.t())
    nearest_indices = similarities.argmax(dim=1).cpu().numpy()

    context_lanes = lane_indices[nearest_indices]
    dominant_lane = Counter(context_lanes).most_common(1)[0][0]

    return dominant_lane


@torch.no_grad()
def evaluate_cascade(
    lvm_model,
    val_contexts,
    val_targets,
    vector_bank,
    lane_indices,
    device='cpu',
    k0_values=[100, 200, 500, 1000],
    k1=50,
    tmd_weight=0.3,
    batch_size=32
):
    """
    Evaluate cascade retrieval with multiple K₀ values.

    Args:
        lvm_model: Trained LVM re-ranker
        val_contexts: [N, context_len, D] validation contexts
        val_targets: [N, D] validation targets
        vector_bank: [M, D] full vector bank
        lane_indices: [M] TMD lane for each vector
        device: 'cpu' or 'mps'
        k0_values: List of K₀ (FAISS recall) values to test
        k1: K₁ (LVM re-rank top-N)
        tmd_weight: Weight for TMD lane boost (0.0-1.0)
        batch_size: Evaluation batch size

    Returns:
        results: Dict of {k0: {hit@1, hit@5, hit@10}}
    """
    lvm_model.eval()
    vector_bank = vector_bank.to(device)
    bank_norm = F.normalize(vector_bank, dim=1)

    num_samples = len(val_contexts)
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Compute target indices (ground truth)
    logger.info("Computing target indices...")
    targets_on_device = val_targets.to(device)
    target_norm = F.normalize(targets_on_device, dim=1)
    similarities = torch.mm(target_norm, bank_norm.t())
    target_indices = similarities.argmax(dim=1).cpu().numpy()

    logger.info(f"✓ Computed {len(target_indices)} target indices")

    # Results storage
    results = {}
    for k0 in k0_values:
        results[k0] = {
            'stage1_hit1': [],
            'stage1_hit5': [],
            'stage1_hit10': [],
            'stage2_hit1': [],
            'stage2_hit5': [],
            'stage2_hit10': [],
            'stage3_hit1': [],
            'stage3_hit5': [],
            'stage3_hit10': [],
        }

    logger.info(f"\nEvaluating cascade with K₀ ∈ {k0_values}, K₁={k1}, TMD weight={tmd_weight}")

    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        batch_contexts = val_contexts[start_idx:end_idx].to(device)
        batch_target_ids = target_indices[start_idx:end_idx]

        # LVM prediction
        lvm_predictions = lvm_model(batch_contexts)
        lvm_pred_norm = F.normalize(lvm_predictions, dim=1)

        # For each sample in batch
        for i in range(len(batch_contexts)):
            target_idx = batch_target_ids[i]

            # Get context TMD lane
            context_lane = get_context_lane(
                batch_contexts[i].cpu(),
                vector_bank.cpu(),
                lane_indices
            )

            # Test each K₀ value
            for k0 in k0_values:
                # === STAGE 1: FAISS Recall ===
                # (In real implementation, this would be FAISS IVF search)
                # Here we simulate with full cosine search
                faiss_scores = torch.mm(lvm_pred_norm[i:i+1], bank_norm.t()).squeeze()
                faiss_topk_scores, faiss_topk_indices = faiss_scores.topk(k0)

                faiss_candidates = faiss_topk_indices.cpu().numpy()

                # Stage-1 metrics
                results[k0]['stage1_hit1'].append(target_idx == faiss_candidates[0])
                results[k0]['stage1_hit5'].append(target_idx in faiss_candidates[:5])
                results[k0]['stage1_hit10'].append(target_idx in faiss_candidates[:10])

                # === STAGE 2: LVM Re-ranking ===
                # Score K₀ candidates with LVM (already have predictions)
                # In this simplified version, we use the same scores (FAISS = LVM cosine)
                # In production, you could re-score with a different LVM or use attention
                lvm_scores = faiss_topk_scores.cpu().numpy()

                # Keep top-K₁
                if k0 > k1:
                    top_k1_order = np.argsort(-lvm_scores)[:k1]
                    k1_candidates = faiss_candidates[top_k1_order]
                    k1_scores = lvm_scores[top_k1_order]
                else:
                    k1_candidates = faiss_candidates
                    k1_scores = lvm_scores

                # Stage-2 metrics
                results[k0]['stage2_hit1'].append(target_idx == k1_candidates[0])
                results[k0]['stage2_hit5'].append(target_idx in k1_candidates[:5])
                results[k0]['stage2_hit10'].append(target_idx in k1_candidates[:10])

                # === STAGE 3: TMD Re-ranking ===
                # Boost candidates in same TMD lane
                candidate_lanes = lane_indices[k1_candidates]
                lane_matches = (candidate_lanes == context_lane).astype(float)

                # Combined score: (1 - w) * LVM + w * TMD_boost
                tmd_boosted_scores = (1 - tmd_weight) * k1_scores + tmd_weight * lane_matches

                # Re-rank
                tmd_order = np.argsort(-tmd_boosted_scores)
                final_candidates = k1_candidates[tmd_order]

                # Stage-3 metrics
                results[k0]['stage3_hit1'].append(target_idx == final_candidates[0])
                results[k0]['stage3_hit5'].append(target_idx in final_candidates[:5])
                results[k0]['stage3_hit10'].append(target_idx in final_candidates[:10])

    # Calculate metrics
    summary = {}
    for k0 in k0_values:
        summary[k0] = {
            'stage1': {
                'hit@1': np.mean(results[k0]['stage1_hit1']) * 100,
                'hit@5': np.mean(results[k0]['stage1_hit5']) * 100,
                'hit@10': np.mean(results[k0]['stage1_hit10']) * 100,
            },
            'stage2': {
                'hit@1': np.mean(results[k0]['stage2_hit1']) * 100,
                'hit@5': np.mean(results[k0]['stage2_hit5']) * 100,
                'hit@10': np.mean(results[k0]['stage2_hit10']) * 100,
            },
            'stage3': {
                'hit@1': np.mean(results[k0]['stage3_hit1']) * 100,
                'hit@5': np.mean(results[k0]['stage3_hit5']) * 100,
                'hit@10': np.mean(results[k0]['stage3_hit10']) * 100,
            }
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FAISS → LVM → TMD cascade retrieval"
    )
    parser.add_argument(
        '--model',
        type=Path,
        default=Path('artifacts/lvm/models_phase3/run_1000ctx_pilot/best_val_hit5.pt'),
        help='Path to trained LVM model'
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('artifacts/lvm/data_phase3/validation_phase3_exact.npz'),
        help='Path to validation data'
    )
    parser.add_argument(
        '--vectors',
        type=Path,
        default=Path('artifacts/wikipedia_637k_phase3_vectors.npz'),
        help='Path to vector bank'
    )
    parser.add_argument(
        '--k0-values',
        type=int,
        nargs='+',
        default=[100, 200, 500, 1000],
        help='K₀ values (FAISS recall) to test'
    )
    parser.add_argument(
        '--k1',
        type=int,
        default=50,
        help='K₁ (LVM re-rank top-N)'
    )
    parser.add_argument(
        '--tmd-weight',
        type=float,
        default=0.3,
        help='TMD lane boost weight (0.0-1.0)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'mps', 'cuda'],
        help='Device to use'
    )

    args = parser.parse_args()

    # Load components
    lvm_model = load_model(args.model, device=args.device)
    val_contexts, val_targets = load_validation_data(args.data)
    vector_bank, lane_indices = load_vector_bank(args.vectors)

    # Evaluate cascade
    results = evaluate_cascade(
        lvm_model=lvm_model,
        val_contexts=val_contexts,
        val_targets=val_targets,
        vector_bank=vector_bank,
        lane_indices=lane_indices,
        device=args.device,
        k0_values=args.k0_values,
        k1=args.k1,
        tmd_weight=args.tmd_weight
    )

    # Print results
    logger.info("\n" + "="*80)
    logger.info("CASCADE RETRIEVAL RESULTS")
    logger.info("="*80)
    logger.info(f"Configuration: K₁={args.k1}, TMD weight={args.tmd_weight}")
    logger.info("")

    for k0 in args.k0_values:
        logger.info(f"\n📊 K₀ = {k0} (FAISS recall)")
        logger.info("-" * 80)

        logger.info(f"  Stage-1 (FAISS only):")
        logger.info(f"    Hit@1:  {results[k0]['stage1']['hit@1']:.2f}%")
        logger.info(f"    Hit@5:  {results[k0]['stage1']['hit@5']:.2f}%")
        logger.info(f"    Hit@10: {results[k0]['stage1']['hit@10']:.2f}%")

        logger.info(f"  Stage-2 (FAISS → LVM):")
        logger.info(f"    Hit@1:  {results[k0]['stage2']['hit@1']:.2f}% " +
                   f"({results[k0]['stage2']['hit@1'] - results[k0]['stage1']['hit@1']:+.2f}%)")
        logger.info(f"    Hit@5:  {results[k0]['stage2']['hit@5']:.2f}% " +
                   f"({results[k0]['stage2']['hit@5'] - results[k0]['stage1']['hit@5']:+.2f}%)")
        logger.info(f"    Hit@10: {results[k0]['stage2']['hit@10']:.2f}% " +
                   f"({results[k0]['stage2']['hit@10'] - results[k0]['stage1']['hit@10']:+.2f}%)")

        logger.info(f"  Stage-3 (FAISS → LVM → TMD):")
        logger.info(f"    Hit@1:  {results[k0]['stage3']['hit@1']:.2f}% " +
                   f"({results[k0]['stage3']['hit@1'] - results[k0]['stage2']['hit@1']:+.2f}%)")
        logger.info(f"    Hit@5:  {results[k0]['stage3']['hit@5']:.2f}% " +
                   f"({results[k0]['stage3']['hit@5'] - results[k0]['stage2']['hit@5']:+.2f}%)")
        logger.info(f"    Hit@10: {results[k0]['stage3']['hit@10']:.2f}% " +
                   f"({results[k0]['stage3']['hit@10'] - results[k0]['stage2']['hit@10']:+.2f}%)")

    # Find best K₀
    best_k0 = max(args.k0_values, key=lambda k: results[k]['stage3']['hit@5'])
    logger.info("\n" + "="*80)
    logger.info(f"🏆 Best K₀: {best_k0} → {results[best_k0]['stage3']['hit@5']:.2f}% Hit@5")
    logger.info("="*80)


if __name__ == '__main__':
    main()
