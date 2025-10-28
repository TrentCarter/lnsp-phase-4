#!/usr/bin/env python3
"""
Inspect Positive/Negative Cosine Separation
============================================

Checks if model is learning to distinguish positives from negatives.

Usage:
    python tools/inspect_batch_cosines.py \
        --checkpoint artifacts/lvm/models/mamba_s_pure_infonce/best.pt \
        --sample 2048 \
        --out artifacts/lvm/epoch1_posneg_stats.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.lvm.mamba import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--sample', type=int, default=2048,
                        help='Number of samples to check')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    print("=" * 80)
    print("POSITIVE/NEGATIVE COSINE SEPARATION CHECK")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {args.sample}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model_args = ckpt['args']

    model = create_model(
        model_type=model_args['model_type'],
        d_model=model_args['d_model'],
        n_layers=model_args['n_layers'],
        d_state=model_args.get('d_state', 128),
        conv_sz=model_args.get('conv_sz', 4),
        expand=model_args.get('expand', 2),
        dropout=model_args.get('dropout', 0.1),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"  Model: {model_args['model_type']}")
    print(f"  Epoch: {ckpt['epoch']}")
    print(f"  Val cosine: {ckpt['val_cosine']:.4f}")
    print()

    # Load training data
    print("Loading training data...")
    train_npz = model_args.get('train_npz', 'artifacts/lvm/train_payload_aligned.npz')
    data = np.load(train_npz, allow_pickle=True)
    contexts = data['context_sequences']
    targets = data['target_vectors']

    # Sample
    n_samples = min(args.sample, len(contexts))
    indices = np.random.choice(len(contexts), n_samples, replace=False)
    sample_contexts = torch.from_numpy(contexts[indices]).float()
    sample_targets = torch.from_numpy(targets[indices]).float()

    print(f"  Sampled: {n_samples} sequences")
    print()

    # Generate predictions
    print("Generating predictions...")
    with torch.no_grad():
        preds = model(sample_contexts)
        if len(preds.shape) == 3:
            preds = preds[:, -1, :]

        # Normalize
        preds_norm = F.normalize(preds, p=2, dim=1)
        targets_norm = F.normalize(sample_targets, p=2, dim=1)

    print(f"  Generated {len(preds)} predictions")
    print()

    # Compute cosines
    print("Computing positive/negative cosines...")

    # Positive cosines (pred[i] vs target[i])
    pos_cosines = F.cosine_similarity(preds_norm, targets_norm, dim=1).numpy()

    # Negative cosines (pred[i] vs target[j≠i])
    # Sample 100 negatives per query
    neg_cosines_per_query = []
    for i in range(min(500, len(preds_norm))):  # Limit to 500 for speed
        # Get other targets as negatives
        neg_indices = np.random.choice(
            [j for j in range(len(targets_norm)) if j != i],
            size=min(100, len(targets_norm) - 1),
            replace=False
        )
        neg_targets = targets_norm[neg_indices]

        # Compute cosines
        query = preds_norm[i].unsqueeze(0)  # [1, 768]
        neg_cos = F.cosine_similarity(query, neg_targets, dim=1).numpy()
        neg_cosines_per_query.extend(neg_cos)

    neg_cosines = np.array(neg_cosines_per_query)

    # Statistics
    pos_mean = float(pos_cosines.mean())
    pos_std = float(pos_cosines.std())
    neg_mean = float(neg_cosines.mean())
    neg_std = float(neg_cosines.std())
    separation = pos_mean - neg_mean

    # AUC approximation (assuming normal distributions)
    from scipy import stats

    # Overlap coefficient
    if pos_std + neg_std > 0:
        z = separation / np.sqrt(pos_std**2 + neg_std**2)
        auc = float(stats.norm.cdf(z))
    else:
        auc = 1.0 if separation > 0 else 0.5

    # Results
    results = {
        'checkpoint': str(args.checkpoint),
        'epoch': int(ckpt['epoch']),
        'val_cosine': float(ckpt['val_cosine']),
        'n_samples': int(n_samples),
        'positive_cosines': {
            'mean': pos_mean,
            'std': pos_std,
            'min': float(pos_cosines.min()),
            'max': float(pos_cosines.max()),
            'p50': float(np.percentile(pos_cosines, 50)),
        },
        'negative_cosines': {
            'mean': neg_mean,
            'std': neg_std,
            'min': float(neg_cosines.min()),
            'max': float(neg_cosines.max()),
            'p50': float(np.percentile(neg_cosines, 50)),
        },
        'separation': {
            'delta': separation,
            'auc': auc,
        },
    }

    # Print
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Positive cosines (pred[i] vs target[i]):")
    print(f"  Mean: {pos_mean:.4f} ± {pos_std:.4f}")
    print(f"  Range: [{pos_cosines.min():.4f}, {pos_cosines.max():.4f}]")
    print()
    print(f"Negative cosines (pred[i] vs target[j≠i]):")
    print(f"  Mean: {neg_mean:.4f} ± {neg_std:.4f}")
    print(f"  Range: [{neg_cosines.min():.4f}, {neg_cosines.max():.4f}]")
    print()
    print(f"Separation:")
    print(f"  Δ (pos - neg): {separation:.4f}")
    print(f"  AUC: {auc:.4f}")
    print()

    if separation >= 0.15:
        print("✅ STRONG SEPARATION: Model distinguishing positives from negatives")
    elif separation >= 0.10:
        print("✅ GOOD SEPARATION: Model learning useful geometry")
    elif separation >= 0.05:
        print("⚠️  WEAK SEPARATION: Some signal but borderline")
    else:
        print("❌ NO SEPARATION: Model not learning retrieval task")
        print("   → Pivot to two-tower architecture")

    print("=" * 80)

    # Save
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.out}")


if __name__ == '__main__':
    main()
