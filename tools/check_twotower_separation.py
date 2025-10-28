#!/usr/bin/env python3
"""
Check cosine separation on val mini-batch for two-tower model.

Usage:
    python3 tools/check_twotower_separation.py \
        --checkpoint artifacts/lvm/models/twotower_mamba_s/epoch1.pt \
        --eval-npz artifacts/lvm/eval_clean_disjoint.npz \
        --n-samples 256 \
        --device cpu

Pass --eval-npz explicitly if you need to run against a different split
(for example a quarantined leaked eval set).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.lvm.train_twotower import QueryTower, PayloadTower


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--eval-npz', type=Path, required=True)
    parser.add_argument('--n-samples', type=int, default=256)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print("=" * 80)
    print("TWO-TOWER SEPARATION CHECK")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {args.n_samples}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_args = checkpoint['args']

    # Create towers
    q_tower = QueryTower(
        backbone_type=ckpt_args.get('arch_q', 'mamba_s'),
        d_model=ckpt_args.get('d_model', 768),
        n_layers=ckpt_args.get('n_layers', 8),
        d_state=ckpt_args.get('d_state', 128),
        conv_sz=ckpt_args.get('conv_sz', 4),
        expand=ckpt_args.get('expand', 2),
        dropout=ckpt_args.get('dropout', 0.1),
    ).to(args.device)

    p_tower = PayloadTower(
        backbone_type=ckpt_args.get('arch_p', 'mamba_s'),
        d_model=ckpt_args.get('d_model', 768),
    ).to(args.device)

    q_tower.load_state_dict(checkpoint['q_tower_state_dict'])
    p_tower.load_state_dict(checkpoint['p_tower_state_dict'])
    q_tower.eval()
    p_tower.eval()
    print(f"  Loaded epoch {checkpoint['epoch']}")
    print()

    # Load eval data
    print("Loading eval data...")
    data = np.load(args.eval_npz, allow_pickle=True)
    contexts = data['context_sequences']
    targets = data['target_vectors']

    # Sample
    n = min(args.n_samples, len(contexts))
    indices = np.random.choice(len(contexts), n, replace=False)
    contexts = contexts[indices]
    targets = targets[indices]
    print(f"  Sampled: {n}")
    print()

    # Encode
    print("Encoding...")
    with torch.no_grad():
        ctx_tensor = torch.from_numpy(contexts).float().to(args.device)
        tgt_tensor = torch.from_numpy(targets).float().to(args.device)

        q = q_tower(ctx_tensor)
        p = p_tower(tgt_tensor)

        q = q.cpu().numpy()
        p = p.cpu().numpy()
    print()

    # Compute cosines
    print("Computing cosine separation...")

    # Positive cosines (q[i] vs p[i])
    pos_cosines = np.sum(q * p, axis=1)
    pos_mean = pos_cosines.mean()
    pos_std = pos_cosines.std()

    # Negative cosines (q[i] vs p[j≠i])
    neg_cosines = []
    for i in range(min(100, len(q))):
        neg_indices = [j for j in range(len(p)) if j != i]
        if len(neg_indices) > 50:
            neg_indices = np.random.choice(neg_indices, 50, replace=False)
        neg_cos = np.dot(q[i], p[neg_indices].T)
        neg_cosines.extend(neg_cos)

    neg_mean = np.mean(neg_cosines)
    neg_std = np.std(neg_cosines)
    separation = pos_mean - neg_mean

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Samples: {n}")
    print()
    print(f"Positive cosines (q[i] vs p[i]):")
    print(f"  Mean: {pos_mean:.4f} ± {pos_std:.4f}")
    print()
    print(f"Negative cosines (q[i] vs p[j≠i]):")
    print(f"  Mean: {neg_mean:.4f} ± {neg_std:.4f}")
    print()
    print(f"Separation (Δ): {separation:.4f}")
    print()

    if separation > 0.1:
        print(f"✅ GOOD: Δ = {separation:.4f} > 0.1 (should see non-zero R@5)")
    elif separation > 0.05:
        print(f"⚠️  WEAK: Δ = {separation:.4f} (marginal separation)")
    else:
        print(f"❌ FAIL: Δ = {separation:.4f} ≤ 0.05 (no separation)")

    print("=" * 80)


if __name__ == '__main__':
    main()
