#!/usr/bin/env python3
"""
Emit query vectors from Q tower.

Usage:
    python3 tools/emit_query_vectors.py \
        --checkpoint artifacts/lvm/models/twotower_mamba_s/epoch1.pt \
        --eval-npz artifacts/lvm/eval_clean_disjoint.npz \
        --out artifacts/eval/q_ep1.npy \
        --device cpu

Override --eval-npz if you intentionally need to target a different split
(for example a quarantined leaked eval file).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.lvm.train_twotower import QueryTower


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--eval-npz', type=Path, required=True,
                        help='NPZ with context_sequences')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    print("=" * 80)
    print("EMIT QUERY VECTORS (Q TOWER)")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Eval NPZ: {args.eval_npz}")
    print(f"Device: {args.device}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Get architecture params
    ckpt_args = checkpoint['args']

    # Create Q tower
    q_tower = QueryTower(
        backbone_type=ckpt_args.get('arch_q', 'mamba_s'),
        d_model=ckpt_args.get('d_model', 768),
        n_layers=ckpt_args.get('n_layers', 8),
        d_state=ckpt_args.get('d_state', 128),
        conv_sz=ckpt_args.get('conv_sz', 4),
        expand=ckpt_args.get('expand', 2),
        dropout=ckpt_args.get('dropout', 0.1),
    ).to(args.device)

    q_tower.load_state_dict(checkpoint['q_tower_state_dict'])
    q_tower.eval()
    print(f"  Loaded Q tower (epoch {checkpoint['epoch']})")
    print()

    # Load eval contexts
    print("Loading eval contexts...")
    data = np.load(args.eval_npz, allow_pickle=True)
    contexts = data['context_sequences']
    print(f"  Eval size: {len(contexts)}")
    print()

    # Encode
    print("Encoding queries...")
    all_q = []

    with torch.no_grad():
        for i in range(0, len(contexts), args.batch_size):
            batch = torch.from_numpy(contexts[i:i+args.batch_size]).float().to(args.device)
            q = q_tower(batch)
            all_q.append(q.cpu().numpy())

            if (i // args.batch_size) % 10 == 0:
                print(f"  {i}/{len(contexts)}")

    all_q = np.concatenate(all_q, axis=0)
    print(f"  Generated: {all_q.shape}")
    print()

    # Verify normalization
    norms = np.linalg.norm(all_q, axis=1)
    print(f"Norm check: mean={norms.mean():.6f}, std={norms.std():.6f}")
    if abs(norms.mean() - 1.0) > 0.01:
        print("  ⚠️  Vectors not L2-normalized! Normalizing now...")
        all_q = all_q / (np.linalg.norm(all_q, axis=1, keepdims=True) + 1e-12)
        norms = np.linalg.norm(all_q, axis=1)
        print(f"  After norm: mean={norms.mean():.6f}, std={norms.std():.6f}")
    else:
        print("  ✅ Vectors L2-normalized")
    print()

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, all_q)
    print(f"✅ Saved: {args.out}")
    print(f"   Shape: {all_q.shape}")
    print("=" * 80)


if __name__ == '__main__':
    main()
