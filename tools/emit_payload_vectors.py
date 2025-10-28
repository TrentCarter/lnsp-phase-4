#!/usr/bin/env python3
"""
Emit payload vectors from P tower.

Usage:
    python3 tools/emit_payload_vectors.py \
        --checkpoint artifacts/lvm/models/twotower_mamba_s/epoch1.pt \
        --npz artifacts/lvm/eval_clean_disjoint.npz \
        --out artifacts/corpus/p_ep1.npy \
        --device cpu

Override --npz if you intentionally need an alternative split (e.g. a
quarantined leaked eval file).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.lvm.train_twotower import PayloadTower


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--npz', type=Path, required=True,
                        help='NPZ with target_vectors')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()

    print("=" * 80)
    print("EMIT PAYLOAD VECTORS (P TOWER)")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"NPZ: {args.npz}")
    print(f"Device: {args.device}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Get architecture params
    ckpt_args = checkpoint['args']

    # Create P tower
    p_tower = PayloadTower(
        backbone_type=ckpt_args.get('arch_p', 'mamba_s'),
        d_model=ckpt_args.get('d_model', 768),
    ).to(args.device)

    p_tower.load_state_dict(checkpoint['p_tower_state_dict'])
    p_tower.eval()
    print(f"  Loaded P tower (epoch {checkpoint['epoch']})")
    print()

    # Load target vectors (corpus)
    print("Loading target vectors...")
    data = np.load(args.npz, allow_pickle=True)
    targets = data['target_vectors']
    print(f"  Corpus size: {len(targets)}")
    print()

    # Encode
    print("Encoding corpus...")
    all_p = []

    with torch.no_grad():
        for i in range(0, len(targets), args.batch_size):
            batch = torch.from_numpy(targets[i:i+args.batch_size]).float().to(args.device)
            p = p_tower(batch)
            all_p.append(p.cpu().numpy())

            if (i // args.batch_size) % 10 == 0:
                print(f"  {i}/{len(targets)}")

    all_p = np.concatenate(all_p, axis=0)
    print(f"  Generated: {all_p.shape}")
    print()

    # Verify normalization
    norms = np.linalg.norm(all_p, axis=1)
    print(f"Norm check: mean={norms.mean():.6f}, std={norms.std():.6f}")
    if abs(norms.mean() - 1.0) > 0.01:
        print("  ⚠️  Vectors not L2-normalized! Normalizing now...")
        all_p = all_p / (np.linalg.norm(all_p, axis=1, keepdims=True) + 1e-12)
        norms = np.linalg.norm(all_p, axis=1)
        print(f"  After norm: mean={norms.mean():.6f}, std={norms.std():.6f}")
    else:
        print("  ✅ Vectors L2-normalized")
    print()

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, all_p)
    print(f"✅ Saved: {args.out}")
    print(f"   Shape: {all_p.shape}")
    print("=" * 80)


if __name__ == '__main__':
    main()
