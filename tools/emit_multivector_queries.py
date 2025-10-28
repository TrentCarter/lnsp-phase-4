#!/usr/bin/env python3
"""
Emit multi-vector queries (one per context chunk, no compression).

Instead of compressing 5 chunks → 1 vector, emit all 5 vectors.
MaxSim scoring at retrieval time: max_i(q_i · p)

Usage:
    python3 tools/emit_multivector_queries.py \
        --checkpoint artifacts/lvm/models/twotower_mamba_s/epoch2.pt \
        --eval-npz artifacts/lvm/eval_clean_disjoint.npz \
        --out artifacts/eval/q_multivec.npy \
        --device cpu
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
    parser.add_argument('--eval-npz', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    print("=" * 80)
    print("EMIT MULTI-VECTOR QUERIES (No Compression)")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Eval NPZ: {args.eval_npz}")
    print(f"Strategy: Emit all 5 context vectors (no Q-tower compression)")
    print()

    # Load checkpoint to get P tower (we'll use it to encode context chunks)
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_args = checkpoint['args']

    # Use P tower to encode context chunks (same encoding as payload)
    p_tower = PayloadTower(
        backbone_type=ckpt_args.get('arch_p', 'mamba_s'),
        d_model=ckpt_args.get('d_model', 768),
    ).to(args.device)
    p_tower.load_state_dict(checkpoint['p_tower_state_dict'])
    p_tower.eval()
    print(f"  Loaded P tower (epoch {checkpoint['epoch']})")
    print()

    # Load eval contexts
    print("Loading eval contexts...")
    data = np.load(args.eval_npz, allow_pickle=True)
    contexts = data['context_sequences']  # [N, 5, 768]
    print(f"  Contexts: {contexts.shape}")
    print()

    # Encode each context chunk separately
    print("Encoding multi-vector queries...")
    all_queries = []

    with torch.no_grad():
        for i in range(0, len(contexts), args.batch_size):
            batch_contexts = contexts[i:i+args.batch_size]  # [B, 5, 768]
            B, K, D = batch_contexts.shape

            # Reshape to [B*K, 768] to encode all chunks at once
            flat_chunks = batch_contexts.reshape(B * K, D)
            flat_chunks_tensor = torch.from_numpy(flat_chunks).float().to(args.device)

            # Encode with P tower
            q_vecs = p_tower(flat_chunks_tensor)  # [B*K, 768]

            # Reshape back to [B, K, 768]
            q_vecs = q_vecs.cpu().numpy().reshape(B, K, D)
            all_queries.append(q_vecs)

            if (i // args.batch_size) % 10 == 0:
                print(f"  {i}/{len(contexts)}")

    all_queries = np.concatenate(all_queries, axis=0)  # [N, 5, 768]
    print(f"  Generated: {all_queries.shape}")
    print()

    # Verify normalization
    norms = np.linalg.norm(all_queries, axis=2)
    print(f"Norm check: mean={norms.mean():.6f}, std={norms.std():.6f}")
    print()

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, all_queries)
    print(f"✅ Saved: {args.out}")
    print(f"   Shape: {all_queries.shape} (N queries × 5 vectors × 768D)")
    print("=" * 80)


if __name__ == '__main__':
    main()
