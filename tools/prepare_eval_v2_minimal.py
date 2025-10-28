#!/usr/bin/env python3
"""
Minimal V2 Evaluation Prep - NO PAYLOAD NEEDED
===============================================

Creates v2 NPZ without building the memory-hungry payload.
Payload not needed for metrics calculation!
"""

import numpy as np
import torch
import torch.nn as nn
import sys

sys.path.insert(0, 'app/lvm')
from models import AttentionMixtureNetwork


def main():
    print("=" * 80)
    print("MINIMAL V2 PREP (No Payload)")
    print("=" * 80)
    print()

    # Load AMN
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[1/3] Loading AMN model (device: {device})...")

    checkpoint = torch.load("artifacts/lvm/production_model/best_model.pt", map_location=device)
    hparams = checkpoint.get('hyperparameters', {})

    model = AttentionMixtureNetwork(
        input_dim=hparams.get('input_dim', 768),
        d_model=hparams.get('d_model', 256),
        hidden_dim=hparams.get('hidden_dim', 512)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    print("  ✅ Model loaded\n")

    # Load OOD test
    print("[2/3] Loading OOD test set...")
    ood = np.load("artifacts/lvm/wikipedia_ood_test_ctx5_fresh.npz", allow_pickle=True)
    contexts = ood['context_sequences']
    metadata = ood['metadata']
    print(f"  ✅ Loaded {len(contexts):,} sequences\n")

    # Run inference
    print("[3/3] Running AMN inference...")
    pred_vecs = []
    batch_size = 128  # Larger batch for speed

    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i:i+batch_size]
            batch_t = torch.from_numpy(batch).float().to(device)
            preds = model(batch_t).cpu().numpy()
            preds = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8)
            pred_vecs.append(preds)

            if (i + batch_size) % 2560 == 0:
                print(f"  Progress: {min(i + batch_size, len(contexts)):,} / {len(contexts):,}")

    pred_vecs = np.vstack(pred_vecs).astype(np.float32)
    print(f"  ✅ Complete: {pred_vecs.shape}\n")

    # Build metadata (fast)
    print("Building metadata arrays...")
    last_meta = np.array([{
        "article_index": int(m['article_index']),
        "chunk_index": int(m['last_chunk_index'])
    } for m in metadata], dtype=object)

    truth_keys = np.array([[
        int(m['article_index']),
        int(m['target_chunk_index'])
    ] for m in metadata], dtype=np.int32)

    # Save
    print("\n💾 Saving v2 NPZ...")
    np.savez_compressed(
        "artifacts/lvm/wikipedia_ood_test_ctx5_v2_fresh.npz",
        pred_vecs=pred_vecs,
        last_meta=last_meta,
        truth_keys=truth_keys
    )

    print("\n" + "=" * 80)
    print("✅ V2 NPZ READY!")
    print("=" * 80)
    print(f"\nSamples: {len(pred_vecs):,}")
    print("\nNOTE: Payload not created (not needed for metrics)")
    print("You can run evaluation with --no-payload flag")
    print()


if __name__ == "__main__":
    main()
