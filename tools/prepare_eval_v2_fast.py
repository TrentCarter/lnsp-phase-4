#!/usr/bin/env python3
"""
Fast V2 Evaluation Data Preparation
====================================

Uses metadata already in OOD test set - no FAISS search needed!
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, 'app/lvm')
from models import AttentionMixtureNetwork


def main():
    print("=" * 80)
    print("FAST V2 EVALUATION PREP")
    print("=" * 80)
    print()

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load AMN model
    print("[1/5] Loading AMN model...")
    checkpoint = torch.load("artifacts/lvm/production_model/best_model.pt", map_location=device)
    hparams = checkpoint.get('hyperparameters', {})

    model = AttentionMixtureNetwork(
        input_dim=hparams.get('input_dim', 768),
        d_model=hparams.get('d_model', 256),
        hidden_dim=hparams.get('hidden_dim', 512)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("  ✅ Model loaded\n")

    # Load OOD test data (already has metadata!)
    print("[2/5] Loading OOD test set...")
    ood = np.load("artifacts/lvm/wikipedia_ood_test_ctx5_fresh.npz", allow_pickle=True)
    contexts = ood['context_sequences']
    metadata = ood['metadata']
    print(f"  ✅ Loaded {len(contexts):,} sequences\n")

    # Run AMN inference
    print("[3/5] Running AMN inference...")
    pred_vecs = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i:i+batch_size]
            batch_t = torch.from_numpy(batch).float().to(device)
            preds = model(batch_t).cpu().numpy()
            preds = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8)
            pred_vecs.append(preds)

            if (i // batch_size + 1) % 20 == 0:
                print(f"  Progress: {i + len(batch):,} / {len(contexts):,}")

    pred_vecs = np.vstack(pred_vecs).astype(np.float32)
    print(f"  ✅ Inference complete: {pred_vecs.shape}\n")

    # Build last_meta and truth_keys from existing metadata
    print("[4/5] Building metadata from OOD test set...")
    last_meta = []
    truth_keys = []

    for meta in metadata:
        last_meta.append({
            "article_index": int(meta['article_index']),
            "chunk_index": int(meta['last_chunk_index'])
        })
        truth_keys.append([
            int(meta['article_index']),
            int(meta['target_chunk_index'])
        ])

    last_meta = np.array(last_meta, dtype=object)
    truth_keys = np.array(truth_keys, dtype=np.int32)
    print(f"  ✅ Metadata ready for {len(last_meta):,} sequences\n")

    # Build payload
    print("[5/5] Building payload...")
    wiki_data = np.load("artifacts/wikipedia_584k_fresh.npz", allow_pickle=True)

    payload = {}
    for idx in range(len(wiki_data['vectors'])):
        meta = {
            "article_index": int(wiki_data['article_indices'][idx]),
            "chunk_index": int(wiki_data['chunk_indices'][idx]),
            "cpe_id": str(wiki_data['cpe_ids'][idx])
        }
        payload[idx] = (
            wiki_data['concept_texts'][idx],
            meta,
            wiki_data['vectors'][idx]
        )

        if (idx + 1) % 100000 == 0:
            print(f"  Progress: {idx + 1:,} / {len(wiki_data['vectors']):,}")

    print(f"  ✅ Payload built with {len(payload):,} entries\n")

    # Save outputs
    print("💾 Saving outputs...")
    np.savez_compressed(
        "artifacts/lvm/wikipedia_ood_test_ctx5_v2_fresh.npz",
        pred_vecs=pred_vecs,
        last_meta=last_meta,
        truth_keys=truth_keys
    )
    print("  ✅ Saved v2 NPZ")

    np.save("artifacts/wikipedia_584k_payload.npy", payload, allow_pickle=True)
    print("  ✅ Saved payload\n")

    print("=" * 80)
    print("✅ V2 EVALUATION DATA READY!")
    print("=" * 80)
    print(f"\nTest samples: {len(pred_vecs):,}")
    print("\nNext: Run baseline and improved evaluations")
    print()


if __name__ == "__main__":
    main()
