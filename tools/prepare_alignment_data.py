#!/usr/bin/env python3
"""
Prepare Alignment Head Training Data
=====================================

Extracts (v_pred, v_true) pairs from training sequences for alignment head.

Balances by article to avoid overfitting to frequent pages.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
import torch


def main():
    print("=" * 80)
    print("PREPARING ALIGNMENT HEAD TRAINING DATA")
    print("=" * 80)
    print()

    # Load training sequences (context → target pairs)
    print("Loading training sequences...")
    seq_path = Path("artifacts/lvm/training_sequences_ctx5.npz")
    data = np.load(seq_path, allow_pickle=True)

    contexts = data["contexts"]  # [N, ctx_len, 768]
    targets = data["targets"]    # [N, 768]
    metadata = data["metadata"]  # [N] list of dicts

    N = len(targets)
    print(f"  Loaded {N:,} training sequences\n")

    # Load LVM model for prediction
    print("Loading LVM model...")
    import sys
    sys.path.insert(0, 'app/lvm')
    from models import LSTMVectorPredictor

    model_path = Path("artifacts/lvm/production_model/checkpoint_epoch_100.pt")
    checkpoint = torch.load(model_path, map_location="cpu")

    model = LSTMVectorPredictor(
        input_dim=768,
        hidden_dim=checkpoint.get("hidden_dim", 512),
        num_layers=checkpoint.get("num_layers", 2),
        output_dim=768,
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded model from {model_path}\n")

    # Generate predictions
    print("Generating predictions...")
    v_preds = []
    v_trues = []
    article_ids = []

    batch_size = 512
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_ctx = torch.from_numpy(contexts[i:i+batch_size]).float()
            batch_pred = model(batch_ctx).numpy()

            v_preds.append(batch_pred)
            v_trues.append(targets[i:i+batch_size])

            # Extract article IDs for balancing
            for meta in metadata[i:i+batch_size]:
                article_ids.append(int(meta["article_index"]))

            if (i // batch_size + 1) % 50 == 0:
                print(f"  Progress: {i+batch_size:,} / {N:,}")

    v_preds = np.concatenate(v_preds, axis=0)
    v_trues = np.concatenate(v_trues, axis=0)
    article_ids = np.array(article_ids)

    print(f"\n✓ Generated {len(v_preds):,} (v_pred, v_true) pairs\n")

    # Balance by article (avoid overfitting to frequent pages)
    print("Balancing by article...")
    article_to_indices = defaultdict(list)
    for i, article_id in enumerate(article_ids):
        article_to_indices[article_id].append(i)

    # Sample up to 100 pairs per article
    max_per_article = 100
    balanced_indices = []
    for article_id, indices in article_to_indices.items():
        if len(indices) <= max_per_article:
            balanced_indices.extend(indices)
        else:
            # Sample without replacement
            sampled = np.random.choice(indices, size=max_per_article, replace=False)
            balanced_indices.extend(sampled)

    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)

    v_preds_balanced = v_preds[balanced_indices]
    v_trues_balanced = v_trues[balanced_indices]

    print(f"  Original: {len(v_preds):,} pairs from {len(article_to_indices):,} articles")
    print(f"  Balanced: {len(v_preds_balanced):,} pairs (max {max_per_article} per article)\n")

    # Split into train/val (90/10)
    split_idx = int(0.9 * len(v_preds_balanced))
    indices = np.arange(len(v_preds_balanced))
    np.random.shuffle(indices)

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    # Save
    output_path = Path("artifacts/lvm/alignment_training_data.npz")
    print(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        v_pred_train=v_preds_balanced[train_idx],
        v_true_train=v_trues_balanced[train_idx],
        v_pred_val=v_preds_balanced[val_idx],
        v_true_val=v_trues_balanced[val_idx],
    )

    file_size = output_path.stat().st_size / (1024**2)
    print(f"✓ Saved {file_size:.1f} MB\n")

    print("Dataset statistics:")
    print(f"  Train: {len(train_idx):,} pairs")
    print(f"  Val:   {len(val_idx):,} pairs")
    print()

    print("=" * 80)
    print("✓ ALIGNMENT DATA READY!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
