#!/usr/bin/env python3
"""
Prepare Alignment Head Training Data (Simple)
==============================================

Generates (v_pred, v_true) pairs from existing training sequences.
Uses LVM to predict, then saves pairs for alignment head training.
"""

import numpy as np
from pathlib import Path
import torch


def generate_predictions(model, contexts, batch_size=512):
    """Generate predictions for context sequences."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch = torch.from_numpy(contexts[i:i+batch_size]).float()
            preds = model(batch).numpy()
            predictions.append(preds)

            if (i // batch_size + 1) % 50 == 0:
                print(f"    Progress: {i+batch_size:,} / {len(contexts):,}")

    return np.concatenate(predictions, axis=0)


def main():
    print("=" * 80)
    print("PREPARING ALIGNMENT HEAD TRAINING DATA")
    print("=" * 80)
    print()

    # Load sequences (already split into train/val)
    print("Loading sequences...")
    seq_path = Path("artifacts/lvm/wikipedia_fresh_sequences_ctx5.npz")
    data = np.load(seq_path)

    train_contexts = data["context_sequences"]
    train_targets = data["target_vectors"]
    val_contexts = data["val_context_sequences"]
    val_targets = data["val_target_vectors"]

    print(f"  Train: {len(train_contexts):,} sequences")
    print(f"  Val:   {len(val_contexts):,} sequences\n")

    # Load LVM model
    print("Loading LVM model...")
    import sys
    sys.path.insert(0, 'app/lvm')
    from models import AttentionMixtureNetwork

    model_path = Path("artifacts/lvm/production_model/best_model.pt")
    checkpoint = torch.load(model_path, map_location="cpu")

    model_config = checkpoint["model_config"]
    model = AttentionMixtureNetwork(
        input_dim=model_config["input_dim"],
        d_model=model_config["d_model"],
        hidden_dim=model_config["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded AMN model from {model_path}\n")

    # Generate predictions
    print("Generating train predictions...")
    train_preds = generate_predictions(model, train_contexts)
    print(f"  ✓ Generated {len(train_preds):,} train predictions\n")

    print("Generating val predictions...")
    val_preds = generate_predictions(model, val_contexts)
    print(f"  ✓ Generated {len(val_preds):,} val predictions\n")

    # Subsample to reasonable size (optional, for faster training)
    max_train = 100000
    max_val = 10000

    if len(train_preds) > max_train:
        print(f"Subsampling train to {max_train:,}...")
        indices = np.random.choice(len(train_preds), size=max_train, replace=False)
        train_preds = train_preds[indices]
        train_targets = train_targets[indices]

    if len(val_preds) > max_val:
        print(f"Subsampling val to {max_val:,}...")
        indices = np.random.choice(len(val_preds), size=max_val, replace=False)
        val_preds = val_preds[indices]
        val_targets = val_targets[indices]

    print(f"  Final train: {len(train_preds):,}")
    print(f"  Final val:   {len(val_preds):,}\n")

    # Save
    output_path = Path("artifacts/lvm/alignment_training_data.npz")
    print(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        v_pred_train=train_preds,
        v_true_train=train_targets,
        v_pred_val=val_preds,
        v_true_val=val_targets,
    )

    file_size = output_path.stat().st_size / (1024**2)
    print(f"✓ Saved {file_size:.1f} MB\n")

    # Compute baseline alignment (before training)
    print("Baseline alignment (cosine similarity):")
    train_cos = (train_preds * train_targets).sum(axis=1)  # Already normalized
    val_cos = (val_preds * val_targets).sum(axis=1)
    print(f"  Train mean cosine: {train_cos.mean():.4f}")
    print(f"  Val mean cosine:   {val_cos.mean():.4f}")
    print()

    print("=" * 80)
    print("✓ ALIGNMENT DATA READY!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
