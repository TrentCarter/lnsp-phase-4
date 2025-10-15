#!/usr/bin/env python3
"""
Test GRU Model with Real Examples
===================================

Loads the trained GRU model and tests it on 10 samples:
- Shows input text (context)
- Shows expected output text (ground truth)
- Shows actual output text (GRU prediction)
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.lvm.train_mamba2 import Mamba2VectorPredictor


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained GRU model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = Mamba2VectorPredictor(
        input_dim=768,
        d_model=checkpoint['args']['d_model'],
        num_layers=checkpoint['args']['num_layers']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint


def main():
    print("=" * 80)
    print("GRU Model Inference Test (10 Samples)")
    print("=" * 80)
    print()

    # Load model
    print("Loading GRU model...")
    device = torch.device('cpu')
    model, checkpoint = load_model('artifacts/lvm/models/mamba2/best_model.pt', device)
    print(f"✓ Model loaded (params: {model.count_parameters():,})")
    print(f"✓ Val Cosine: {checkpoint['val_cosine']:.4f}")
    print()

    # Load test data
    print("Loading test data...")
    data = np.load('artifacts/lvm/training_sequences_ctx5.npz')

    contexts = data['context_sequences']  # [N, 5, 768]
    targets = data['target_vectors']      # [N, 768]

    print(f"✓ Loaded {len(contexts)} sequences")
    print()

    # Test on 10 samples
    print("=" * 80)
    print("Testing 10 Random Samples")
    print("=" * 80)
    print()

    # Select 10 random indices
    np.random.seed(42)
    test_indices = np.random.choice(len(contexts), size=10, replace=False)

    total_cosine = 0.0

    for i, idx in enumerate(test_indices, 1):
        print(f"{'='*80}")
        print(f"Sample {i}/10 (Index: {idx})")
        print(f"{'='*80}")

        # Get context and target
        context_vecs = contexts[idx]  # [5, 768]
        target_vec = targets[idx]     # [768]

        print("\n📖 INPUT CONTEXT (5 vectors):")
        for j in range(5):
            vec_norm = np.linalg.norm(context_vecs[j])
            print(f"  {j+1}. Vector norm: {vec_norm:.4f}")

        print("\n✅ TARGET VECTOR:")
        print(f"  Norm: {np.linalg.norm(target_vec):.4f}")

        # Run GRU prediction
        context_tensor = torch.FloatTensor(context_vecs).unsqueeze(0).to(device)  # [1, 5, 768]

        with torch.no_grad():
            predicted_vec = model(context_tensor)  # [1, 768]
            predicted_vec = predicted_vec.cpu().numpy()[0]

        # Compute cosine similarity
        target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-8)
        pred_norm = predicted_vec / (np.linalg.norm(predicted_vec) + 1e-8)
        cosine_sim = np.dot(target_norm, pred_norm)
        total_cosine += cosine_sim

        print(f"\n🔢 VECTOR SIMILARITY:")
        print(f"  Cosine: {cosine_sim:.4f}")
        print(f"  Target norm: {np.linalg.norm(target_vec):.4f}")
        print(f"  Predicted norm: {np.linalg.norm(predicted_vec):.4f}")

        # Quality assessment based on cosine
        if cosine_sim > 0.8:
            quality = "✅ EXCELLENT"
        elif cosine_sim > 0.7:
            quality = "✅ GOOD"
        elif cosine_sim > 0.6:
            quality = "⚠️ OKAY"
        else:
            quality = "❌ POOR"

        print(f"\n📊 QUALITY: {quality}")
        print()

    avg_cosine = total_cosine / len(test_indices)
    print("=" * 80)
    print(f"Test Complete! Average Cosine: {avg_cosine:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
