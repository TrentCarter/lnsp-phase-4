#!/usr/bin/env python3
"""
Task 1.2: Mean Vector Baseline Test
====================================

Tests whether model has collapsed to outputting the global mean vector.

If model_cosine ≈ baseline_cosine, the model learned nothing but the mean.

Expected Result:
- Global mean baseline: ~75-80% cosine
- Collapsed model: Also ~75-80% cosine (same as baseline)
- Good model: >80% cosine (better than baseline)
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
    """Load trained model"""
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


def cosine_similarity_np(a, b):
    """Compute cosine similarity between vectors"""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a_norm, b_norm)


def main():
    print("=" * 80)
    print("Task 1.2: Mean Vector Baseline Test")
    print("=" * 80)
    print()

    # Load training data
    print("Loading training data...")
    data = np.load('artifacts/lvm/training_sequences_ctx5.npz')

    contexts = data['context_sequences']  # [N, 5, 768]
    targets = data['target_vectors']      # [N, 768]

    print(f"✓ Loaded {len(contexts)} sequences")
    print()

    # Compute global mean of all targets
    print("Computing global mean vector...")
    global_mean = targets.mean(axis=0)  # [768]
    global_mean_norm = np.linalg.norm(global_mean)

    print(f"Global mean statistics:")
    print(f"  Norm: {global_mean_norm:.4f}")
    print(f"  Mean: {global_mean.mean():.6f}")
    print(f"  Std: {global_mean.std():.6f}")
    print()

    # Test on random sample
    print("=" * 80)
    print("Testing on 100 random samples")
    print("=" * 80)
    print()

    np.random.seed(42)
    test_indices = np.random.choice(len(contexts), size=100, replace=False)

    # Baseline: cosine between each target and global mean
    baseline_cosines = []
    for idx in test_indices:
        target = targets[idx]
        cosine = cosine_similarity_np(target, global_mean)
        baseline_cosines.append(cosine)

    baseline_mean = np.mean(baseline_cosines)
    baseline_std = np.std(baseline_cosines)

    print(f"BASELINE (target vs global mean):")
    print(f"  Average cosine: {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"  Min: {np.min(baseline_cosines):.4f}")
    print(f"  Max: {np.max(baseline_cosines):.4f}")
    print()

    # Load model
    print("Loading GRU model...")
    device = torch.device('cpu')
    model, checkpoint = load_model('artifacts/lvm/models/mamba2/best_model.pt', device)
    print(f"✓ Model loaded (val_cosine from training: {checkpoint['val_cosine']:.4f})")
    print()

    # Model predictions
    print("Running model predictions...")
    model_cosines = []
    prediction_to_mean_cosines = []

    for idx in test_indices:
        context_vecs = contexts[idx]  # [5, 768]
        target_vec = targets[idx]     # [768]

        # Model prediction
        context_tensor = torch.FloatTensor(context_vecs).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_vec = model(context_tensor).cpu().numpy()[0]

        # Cosine: prediction vs target
        cosine_pred_target = cosine_similarity_np(predicted_vec, target_vec)
        model_cosines.append(cosine_pred_target)

        # Cosine: prediction vs global mean
        cosine_pred_mean = cosine_similarity_np(predicted_vec, global_mean)
        prediction_to_mean_cosines.append(cosine_pred_mean)

    model_mean = np.mean(model_cosines)
    model_std = np.std(model_cosines)
    pred_to_mean_mean = np.mean(prediction_to_mean_cosines)
    pred_to_mean_std = np.std(prediction_to_mean_cosines)

    print(f"MODEL (prediction vs target):")
    print(f"  Average cosine: {model_mean:.4f} ± {model_std:.4f}")
    print(f"  Min: {np.min(model_cosines):.4f}")
    print(f"  Max: {np.max(model_cosines):.4f}")
    print()

    print(f"MODEL (prediction vs global mean):")
    print(f"  Average cosine: {pred_to_mean_mean:.4f} ± {pred_to_mean_std:.4f}")
    print(f"  Min: {np.min(prediction_to_mean_cosines):.4f}")
    print(f"  Max: {np.max(prediction_to_mean_cosines):.4f}")
    print()

    # Analysis
    print("=" * 80)
    print("Analysis")
    print("=" * 80)
    print()

    diff = model_mean - baseline_mean
    improvement_pct = (diff / baseline_mean) * 100

    print(f"Model improvement over baseline: {diff:+.4f} ({improvement_pct:+.1f}%)")
    print()

    # Check for mode collapse
    collapse_threshold = 0.02  # If within 2% of baseline, likely collapsed

    if abs(diff) < collapse_threshold:
        print("🔥 MODE COLLAPSE DETECTED!")
        print(f"  Model performance ({model_mean:.4f}) ≈ Baseline ({baseline_mean:.4f})")
        print(f"  The model has learned nothing but to output the mean vector!")
        print()
        print("Evidence:")
        print(f"  - Predictions are {pred_to_mean_mean:.1%} similar to global mean")
        print(f"  - This is only {diff:.4f} better than just using mean")
        print()
        print("Root Cause: MSE loss encourages predicting the mean")
    elif diff < 0.05:
        print("⚠️ WEAK PERFORMANCE")
        print(f"  Model only {improvement_pct:.1f}% better than baseline")
        print(f"  This suggests partial mode collapse or poor training")
    else:
        print("✅ GOOD PERFORMANCE")
        print(f"  Model is {improvement_pct:.1f}% better than baseline")
        print(f"  Model learned meaningful patterns beyond just the mean")

    print()

    # Diversity check
    print("=" * 80)
    print("Diversity Check")
    print("=" * 80)
    print()

    # Get 10 predictions and check pairwise similarity
    test_sample_indices = test_indices[:10]
    predictions = []

    for idx in test_sample_indices:
        context_vecs = contexts[idx]
        context_tensor = torch.FloatTensor(context_vecs).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_vec = model(context_tensor).cpu().numpy()[0]
        predictions.append(predicted_vec)

    predictions = np.array(predictions)

    # Compute pairwise cosine similarities
    pairwise_cosines = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            cosine = cosine_similarity_np(predictions[i], predictions[j])
            pairwise_cosines.append(cosine)

    diversity_mean = np.mean(pairwise_cosines)
    diversity_std = np.std(pairwise_cosines)

    print(f"Pairwise cosine similarity between predictions:")
    print(f"  Average: {diversity_mean:.4f} ± {diversity_std:.4f}")
    print(f"  Min: {np.min(pairwise_cosines):.4f}")
    print(f"  Max: {np.max(pairwise_cosines):.4f}")
    print()

    if diversity_mean > 0.95:
        print("🔥 SEVERE MODE COLLAPSE!")
        print(f"  Predictions are {diversity_mean:.1%} identical to each other")
        print("  Model outputs nearly the same vector for all inputs")
    elif diversity_mean > 0.85:
        print("⚠️ HIGH SIMILARITY")
        print(f"  Predictions are {diversity_mean:.1%} similar")
        print("  Model lacks diversity in outputs")
    else:
        print("✅ GOOD DIVERSITY")
        print(f"  Predictions vary appropriately ({diversity_mean:.1%} similarity)")

    print()
    print("=" * 80)
    print("Conclusion")
    print("=" * 80)
    print()

    if abs(diff) < collapse_threshold and diversity_mean > 0.85:
        print("DIAGNOSIS: Complete mode collapse to mean vector")
        print()
        print("RECOMMENDED FIXES:")
        print("  1. Replace MSE loss with InfoNCE contrastive loss")
        print("  2. Add variance regularization (penalize low std)")
        print("  3. Use diverse batch sampling (avoid near-duplicates)")
        print("  4. Strengthen projection head (2-layer with LayerNorm)")
    elif abs(diff) < 0.05:
        print("DIAGNOSIS: Weak training, possible partial collapse")
        print()
        print("RECOMMENDED FIXES:")
        print("  1. Check training data quality")
        print("  2. Verify loss function (MSE may be too weak)")
        print("  3. Increase model capacity or training time")
    else:
        print("DIAGNOSIS: Model is learning meaningful patterns")
        print()
        print("Continue with current approach, monitor for collapse")

    print()


if __name__ == '__main__':
    main()
