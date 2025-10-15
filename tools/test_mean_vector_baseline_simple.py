#!/usr/bin/env python3
"""
Task 1.2: Mean Vector Baseline Test (Simple version - no imports)
==================================================================

Tests whether model has collapsed to outputting the global mean vector.
"""

import numpy as np
import torch
import torch.nn as nn


# Copy model architecture inline to avoid import issues
class Mamba2Block(nn.Module):
    """Simplified Mamba-2 block"""
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model * 4)
        self.out_proj = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.proj(x)
        x = nn.functional.gelu(x)
        x = self.out_proj(x)
        return x + residual


class Mamba2VectorPredictor(nn.Module):
    """Mamba-2 model for next-vector prediction"""
    def __init__(self, input_dim=768, d_model=512, num_layers=4):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            Mamba2Block(d_model) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, input_dim),
        )

    def forward(self, x, return_raw: bool = False):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        last_hidden = x[:, -1, :]
        raw = self.head(last_hidden)
        cos = nn.functional.normalize(raw, p=2, dim=-1)
        if return_raw:
            return raw, cos
        return cos


def load_model(checkpoint_path, device='cpu'):
    """Load trained model"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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
    """Compute cosine similarity"""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a_norm, b_norm)


def main():
    print("=" * 80)
    print("Task 1.2: Mean Vector Baseline Test")
    print("=" * 80)
    print()

    # Load data
    print("Loading training data...")
    data = np.load('artifacts/lvm/training_sequences_ctx5_sentence.npz')
    contexts = data['context_sequences']
    targets = data['target_vectors']
    print(f"✓ Loaded {len(contexts)} sequences")
    print()

    # Compute global mean
    print("Computing global mean vector...")
    global_mean = targets.mean(axis=0)
    print(f"Global mean norm: {np.linalg.norm(global_mean):.4f}")
    print()

    # Test on 100 samples
    print("Testing on 100 random samples...")
    np.random.seed(42)
    test_indices = np.random.choice(len(contexts), size=100, replace=False)

    # Baseline: target vs global mean
    baseline_cosines = []
    for idx in test_indices:
        target = targets[idx]
        cosine = cosine_similarity_np(target, global_mean)
        baseline_cosines.append(cosine)

    baseline_mean = np.mean(baseline_cosines)
    print(f"\nBASELINE (target vs global mean): {baseline_mean:.4f}")

    # Load model
    print("\nLoading GRU model...")
    device = torch.device('cpu')
    model, checkpoint = load_model('artifacts/lvm/models/mamba2/best_model.pt', device)
    print(f"✓ Model loaded (training val_cosine: {checkpoint['val_cosine']:.4f})")

    # Model predictions
    print("\nRunning model predictions...")
    model_cosines = []
    pred_to_mean_cosines = []

    for idx in test_indices:
        context_vecs = contexts[idx]
        target_vec = targets[idx]

        context_tensor = torch.FloatTensor(context_vecs).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_vec = model(context_tensor).cpu().numpy()[0]

        cosine_pred_target = cosine_similarity_np(predicted_vec, target_vec)
        model_cosines.append(cosine_pred_target)

        cosine_pred_mean = cosine_similarity_np(predicted_vec, global_mean)
        pred_to_mean_cosines.append(cosine_pred_mean)

    model_mean = np.mean(model_cosines)
    pred_to_mean_mean = np.mean(pred_to_mean_cosines)

    print(f"\nMODEL (prediction vs target): {model_mean:.4f}")
    print(f"MODEL (prediction vs global mean): {pred_to_mean_mean:.4f}")

    # Diversity check
    print("\nDiversity check (10 predictions)...")
    predictions = []
    for idx in test_indices[:10]:
        context_vecs = contexts[idx]
        context_tensor = torch.FloatTensor(context_vecs).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_vec = model(context_tensor).cpu().numpy()[0]
        predictions.append(predicted_vec)

    pairwise_cosines = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            cosine = cosine_similarity_np(predictions[i], predictions[j])
            pairwise_cosines.append(cosine)

    diversity_mean = np.mean(pairwise_cosines)
    print(f"Pairwise similarity: {diversity_mean:.4f}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    diff = model_mean - baseline_mean
    improvement_pct = (diff / baseline_mean) * 100

    print(f"\nModel improvement over baseline: {diff:+.4f} ({improvement_pct:+.1f}%)")
    print(f"Predictions are {pred_to_mean_mean:.1%} similar to global mean")
    print(f"Predictions are {diversity_mean:.1%} similar to each other")

    if abs(diff) < 0.02 and diversity_mean > 0.85:
        print("\n🔥 COMPLETE MODE COLLAPSE!")
        print("   Model outputs nearly identical vectors = global mean")
    elif abs(diff) < 0.05:
        print("\n⚠️  WEAK PERFORMANCE (partial collapse)")
    else:
        print("\n✅ MODEL IS LEARNING")

    print()


if __name__ == '__main__':
    main()
