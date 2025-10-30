#!/usr/bin/env python3
"""
Quick LSTM OOD Evaluation

Evaluates the fixed LSTM model on out-of-distribution test data.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.lvm.model import load_lvm_model


def cosine_similarity(pred, target):
    """Compute cosine similarity"""
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    return (pred_norm * target_norm).sum(dim=1).mean().item()


def evaluate_model(model, contexts, targets, device, batch_size=32):
    """Evaluate model on test data"""
    model.eval()

    # Convert to tensors
    if not isinstance(contexts, torch.Tensor):
        contexts = torch.from_numpy(contexts).float()
    if not isinstance(targets, torch.Tensor):
        targets = torch.from_numpy(targets).float()

    contexts = contexts.to(device)
    targets = targets.to(device)

    cosine_scores = []
    mse_scores = []

    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch_ctx = contexts[i:i+batch_size]
            batch_tgt = targets[i:i+batch_size]

            # Forward pass
            pred = model(batch_ctx)

            # Metrics
            cos_sim = cosine_similarity(pred, batch_tgt)
            mse = F.mse_loss(pred, batch_tgt).item()

            cosine_scores.append(cos_sim)
            mse_scores.append(mse)

    return {
        'cosine_mean': np.mean(cosine_scores),
        'cosine_std': np.std(cosine_scores),
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
    }


def main():
    print("=" * 60)
    print("LSTM OOD EVALUATION")
    print("=" * 60)
    print()

    # Paths
    model_path = Path("artifacts/lvm/models/lstm_v0.pt")
    ood_path = Path("artifacts/lvm/wikipedia_ood_test_ctx5.npz")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"OOD Data: {ood_path}")
    print()

    # Load model
    print("📦 Loading LSTM model...")
    model = load_lvm_model("lstm", str(model_path), device)
    print(f"   ✅ Model loaded (params: {sum(p.numel() for p in model.parameters()):,})")
    print()

    # Load OOD data
    print("📊 Loading OOD test data...")
    ood_data = np.load(ood_path, allow_pickle=True)
    ood_ctx = ood_data['context_sequences']
    ood_tgt = ood_data['target_vectors']
    print(f"   ✅ Loaded {len(ood_ctx):,} OOD test sequences")
    print(f"   Context shape: {ood_ctx.shape}")
    print(f"   Target shape: {ood_tgt.shape}")
    print()

    # Evaluate
    print("🔬 Evaluating OOD performance...")
    results = evaluate_model(model, ood_ctx, ood_tgt, device)
    print()

    # Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"OOD Cosine Similarity: {results['cosine_mean']:.4f} ± {results['cosine_std']:.4f}")
    print(f"OOD MSE Loss:          {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
    print()

    # Compare with training val score
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    val_cosine = checkpoint.get('val_cosine', 'N/A')
    print("COMPARISON:")
    print(f"  In-Distribution (Val): {val_cosine:.4f}" if isinstance(val_cosine, float) else f"  In-Distribution (Val): {val_cosine}")
    print(f"  Out-of-Distribution:   {results['cosine_mean']:.4f}")

    if isinstance(val_cosine, float):
        delta = results['cosine_mean'] - val_cosine
        print(f"  Δ (OOD - In-Dist):     {delta:+.4f}")

        if delta > 0:
            print(f"  ✅ Excellent! Model generalizes better to OOD data (+{delta:.4f})")
        elif delta > -0.05:
            print(f"  ✅ Good! Model maintains performance on OOD data ({delta:.4f})")
        else:
            print(f"  ⚠️  Model struggles with OOD data ({delta:.4f})")

    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
