#!/usr/bin/env python3
"""
Generic OOD Evaluation for Any LVM Model

Evaluates any LVM model on out-of-distribution test data.

Usage:
    ./.venv/bin/python tools/eval_model_ood.py \
        --model artifacts/lvm/models/amn_790k_20251030_110346/best_model.pt
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Evaluate LVM model on OOD data")
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--ood-data', default='artifacts/lvm/wikipedia_ood_test_ctx5.npz',
                        help='OOD test data file')
    parser.add_argument('--device', default='mps', help='Device (mps, cuda, cpu)')
    args = parser.parse_args()

    model_path = Path(args.model)
    ood_path = Path(args.ood_data)

    print("=" * 60)
    print("LVM OOD EVALUATION")
    print("=" * 60)
    print()

    # Determine device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"OOD Data: {ood_path}")
    print()

    # Load checkpoint to determine model type
    print("📦 Loading model checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_type = checkpoint.get('model_type', 'unknown')
    val_cosine = checkpoint.get('val_cosine', 'N/A')

    print(f"   Model Type: {model_type}")
    print(f"   In-Dist Val Cosine: {val_cosine:.4f}" if isinstance(val_cosine, float) else f"   In-Dist Val Cosine: {val_cosine}")
    print()

    # Load model
    print(f"📦 Loading {model_type.upper()} model...")
    model = load_lvm_model(model_type, str(model_path), device)
    params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model loaded (params: {params:,})")
    print()

    # Load OOD data
    if not ood_path.exists():
        print(f"❌ ERROR: OOD test data not found: {ood_path}")
        sys.exit(1)

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
