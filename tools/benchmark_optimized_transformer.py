#!/usr/bin/env python3
"""
Benchmark Optimized Transformer
================================

Test the optimized Transformer (with LR warmup + cosine annealing) on both
in-distribution and out-of-distribution test sets.

Compare against baseline Transformer:
- Baseline in-dist: 0.5774
- Baseline OOD: 0.6214
"""

import torch
import numpy as np
from pathlib import Path
import sys
import time
import torch.nn.functional as F

sys.path.insert(0, 'app/lvm')
from models import create_model

# Model paths
OPTIMIZED_MODEL = 'artifacts/lvm/models/transformer_optimized_20251024_072726'

def cosine_similarity(pred, target):
    """Compute cosine similarity"""
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    return (pred_norm * target_norm).sum(dim=1).mean().item()

def benchmark_model(model, contexts, targets, device, batch_size=32):
    """Benchmark a model"""
    model.eval()

    contexts = torch.from_numpy(contexts).float().to(device)
    targets = torch.from_numpy(targets).float().to(device)

    cosine_scores = []
    mse_scores = []

    # Accuracy
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch_ctx = contexts[i:i+batch_size]
            batch_tgt = targets[i:i+batch_size]

            pred = model(batch_ctx)

            cosine = cosine_similarity(pred, batch_tgt)
            cosine_scores.append(cosine)

            mse = F.mse_loss(pred, batch_tgt).item()
            mse_scores.append(mse)

    # Latency (single query)
    single_ctx = contexts[0:1]
    latencies = []

    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(single_ctx)
            if device.type == 'mps':
                torch.mps.synchronize()
            latencies.append((time.time() - start) * 1000)

    return {
        'cosine_mean': np.mean(cosine_scores),
        'cosine_std': np.std(cosine_scores),
        'mse_mean': np.mean(mse_scores),
        'latency_ms': np.median(latencies)
    }

def main():
    print("="*80)
    print("BENCHMARK OPTIMIZED TRANSFORMER")
    print("="*80)
    print()

    # Load data
    print("📊 Loading test data...")
    in_dist = np.load('artifacts/lvm/wikipedia_fresh_sequences_ctx5.npz')
    in_dist_ctx = in_dist['val_context_sequences']
    in_dist_tgt = in_dist['val_target_vectors']
    print(f"   In-distribution: {len(in_dist_ctx):,} sequences")

    ood = np.load('artifacts/lvm/wikipedia_ood_test_ctx5.npz')
    ood_ctx = ood['context_sequences']
    ood_tgt = ood['target_vectors']
    print(f"   Out-of-distribution: {len(ood_ctx):,} sequences")
    print()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    print()

    # Load optimized model
    print("="*80)
    print("📈 OPTIMIZED TRANSFORMER (Warmup + Cosine Annealing)")
    print("="*80)

    checkpoint_path = Path(OPTIMIZED_MODEL) / 'best_model.pt'
    if not checkpoint_path.exists():
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model using saved config
    model_config = checkpoint.get('model_config', {})
    model = create_model('transformer', **model_config)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    params = model.count_parameters()
    print(f"Parameters: {params:,}")
    print(f"Training val cosine: {checkpoint['val_cosine']:.4f}")
    print()

    # Benchmark in-distribution
    print("Testing in-distribution...")
    in_metrics = benchmark_model(model, in_dist_ctx, in_dist_tgt, device)
    print(f"  Cosine: {in_metrics['cosine_mean']:.4f} ± {in_metrics['cosine_std']:.4f}")
    print(f"  MSE: {in_metrics['mse_mean']:.4f}")
    print(f"  Latency: {in_metrics['latency_ms']:.2f} ms")
    print()

    # Benchmark OOD
    print("Testing out-of-distribution...")
    ood_metrics = benchmark_model(model, ood_ctx, ood_tgt, device)
    delta = ood_metrics['cosine_mean'] - in_metrics['cosine_mean']
    print(f"  Cosine: {ood_metrics['cosine_mean']:.4f} (Δ {delta:+.4f})")
    print()

    # Summary
    print("="*80)
    print("📊 COMPARISON WITH BASELINE")
    print("="*80)
    print()
    print("Baseline Transformer (ReduceLROnPlateau):")
    print("  In-dist:  0.5774")
    print("  OOD:      0.6214")
    print()
    print("Optimized Transformer (Warmup + Cosine Annealing):")
    print(f"  In-dist:  {in_metrics['cosine_mean']:.4f}  ({in_metrics['cosine_mean'] - 0.5774:+.4f})")
    print(f"  OOD:      {ood_metrics['cosine_mean']:.4f}  ({ood_metrics['cosine_mean'] - 0.6214:+.4f})")
    print(f"  Latency:  {in_metrics['latency_ms']:.2f} ms")
    print()

    # Verdict
    in_improvement = in_metrics['cosine_mean'] - 0.5774
    ood_improvement = ood_metrics['cosine_mean'] - 0.6214

    print("="*80)
    print("🏆 VERDICT")
    print("="*80)
    if in_improvement > 0.001 or ood_improvement > 0.001:
        print("✅ OPTIMIZATION SUCCESSFUL!")
        print()
        if in_improvement > 0.001:
            print(f"   In-dist improved by {in_improvement*100:.2f}%")
        if ood_improvement > 0.001:
            print(f"   OOD improved by {ood_improvement*100:.2f}%")
        print()
        print("   Consultant's suggestions (LR warmup + cosine decay) worked! 🎉")
    else:
        print("⚠️  MIXED RESULTS")
        print()
        print(f"   In-dist change: {in_improvement*100:+.2f}%")
        print(f"   OOD change: {ood_improvement*100:+.2f}%")
        print()
        print("   Optimizations showed minimal impact on this dataset.")

    print("="*80)

if __name__ == "__main__":
    main()
