#!/usr/bin/env python3
"""
Benchmark Today's Trained Models (Oct 23, 2025)

Quick benchmark script for the 4 models trained today.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import time
import torch.nn.functional as F
from datetime import datetime

sys.path.insert(0, 'app/lvm')
from models import create_model

# Today's model directories
MODELS = {
    'lstm': 'artifacts/lvm/models/lstm_20251023_202152',
    'amn': 'artifacts/lvm/models/amn_20251023_204747',
    'gru': 'artifacts/lvm/models/gru_20251023_211205',
    'transformer': 'artifacts/lvm/models/transformer_20251023_221917'
}

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
    print("BENCHMARK TODAY'S LVM MODELS (Oct 23, 2025)")
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

    results = {}

    for model_type, model_dir in MODELS.items():
        print("="*80)
        print(f"📈 {model_type.upper()}")
        print("="*80)

        checkpoint_path = Path(model_dir) / 'best_model.pt'
        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            continue

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model using saved config
        model_config = checkpoint.get('model_config', {})
        model = create_model(model_type, **model_config)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        params = model.count_parameters()
        print(f"Parameters: {params:,}")

        # Benchmark in-distribution
        print("Testing in-distribution...")
        in_metrics = benchmark_model(model, in_dist_ctx, in_dist_tgt, device)
        print(f"  Cosine: {in_metrics['cosine_mean']:.4f} ± {in_metrics['cosine_std']:.4f}")
        print(f"  MSE: {in_metrics['mse_mean']:.4f}")
        print(f"  Latency: {in_metrics['latency_ms']:.2f} ms")

        # Benchmark OOD
        print("Testing out-of-distribution...")
        ood_metrics = benchmark_model(model, ood_ctx, ood_tgt, device)
        delta = ood_metrics['cosine_mean'] - in_metrics['cosine_mean']
        print(f"  Cosine: {ood_metrics['cosine_mean']:.4f} (Δ {delta:+.4f})")
        print()

        results[model_type] = {
            'params': params,
            'in_dist': in_metrics,
            'ood': ood_metrics,
            'delta_cosine': delta
        }

    # Summary
    print("="*80)
    print("📊 SUMMARY")
    print("="*80)
    print()
    print("In-Distribution Performance:")
    print(f"{'Model':<15} {'Cosine':<10} {'MSE':<10} {'Latency':<12} {'Params'}")
    print("-"*65)
    for model_type in ['lstm', 'amn', 'gru', 'transformer']:
        if model_type not in results:
            continue
        r = results[model_type]
        print(f"{model_type.upper():<15} {r['in_dist']['cosine_mean']:.4f}     "
              f"{r['in_dist']['mse_mean']:.4f}     "
              f"{r['in_dist']['latency_ms']:.2f} ms      "
              f"{r['params']:,}")

    print()
    print("Out-of-Distribution Performance:")
    print(f"{'Model':<15} {'Cosine':<10} {'Δ Cosine':<12} {'Status'}")
    print("-"*55)
    for model_type in ['lstm', 'amn', 'gru', 'transformer']:
        if model_type not in results:
            continue
        r = results[model_type]
        status = "✅ Good" if r['delta_cosine'] >= -0.03 else "⚠️ Degraded"
        print(f"{model_type.upper():<15} {r['ood']['cosine_mean']:.4f}     "
              f"{r['delta_cosine']:+.4f}       {status}")

    print()
    print("="*80)

    # Find best
    best_in_dist = max(results.items(), key=lambda x: x[1]['in_dist']['cosine_mean'])
    best_ood = max(results.items(), key=lambda x: x[1]['ood']['cosine_mean'])
    fastest = min(results.items(), key=lambda x: x[1]['in_dist']['latency_ms'])

    print("🏆 WINNERS:")
    print(f"   Best In-Distribution: {best_in_dist[0].upper()} ({best_in_dist[1]['in_dist']['cosine_mean']:.4f})")
    print(f"   Best OOD: {best_ood[0].upper()} ({best_ood[1]['ood']['cosine_mean']:.4f})")
    print(f"   Fastest: {fastest[0].upper()} ({fastest[1]['in_dist']['latency_ms']:.2f} ms)")
    print("="*80)

if __name__ == "__main__":
    main()
