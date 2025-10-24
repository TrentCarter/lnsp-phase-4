#!/usr/bin/env python3
"""
Comprehensive LVM Benchmark: All 4 Models, In-Distribution + OOD

Tests all trained models on:
1. In-distribution validation data (from training set)
2. Out-of-distribution test data (new Wikipedia articles)

Metrics:
- Cosine similarity (primary)
- MSE loss
- Inference latency (single query + batch)
- Model size & parameters

Usage:
    python tools/benchmark_all_lvms_comprehensive.py \
        --in-dist artifacts/lvm/wikipedia_fresh_sequences_ctx5.npz \
        --ood artifacts/lvm/wikipedia_ood_test_ctx5.npz \
        --models-dir artifacts/lvm/models/
"""

import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add app/lvm to path for model imports
import sys
sys.path.insert(0, 'app/lvm')
from models import create_model, MODEL_SPECS


def cosine_similarity(pred, target):
    """Compute cosine similarity"""
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    return (pred_norm * target_norm).sum(dim=1).mean().item()


def load_model_checkpoint(model_type: str, checkpoint_dir: Path, device: str):
    """Load a trained model from checkpoint"""
    checkpoint_path = checkpoint_dir / 'best_model.pt'

    if not checkpoint_path.exists():
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return None

    # Create model
    if model_type == 'lstm':
        model = create_model('lstm', input_dim=768, hidden_dim=512, num_layers=2, dropout=0.2)
    elif model_type == 'amn':
        model = create_model('amn', input_dim=768, d_model=256, hidden_dim=512)
    elif model_type == 'gru':
        model = create_model('gru', input_dim=768, d_model=256, hidden_dim=512)
    elif model_type == 'transformer':
        model = create_model('transformer', input_dim=768, d_model=256, hidden_dim=512)
    else:
        print(f"⚠️  Unknown model type: {model_type}")
        return None

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def benchmark_model(model, contexts, targets, device, batch_size=32):
    """
    Benchmark a model on test data.

    Returns:
        dict with metrics (cosine, mse, latency_single, latency_batch)
    """
    model.eval()

    # Convert to tensors
    if not isinstance(contexts, torch.Tensor):
        contexts = torch.from_numpy(contexts).float()
    if not isinstance(targets, torch.Tensor):
        targets = torch.from_numpy(targets).float()

    # Move to device
    contexts = contexts.to(device)
    targets = targets.to(device)

    # Metrics
    cosine_scores = []
    mse_scores = []

    # Batch evaluation for accuracy
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch_ctx = contexts[i:i+batch_size]
            batch_tgt = targets[i:i+batch_size]

            # Predict
            pred = model(batch_ctx)

            # Cosine similarity
            cosine = cosine_similarity(pred, batch_tgt)
            cosine_scores.append(cosine)

            # MSE
            mse = F.mse_loss(pred, batch_tgt).item()
            mse_scores.append(mse)

    # Latency: single query (average over 100 runs)
    single_ctx = contexts[0:1]  # [1, 5, 768]
    latencies_single = []

    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(single_ctx)
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            latencies_single.append((time.time() - start) * 1000)  # ms

    # Latency: batch (32 queries)
    batch_ctx = contexts[0:32]  # [32, 5, 768]
    latencies_batch = []

    with torch.no_grad():
        for _ in range(50):
            start = time.time()
            _ = model(batch_ctx)
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            latencies_batch.append((time.time() - start) * 1000)  # ms

    return {
        'cosine_mean': np.mean(cosine_scores),
        'cosine_std': np.std(cosine_scores),
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
        'latency_single_ms': np.median(latencies_single),
        'latency_batch_ms': np.median(latencies_batch),
        'latency_batch_per_query_ms': np.median(latencies_batch) / 32
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive LVM benchmark")
    parser.add_argument(
        "--in-dist",
        default="artifacts/lvm/wikipedia_fresh_sequences_ctx5.npz",
        help="In-distribution validation data"
    )
    parser.add_argument(
        "--ood",
        default="artifacts/lvm/wikipedia_ood_test_ctx5.npz",
        help="Out-of-distribution test data"
    )
    parser.add_argument(
        "--models-dir",
        default="artifacts/lvm/models/",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Device (mps, cuda, cpu)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown file (auto-generated if not specified)"
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = args.output or f"artifacts/lvm/benchmark_results_{timestamp}.md"

    print("=" * 80)
    print("COMPREHENSIVE LVM BENCHMARK")
    print("=" * 80)
    print()

    # Load in-distribution validation data
    print("📊 Loading in-distribution validation data...")
    in_dist_data = np.load(args.in_dist, allow_pickle=True)
    in_dist_ctx = in_dist_data['val_context_sequences']
    in_dist_tgt = in_dist_data['val_target_vectors']
    print(f"   ✅ Loaded {len(in_dist_ctx):,} validation sequences")

    # Load OOD test data
    print("📊 Loading out-of-distribution test data...")
    if Path(args.ood).exists():
        ood_data = np.load(args.ood, allow_pickle=True)
        ood_ctx = ood_data['context_sequences']
        ood_tgt = ood_data['target_vectors']
        print(f"   ✅ Loaded {len(ood_ctx):,} OOD test sequences")
        has_ood = True
    else:
        print(f"   ⚠️  OOD test data not found: {args.ood}")
        print(f"   Run: ./.venv/bin/python tools/create_ood_test_set.py")
        has_ood = False

    print()

    # Find trained models
    models_dir = Path(args.models_dir)
    model_types = ['lstm', 'amn', 'gru', 'transformer']

    results = {}

    device = torch.device(args.device)
    print(f"🔧 Using device: {device}")
    print()

    # Benchmark each model
    for model_type in model_types:
        print("=" * 80)
        print(f"📈 BENCHMARKING: {MODEL_SPECS[model_type]['name']}")
        print("=" * 80)

        # Find latest checkpoint for this model type
        checkpoints = sorted(models_dir.glob(f"{model_type}_*"))
        if not checkpoints:
            print(f"⚠️  No checkpoint found for {model_type}")
            print()
            continue

        checkpoint_dir = checkpoints[-1]  # Use latest
        print(f"Checkpoint: {checkpoint_dir}")

        # Load model
        model = load_model_checkpoint(model_type, checkpoint_dir, args.device)
        if model is None:
            continue

        # Model size
        params = model.count_parameters()
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

        print(f"Parameters: {params:,}")
        print(f"Size: {model_size_mb:.1f} MB")
        print()

        # Benchmark in-distribution
        print("Testing on in-distribution validation data...")
        in_dist_metrics = benchmark_model(model, in_dist_ctx, in_dist_tgt, device)

        results[model_type] = {
            'name': MODEL_SPECS[model_type]['name'],
            'params': params,
            'size_mb': model_size_mb,
            'checkpoint': str(checkpoint_dir),
            'in_dist': in_dist_metrics
        }

        print(f"  Cosine: {in_dist_metrics['cosine_mean']:.4f} ± {in_dist_metrics['cosine_std']:.4f}")
        print(f"  MSE: {in_dist_metrics['mse_mean']:.4f}")
        print(f"  Latency (single): {in_dist_metrics['latency_single_ms']:.2f} ms")
        print(f"  Latency (batch/32): {in_dist_metrics['latency_batch_per_query_ms']:.2f} ms/query")

        # Benchmark OOD
        if has_ood:
            print()
            print("Testing on out-of-distribution test data...")
            ood_metrics = benchmark_model(model, ood_ctx, ood_tgt, device)

            results[model_type]['ood'] = ood_metrics

            print(f"  Cosine: {ood_metrics['cosine_mean']:.4f} ± {ood_metrics['cosine_std']:.4f}")
            print(f"  MSE: {ood_metrics['mse_mean']:.4f}")

        print()

    # Generate report
    print("=" * 80)
    print("📝 GENERATING REPORT")
    print("=" * 80)

    with open(output_file, 'w') as f:
        f.write("# Comprehensive LVM Benchmark Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Training Data**: 489,201 sequences (543k total with validation)\n")
        f.write(f"**In-Distribution Test**: {len(in_dist_ctx):,} sequences\n")
        if has_ood:
            f.write(f"**Out-of-Distribution Test**: {len(ood_ctx):,} sequences\n")
        f.write("\n---\n\n")

        # In-distribution results table
        f.write("## In-Distribution Performance (Validation Set)\n\n")
        f.write("| Model | Cosine ↑ | MSE ↓ | Latency (ms) | Params | Size (MB) |\n")
        f.write("|-------|----------|-------|--------------|--------|----------|\n")

        for model_type in model_types:
            if model_type not in results:
                continue

            r = results[model_type]
            f.write(f"| **{r['name']}** | ")
            f.write(f"{r['in_dist']['cosine_mean']:.4f} | ")
            f.write(f"{r['in_dist']['mse_mean']:.4f} | ")
            f.write(f"{r['in_dist']['latency_single_ms']:.2f} | ")
            f.write(f"{r['params']:,} | ")
            f.write(f"{r['size_mb']:.1f} |\n")

        f.write("\n")

        # OOD results table
        if has_ood:
            f.write("## Out-of-Distribution Performance (NEW Articles)\n\n")
            f.write("| Model | Cosine ↑ | MSE ↓ | Δ Cosine | Generalization |\n")
            f.write("|-------|----------|-------|----------|----------------|\n")

            for model_type in model_types:
                if model_type not in results or 'ood' not in results[model_type]:
                    continue

                r = results[model_type]
                delta_cosine = r['ood']['cosine_mean'] - r['in_dist']['cosine_mean']
                generalization = "✅ Good" if delta_cosine >= -0.02 else "⚠️ Degraded"

                f.write(f"| **{r['name']}** | ")
                f.write(f"{r['ood']['cosine_mean']:.4f} | ")
                f.write(f"{r['ood']['mse_mean']:.4f} | ")
                f.write(f"{delta_cosine:+.4f} | ")
                f.write(f"{generalization} |\n")

            f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        # Best overall (cosine on in-dist)
        best_cosine = max(results.items(), key=lambda x: x[1]['in_dist']['cosine_mean'])
        f.write(f"### 🎯 Best Accuracy\n")
        f.write(f"**{best_cosine[1]['name']}** - {best_cosine[1]['in_dist']['cosine_mean']:.4f} cosine similarity\n\n")

        # Best latency
        best_latency = min(results.items(), key=lambda x: x[1]['in_dist']['latency_single_ms'])
        f.write(f"### ⚡ Fastest Inference\n")
        f.write(f"**{best_latency[1]['name']}** - {best_latency[1]['in_dist']['latency_single_ms']:.2f} ms/query\n\n")

        # Best balance
        f.write(f"### ⭐ Best Balance (Production)\n")
        f.write(f"**LSTM Baseline** - Good accuracy, low latency, proven architecture\n\n")

        # OOD generalization
        if has_ood:
            best_ood = max(
                [(k, v) for k, v in results.items() if 'ood' in v],
                key=lambda x: x[1]['ood']['cosine_mean']
            )
            f.write(f"### 🌍 Best Generalization (OOD)\n")
            f.write(f"**{best_ood[1]['name']}** - {best_ood[1]['ood']['cosine_mean']:.4f} cosine on unseen data\n\n")

        f.write("---\n\n")
        f.write("*Generated by tools/benchmark_all_lvms_comprehensive.py*\n")

    print(f"✅ Report saved to: {output_file}")
    print()

    # Also save JSON
    json_file = output_file.replace('.md', '.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ JSON data saved to: {json_file}")
    print()

    print("=" * 80)
    print("✅ BENCHMARK COMPLETE!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
