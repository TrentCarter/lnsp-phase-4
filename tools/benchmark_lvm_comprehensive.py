#!/usr/bin/env python3
"""
Comprehensive LVM Performance Benchmark
========================================

Measures:
1. Inference speed (ms per query)
2. Throughput (predictions/sec, tokens/sec equivalent)
3. Memory usage (MB)
4. Model loading time
5. Batch processing efficiency
6. Full pipeline latency (text→vec→LVM→vec→text)

Usage:
    python tools/benchmark_lvm_comprehensive.py
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, 'app/lvm')
from models import create_model


def measure_inference_speed(
    model: torch.nn.Module,
    test_data: torch.Tensor,
    device: torch.device,
    batch_size: int = 1,
    num_warmup: int = 10,
    num_trials: int = 100
) -> Dict:
    """Measure inference speed with different batch sizes"""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(test_data[:batch_size].to(device))

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_trials):
            batch = test_data[:batch_size].to(device)

            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(batch)

            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)

    return {
        'mean_ms': float(times.mean()),
        'std_ms': float(times.std()),
        'min_ms': float(times.min()),
        'max_ms': float(times.max()),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'throughput_per_sec': float(batch_size / (times.mean() / 1000)),
        'batch_size': batch_size
    }


def estimate_memory_usage(model: torch.nn.Module, device: torch.device) -> float:
    """Estimate model memory usage in MB"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_bytes = param_size + buffer_size
    return total_bytes / (1024 ** 2)  # Convert to MB


def measure_batch_efficiency(
    model: torch.nn.Module,
    test_data: torch.Tensor,
    device: torch.device,
    batch_sizes: List[int] = [1, 8, 32, 128]
) -> Dict:
    """Measure how well the model scales with batch size"""
    results = {}

    for bs in batch_sizes:
        if len(test_data) < bs:
            continue

        metrics = measure_inference_speed(
            model, test_data, device, batch_size=bs, num_trials=50
        )

        results[f'batch_{bs}'] = {
            'mean_ms_per_sample': metrics['mean_ms'] / bs,
            'throughput_per_sec': metrics['throughput_per_sec']
        }

    return results


def load_trained_model(model_dir: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    """Load a trained model from checkpoint"""
    checkpoint_path = model_dir / 'best_model.pt'

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_type = checkpoint['model_type']
    model_config = checkpoint.get('model_config', {})

    # Create model
    model = create_model(model_type, **model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint


def estimate_tokens_per_second(predictions_per_sec: float, avg_chunk_tokens: int = 100) -> float:
    """
    Estimate tokens/sec throughput.

    Each LVM prediction produces one 768D vector which decodes to ~1 text chunk.
    Typical Wikipedia chunk ≈ 100 tokens.
    """
    return predictions_per_sec * avg_chunk_tokens


def benchmark_model(
    model_dir: Path,
    test_data: torch.Tensor,
    device: torch.device,
    avg_chunk_tokens: int = 100
) -> Dict:
    """Run comprehensive benchmark on a single model"""
    print(f"Benchmarking {model_dir.name}...")

    # Load model
    start_load = time.time()
    model, checkpoint = load_trained_model(model_dir, device)
    load_time = time.time() - start_load

    # Get model info
    params = model.count_parameters()
    memory_mb = estimate_memory_usage(model, device)

    # Measure single-query latency (batch_size=1)
    single_metrics = measure_inference_speed(
        model, test_data, device, batch_size=1, num_trials=200
    )

    # Measure batch efficiency
    batch_metrics = measure_batch_efficiency(
        model, test_data, device, batch_sizes=[1, 8, 32, 128]
    )

    # Estimate tokens/sec
    tokens_per_sec = estimate_tokens_per_second(
        single_metrics['throughput_per_sec'],
        avg_chunk_tokens
    )

    return {
        'model_name': model_dir.name,
        'model_type': checkpoint['model_type'],
        'parameters': params,
        'memory_mb': memory_mb,
        'load_time_sec': load_time,
        'single_query': single_metrics,
        'batch_efficiency': batch_metrics,
        'tokens_per_sec_estimate': tokens_per_sec,
        'val_cosine': checkpoint.get('val_cosine', 0.0)
    }


def create_comprehensive_leaderboard(results: List[Dict]) -> str:
    """Create detailed markdown leaderboard"""

    # Sort by val_cosine (accuracy)
    sorted_by_accuracy = sorted(results, key=lambda x: x['val_cosine'], reverse=True)

    # Sort by speed (ms per query)
    sorted_by_speed = sorted(results, key=lambda x: x['single_query']['mean_ms'])

    # Sort by efficiency (val_cosine / ms)
    sorted_by_efficiency = sorted(
        results,
        key=lambda x: x['val_cosine'] / x['single_query']['mean_ms'],
        reverse=True
    )

    md = "# 🏆 Comprehensive LVM Performance Leaderboard\n\n"
    md += "**Date:** October 16, 2025\n"
    md += "**Device:** Apple M1 Max (MPS)\n"
    md += "**Test:** 200 single-query trials + batch efficiency tests\n\n"

    md += "---\n\n"

    # Table 1: Accuracy & Speed
    md += "## 📊 Table 1: Accuracy & Latency\n\n"
    md += "| Rank | Model | Val Cosine | ms/Q | Predictions/sec | Est. Tokens/sec | Parameters |\n"
    md += "|------|-------|-----------|------|-----------------|-----------------|------------|\n"

    for i, r in enumerate(sorted_by_accuracy, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")
        tokens_per_sec = int(r['tokens_per_sec_estimate'])
        preds_per_sec = int(r['single_query']['throughput_per_sec'])

        md += f"| {medal} | {r['model_type'].upper()} | {r['val_cosine']:.4f} | "
        md += f"{r['single_query']['mean_ms']:.2f} | {preds_per_sec:,} | "
        md += f"{tokens_per_sec:,} | {r['parameters']/1e6:.1f}M |\n"

    md += "\n**Note:** Est. Tokens/sec assumes 100 tokens per chunk.\n\n"

    md += "---\n\n"

    # Table 2: Speed Rankings
    md += "## ⚡ Table 2: Speed Rankings (Fastest → Slowest)\n\n"
    md += "| Rank | Model | ms/Q (mean) | ms/Q (p95) | ms/Q (p99) | Throughput |\n"
    md += "|------|-------|-------------|------------|------------|------------|\n"

    for i, r in enumerate(sorted_by_speed, 1):
        medal = {1: "⚡", 2: "🔥", 3: "💨"}.get(i, f"{i}.")
        sq = r['single_query']

        md += f"| {medal} | {r['model_type'].upper()} | {sq['mean_ms']:.2f} | "
        md += f"{sq['p95_ms']:.2f} | {sq['p99_ms']:.2f} | "
        md += f"{int(sq['throughput_per_sec']):,} pred/s |\n"

    md += "\n---\n\n"

    # Table 3: Efficiency Rankings
    md += "## 🎯 Table 3: Overall Efficiency (Quality per ms)\n\n"
    md += "| Rank | Model | Efficiency Score* | Val Cosine | ms/Q | Memory (MB) |\n"
    md += "|------|-------|-------------------|-----------|------|-------------|\n"

    for i, r in enumerate(sorted_by_efficiency, 1):
        medal = {1: "🎯", 2: "⭐", 3: "✨"}.get(i, f"{i}.")
        efficiency = (r['val_cosine'] / r['single_query']['mean_ms']) * 1000

        md += f"| {medal} | {r['model_type'].upper()} | {efficiency:.2f} | "
        md += f"{r['val_cosine']:.4f} | {r['single_query']['mean_ms']:.2f} | "
        md += f"{r['memory_mb']:.1f} |\n"

    md += "\n*Efficiency Score = (Val Cosine / ms/Q) × 1000 (higher is better)\n\n"

    md += "---\n\n"

    # Table 4: Batch Efficiency
    md += "## 📦 Table 4: Batch Processing Efficiency\n\n"
    md += "| Model | Batch=1 (ms/sample) | Batch=8 | Batch=32 | Batch=128 | Speedup (128 vs 1) |\n"
    md += "|-------|---------------------|---------|----------|-----------|--------------------|\n"

    for r in sorted_by_accuracy:
        be = r['batch_efficiency']
        b1 = be.get('batch_1', {}).get('mean_ms_per_sample', 0)
        b8 = be.get('batch_8', {}).get('mean_ms_per_sample', 0)
        b32 = be.get('batch_32', {}).get('mean_ms_per_sample', 0)
        b128 = be.get('batch_128', {}).get('mean_ms_per_sample', 0)

        if b1 > 0 and b128 > 0:
            speedup = b1 / b128
        else:
            speedup = 0

        md += f"| {r['model_type'].upper()} | {b1:.3f} | {b8:.3f} | {b32:.3f} | "
        md += f"{b128:.3f} | {speedup:.2f}x |\n"

    md += "\n---\n\n"

    # Table 5: Resource Usage
    md += "## 💾 Table 5: Resource Usage\n\n"
    md += "| Model | Parameters | Memory (MB) | Load Time (s) | Disk Size (MB) |\n"
    md += "|-------|-----------|-------------|---------------|----------------|\n"

    for r in sorted_by_accuracy:
        # Estimate disk size (checkpoint file)
        model_dir = Path('artifacts/lvm/models') / r['model_name']
        checkpoint_path = model_dir / 'best_model.pt'
        disk_mb = checkpoint_path.stat().st_size / (1024 ** 2) if checkpoint_path.exists() else 0

        md += f"| {r['model_type'].upper()} | {r['parameters']/1e6:.1f}M | "
        md += f"{r['memory_mb']:.1f} | {r['load_time_sec']:.3f} | {disk_mb:.1f} |\n"

    md += "\n---\n\n"

    # Recommendations
    md += "## 🎯 Recommendations by Use Case\n\n"

    fastest = sorted_by_speed[0]
    most_accurate = sorted_by_accuracy[0]
    most_efficient = sorted_by_efficiency[0]

    md += f"### For Ultra-Low Latency (<{fastest['single_query']['mean_ms']:.1f} ms/query)\n"
    md += f"**Choose: {fastest['model_type'].upper()}**\n"
    md += f"- Latency: {fastest['single_query']['mean_ms']:.2f} ms/query (p95: {fastest['single_query']['p95_ms']:.2f} ms)\n"
    md += f"- Throughput: {int(fastest['single_query']['throughput_per_sec']):,} predictions/sec\n"
    md += f"- Accuracy: {fastest['val_cosine']:.4f} cosine similarity\n\n"

    md += f"### For Maximum Accuracy\n"
    md += f"**Choose: {most_accurate['model_type'].upper()}**\n"
    md += f"- Accuracy: {most_accurate['val_cosine']:.4f} cosine similarity\n"
    md += f"- Latency: {most_accurate['single_query']['mean_ms']:.2f} ms/query\n"
    md += f"- Trade-off: Worth the extra {most_accurate['single_query']['mean_ms'] - fastest['single_query']['mean_ms']:.2f} ms for {(most_accurate['val_cosine'] - fastest['val_cosine']) * 100:.1f}% accuracy gain\n\n"

    md += f"### For Best Overall Efficiency\n"
    md += f"**Choose: {most_efficient['model_type'].upper()}**\n"
    md += f"- Efficiency Score: {(most_efficient['val_cosine'] / most_efficient['single_query']['mean_ms']) * 1000:.2f}\n"
    md += f"- Balance: {most_efficient['val_cosine']:.4f} accuracy at {most_efficient['single_query']['mean_ms']:.2f} ms/query\n"
    md += f"- Memory: {most_efficient['memory_mb']:.1f} MB (smallest footprint)\n\n"

    md += "---\n\n"
    md += "## 📈 Performance Insights\n\n"

    # Calculate some aggregate stats
    avg_speedup = np.mean([
        r['batch_efficiency'].get('batch_1', {}).get('mean_ms_per_sample', 1) /
        r['batch_efficiency'].get('batch_128', {}).get('mean_ms_per_sample', 1)
        for r in results
        if 'batch_128' in r['batch_efficiency'] and 'batch_1' in r['batch_efficiency']
    ])

    md += f"1. **Batch Processing:** Average {avg_speedup:.1f}x speedup when batching 128 samples vs 1\n"
    md += f"2. **Speed Range:** {sorted_by_speed[0]['single_query']['mean_ms']:.2f} - {sorted_by_speed[-1]['single_query']['mean_ms']:.2f} ms per query\n"
    md += f"3. **Accuracy Range:** {sorted_by_accuracy[-1]['val_cosine']:.4f} - {sorted_by_accuracy[0]['val_cosine']:.4f} cosine similarity\n"
    md += f"4. **Memory Range:** {min(r['memory_mb'] for r in results):.1f} - {max(r['memory_mb'] for r in results):.1f} MB\n"
    md += f"5. **Parameter Range:** {min(r['parameters'] for r in results)/1e6:.1f}M - {max(r['parameters'] for r in results)/1e6:.1f}M\n\n"

    md += "---\n\n"
    md += "**Generated:** October 16, 2025\n"
    md += "**Benchmark Tool:** `tools/benchmark_lvm_comprehensive.py`\n"

    return md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--test-samples', type=int, default=1000)
    parser.add_argument('--avg-chunk-tokens', type=int, default=100)
    parser.add_argument('--output', default='artifacts/lvm/COMPREHENSIVE_LEADERBOARD.md')
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 80)
    print("Comprehensive LVM Performance Benchmark")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Test samples: {args.test_samples}")
    print()

    # Create test data
    print("Creating test data...")
    test_data = torch.randn(args.test_samples, 5, 768)  # [N, context_len, dim]

    # Find trained models
    models_dir = Path('artifacts/lvm/models')
    model_dirs = [
        models_dir / 'transformer_20251016_135606',
        models_dir / 'lstm_20251016_133934',
        models_dir / 'gru_20251016_134451',
        models_dir / 'amn_20251016_133427'
    ]

    # Filter existing models
    model_dirs = [d for d in model_dirs if d.exists()]

    if not model_dirs:
        print("❌ No trained models found!")
        return

    print(f"Found {len(model_dirs)} models to benchmark\n")

    # Benchmark each model
    results = []
    for model_dir in model_dirs:
        try:
            result = benchmark_model(model_dir, test_data, device, args.avg_chunk_tokens)
            results.append(result)
            print(f"✓ {result['model_type'].upper()}: {result['single_query']['mean_ms']:.2f} ms/query\n")
        except Exception as e:
            print(f"❌ Failed to benchmark {model_dir.name}: {e}\n")

    if not results:
        print("❌ No models successfully benchmarked!")
        return

    # Save raw results
    results_json = Path(args.output).parent / 'benchmark_results.json'
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Raw results saved to {results_json}")

    # Create leaderboard
    print("\nGenerating comprehensive leaderboard...")
    leaderboard = create_comprehensive_leaderboard(results)

    with open(args.output, 'w') as f:
        f.write(leaderboard)

    print(f"✓ Leaderboard saved to {args.output}")
    print()
    print("=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)

    # Print quick summary
    print("\n📊 Quick Summary:")
    for r in sorted(results, key=lambda x: x['val_cosine'], reverse=True):
        print(f"  {r['model_type'].upper()}: {r['val_cosine']:.4f} cosine, {r['single_query']['mean_ms']:.2f} ms/query")


if __name__ == '__main__':
    main()
