#!/usr/bin/env python3
"""
Compare LVM Model Results
==========================

Load and compare training results from all 4 LVM architectures.

Usage:
    python tools/compare_lvm_models.py

Output:
    - Comparison table of all models
    - Best model recommendation
    - Performance vs baseline analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def find_model_dirs() -> Dict[str, Path]:
    """Find latest trained model directories for each architecture"""
    base_dir = Path('artifacts/lvm/models')

    model_types = ['amn', 'lstm', 'gru', 'transformer']
    latest_dirs = {}

    for model_type in model_types:
        # Find all directories for this model type
        matching = sorted(base_dir.glob(f'{model_type}_*'))
        if matching:
            # Use the latest one (last in sorted order)
            latest_dirs[model_type] = matching[-1]

    return latest_dirs


def load_model_results(model_dir: Path) -> Dict:
    """Load training history from a model directory"""
    history_file = model_dir / 'training_history.json'

    if not history_file.exists():
        return None

    with open(history_file) as f:
        data = json.load(f)

    return data


def compute_stats(history: List[Dict]) -> Tuple[float, float, float, float]:
    """Compute final and best statistics from training history"""
    if not history:
        return 0.0, 0.0, 0.0, 0.0

    final_epoch = history[-1]
    best_val_cosine = max(h['val_cosine'] for h in history)
    best_train_cosine = max(h['train_cosine'] for h in history)

    return (
        final_epoch['val_cosine'],
        final_epoch['train_cosine'],
        best_val_cosine,
        best_train_cosine
    )


def main():
    print("=" * 80)
    print("LVM Model Comparison")
    print("=" * 80)
    print()

    # Find model directories
    model_dirs = find_model_dirs()

    if not model_dirs:
        print("❌ No trained models found in artifacts/lvm/models/")
        print("   Run tools/train_all_lvms.sh first")
        return

    print(f"Found {len(model_dirs)} trained models:")
    for model_type, path in model_dirs.items():
        print(f"  - {model_type}: {path.name}")
    print()

    # Load all results
    results = {}
    for model_type, path in model_dirs.items():
        data = load_model_results(path)
        if data:
            results[model_type] = data

    if not results:
        print("❌ Could not load any training results")
        return

    # Create comparison table
    print("=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print()

    # Header
    print(f"{'Model':<15} {'Params':<12} {'Final Val':<12} {'Best Val':<12} {'Final Train':<12}")
    print(f"{'':15} {'':12} {'Cosine':<12} {'Cosine':<12} {'Cosine':<12}")
    print("-" * 80)

    # Baseline reference
    BASELINE_LINEAR_AVG = 0.5462
    print(f"{'Linear Avg':<15} {'0':<12} {BASELINE_LINEAR_AVG:<12.4f} {BASELINE_LINEAR_AVG:<12.4f} {'N/A':<12}")
    print("-" * 80)

    # Model results
    comparison_data = []

    for model_type in ['amn', 'lstm', 'gru', 'transformer']:
        if model_type not in results:
            continue

        data = results[model_type]
        params = data.get('final_params', 0)
        history = data.get('history', [])

        final_val, final_train, best_val, best_train = compute_stats(history)

        # Format params
        if params >= 1_000_000:
            param_str = f"{params / 1_000_000:.1f}M"
        else:
            param_str = f"{params / 1000:.1f}K"

        print(f"{model_type.upper():<15} {param_str:<12} {final_val:<12.4f} {best_val:<12.4f} {final_train:<12.4f}")

        comparison_data.append({
            'model': model_type,
            'params': params,
            'final_val': final_val,
            'best_val': best_val,
            'beats_baseline': best_val > BASELINE_LINEAR_AVG
        })

    print()

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    # Find best model
    best_model = max(comparison_data, key=lambda x: x['best_val'])

    print(f"🏆 Best Model: {best_model['model'].upper()}")
    print(f"   Best Val Cosine: {best_model['best_val']:.4f}")
    print(f"   Parameters: {best_model['params']:,}")
    print()

    # Check if any beat baseline
    models_beat_baseline = [m for m in comparison_data if m['beats_baseline']]

    if models_beat_baseline:
        print("✅ Models that beat linear baseline (0.5462):")
        for m in models_beat_baseline:
            improvement = ((m['best_val'] - BASELINE_LINEAR_AVG) / BASELINE_LINEAR_AVG) * 100
            print(f"   - {m['model'].upper()}: {m['best_val']:.4f} (+{improvement:.1f}%)")
    else:
        print("❌ No models beat linear baseline yet")
        print("   This suggests:")
        print("   - May need more training epochs")
        print("   - May need hyperparameter tuning")
        print("   - Task may be harder than expected")

    print()

    # Efficiency analysis
    print("⚡ Efficiency (Cosine / Million Params):")
    for m in sorted(comparison_data, key=lambda x: x['best_val'] / (x['params'] / 1e6), reverse=True):
        efficiency = m['best_val'] / (m['params'] / 1e6)
        print(f"   - {m['model'].upper()}: {efficiency:.4f}")

    print()

    # Recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    if best_model['model'] == 'amn':
        print("✨ AMN (Attention Mixture Network) is performing best!")
        print("   This validates our hypothesis:")
        print("   - Residual learning over baseline works")
        print("   - Lightweight attention is sufficient")
        print("   - Interpretable predictions")
        print()
        print("   Next steps:")
        print("   1. Analyze attention weights (see what model learned)")
        print("   2. Try longer training (30-50 epochs)")
        print("   3. Experiment with residual MLP depth")
    elif best_model['beats_baseline']:
        print(f"✨ {best_model['model'].upper()} is performing best!")
        print(f"   And it beats the linear baseline!")
        print()
        print("   Next steps:")
        print("   1. Analyze where it outperforms baseline")
        print("   2. Try longer training")
        print("   3. Tune hyperparameters")
    else:
        print("⚠️  No models beat linear baseline yet.")
        print()
        print("   Debugging steps:")
        print("   1. Check training curves (validation not overfitting?)")
        print("   2. Try different learning rates")
        print("   3. Increase model capacity")
        print("   4. Check data quality")

    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
