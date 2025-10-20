#!/usr/bin/env python3
"""
Fair Comparison: 232k vs 340k vs 367k Models
=============================================

Evaluates ALL models on the SAME fixed test set to ensure fair comparison.

Test Set Strategy:
- Use FIRST 10,000 vectors from 232k baseline as test set
- These vectors were present in ALL three training datasets
- Ensures apples-to-apples comparison across all model generations

Usage:
    python tools/fair_comparison_all_datasets.py
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json

# Add app/lvm to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent / "app" / "lvm"))
from models import create_model


def cosine_similarity_batch(pred, target):
    """Compute cosine similarity between predictions and targets"""
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True) + 1e-8)
    return (pred_norm * target_norm).sum(dim=1)


def evaluate_model(model, contexts, targets, device, batch_size=64):
    """Evaluate model on test set"""
    model.eval()
    all_cosines = []
    all_mses = []

    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch_ctx = contexts[i:i+batch_size].to(device)
            batch_tgt = targets[i:i+batch_size].to(device)

            pred_raw, pred_cos = model(batch_ctx, return_raw=True)

            # Cosine similarity
            cosines = cosine_similarity_batch(pred_cos, batch_tgt)
            all_cosines.extend(cosines.cpu().numpy())

            # MSE
            mse = nn.functional.mse_loss(pred_cos, batch_tgt, reduction='none').mean(dim=1)
            all_mses.extend(mse.cpu().numpy())

    return {
        'cosine_mean': np.mean(all_cosines),
        'cosine_std': np.std(all_cosines),
        'mse_mean': np.mean(all_mses),
        'mse_std': np.std(all_mses)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-size', type=int, default=10000, help='Number of test samples')
    parser.add_argument('--device', default='mps', choices=['cpu', 'mps', 'cuda'])
    args = parser.parse_args()

    device = torch.device(args.device if torch.backends.mps.is_available() or args.device == 'cpu' else 'cpu')

    print("=" * 80)
    print("Fair Comparison: 232k vs 340k vs 367k Models")
    print("=" * 80)
    print()
    print(f"Test set size: {args.test_size:,} samples")
    print(f"Device: {device}")
    print()

    # ============================================================================
    # Load Test Data (Fixed set from original 232k data)
    # ============================================================================

    print("Step 1: Loading test data...")
    print()

    # Use the current NPZ file but take FIRST 10k as test set
    # (These were in the original 232k baseline)
    data_path = Path("artifacts/lvm/data/training_sequences_ctx5.npz")
    if not data_path.exists():
        print(f"✗ Data not found: {data_path}")
        return 1

    data = np.load(data_path)
    contexts = torch.FloatTensor(data['context_sequences'][:args.test_size])
    targets = torch.FloatTensor(data['target_vectors'][:args.test_size])

    print(f"✓ Loaded test set: {len(contexts):,} samples")
    print(f"  Context shape: {contexts.shape}")
    print(f"  Target shape: {targets.shape}")
    print()

    # ============================================================================
    # Evaluate All Models
    # ============================================================================

    models_to_test = {
        # 232k baseline models (Oct 17)
        '232k_transformer': 'artifacts/lvm/models/transformer_232k_20251017_090129/best_model.pt',
        '232k_lstm': 'artifacts/lvm/models/lstm_232k_20251017_090129/best_model.pt',
        '232k_gru': 'artifacts/lvm/models/gru_232k_20251017_090129/best_model.pt',
        '232k_amn': 'artifacts/lvm/models/amn_232k_20251017_090129/best_model.pt',

        # 340k models (Oct 18 morning)
        '340k_transformer': 'artifacts/lvm/models_340k/transformer/best_model.pt',
        '340k_lstm': 'artifacts/lvm/models_340k/lstm/best_model.pt',
        '340k_gru': 'artifacts/lvm/models_340k/gru/best_model.pt',
        '340k_amn': 'artifacts/lvm/models_340k/amn/best_model.pt',

        # 367k models (Oct 18 afternoon)
        '367k_transformer': 'artifacts/lvm/models_367k/transformer/best_model.pt',
        '367k_lstm': 'artifacts/lvm/models_367k/lstm/best_model.pt',
        '367k_gru': 'artifacts/lvm/models_367k/gru/best_model.pt',
        '367k_amn': 'artifacts/lvm/models_367k/amn/best_model.pt',
    }

    results = {}

    for model_name, model_path in models_to_test.items():
        print(f"Evaluating {model_name}...")

        model_path = Path(model_path)
        if not model_path.exists():
            print(f"  ⚠️  Model not found: {model_path}")
            continue

        # Determine model type from name
        if 'transformer' in model_name:
            model_type = 'transformer'
        elif 'lstm' in model_name:
            model_type = 'lstm'
        elif 'gru' in model_name:
            model_type = 'gru'
        elif 'amn' in model_name:
            model_type = 'amn'
        else:
            print(f"  ⚠️  Unknown model type for {model_name}")
            continue

        # Create model
        model = create_model(model_type, context_length=5, vector_dim=768)

        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)

        # Evaluate
        metrics = evaluate_model(model, contexts, targets, device)
        results[model_name] = metrics

        print(f"  ✓ Cosine: {metrics['cosine_mean']:.4f} ± {metrics['cosine_std']:.4f}")
        print(f"  ✓ MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
        print()

    # ============================================================================
    # Compare Results
    # ============================================================================

    print("=" * 80)
    print("FAIR COMPARISON RESULTS (Same Test Set)")
    print("=" * 80)
    print()

    print("📊 Cosine Similarity (Higher is Better)")
    print()

    print("232k Baseline:")
    for name in ['232k_transformer', '232k_lstm', '232k_gru', '232k_amn']:
        if name in results:
            print(f"  {name:20s}: {results[name]['cosine_mean']:.4f}")

    print()
    print("340k Models:")
    for name in ['340k_transformer', '340k_lstm', '340k_gru', '340k_amn']:
        if name in results:
            print(f"  {name:20s}: {results[name]['cosine_mean']:.4f}")

    print()
    print("367k Models:")
    for name in ['367k_transformer', '367k_lstm', '367k_gru', '367k_amn']:
        if name in results:
            print(f"  {name:20s}: {results[name]['cosine_mean']:.4f}")

    print()
    print("=" * 80)
    print("PROGRESSIVE IMPROVEMENTS")
    print("=" * 80)
    print()

    for arch in ['transformer', 'lstm', 'gru', 'amn']:
        baseline_key = f'232k_{arch}'
        mid_key = f'340k_{arch}'
        new_key = f'367k_{arch}'

        if baseline_key not in results:
            continue

        print(f"{arch.upper()}:")

        baseline_score = results[baseline_key]['cosine_mean']
        print(f"  232k baseline:  {baseline_score:.4f}")

        if mid_key in results:
            mid_score = results[mid_key]['cosine_mean']
            diff_340 = mid_score - baseline_score
            pct_340 = (diff_340 / baseline_score) * 100
            arrow_340 = "⬆️" if diff_340 > 0 else "⬇️"
            print(f"  340k model:     {mid_score:.4f} ({diff_340:+.4f}, {pct_340:+.2f}%) {arrow_340}")

        if new_key in results:
            new_score = results[new_key]['cosine_mean']
            diff_367 = new_score - baseline_score
            pct_367 = (diff_367 / baseline_score) * 100
            arrow_367 = "⬆️" if diff_367 > 0 else "⬇️"
            print(f"  367k model:     {new_score:.4f} ({diff_367:+.4f}, {pct_367:+.2f}%) {arrow_367}")

            # 340k → 367k improvement
            if mid_key in results:
                diff_increment = new_score - mid_score
                pct_increment = (diff_increment / mid_score) * 100
                arrow_increment = "⬆️" if diff_increment > 0 else "⬇️"
                print(f"  340k→367k:      {diff_increment:+.4f} ({pct_increment:+.2f}%) {arrow_increment}")

        print()

    print("=" * 80)
    print()

    # Save results
    output_file = Path("artifacts/lvm/fair_comparison_all_datasets.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {output_file}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
