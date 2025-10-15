#!/usr/bin/env python3
"""
Test 2.1: Top-K Retrieval Accuracy
===================================

Tests LVM predictions against full Wikipedia vector database using Top-K retrieval.
"""

import sys
sys.path.insert(0, 'app/lvm')

import torch
import numpy as np
import time
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Import actual model architectures from training scripts
from train_lstm_baseline import LSTMVectorPredictor
from train_mamba2 import Mamba2VectorPredictor
from train_transformer import TransformerVectorPredictor


def compute_topk_accuracy(predictions, targets, database_vectors, k_values=[1, 5, 10, 20]):
    """
    Compute Top-K retrieval accuracy using vectorized operations.

    Args:
        predictions: (N, 768) predicted vectors
        targets: (N, 768) ground truth vectors
        database_vectors: (M, 768) full database of vectors to search
        k_values: list of K values to test

    Returns:
        dict with Top-K accuracy for each K
    """
    results = {f'top_{k}': 0 for k in k_values}
    n_samples = len(predictions)
    max_k = max(k_values)

    # Normalize vectors for fast cosine similarity via dot product
    pred_norm = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
    target_norm = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8)
    db_norm = database_vectors / (np.linalg.norm(database_vectors, axis=1, keepdims=True) + 1e-8)

    print(f"Finding ground truth indices in database...")
    # Find indices of ground truth vectors in database (batch operation)
    target_similarities = target_norm @ db_norm.T  # (N, M)
    target_indices = np.argmax(target_similarities, axis=1)  # (N,)

    print(f"Computing Top-{max_k} predictions...")
    # Compute all similarities at once: (N, M)
    pred_similarities = pred_norm @ db_norm.T

    # Get Top-K indices for each prediction (only compute up to max_k)
    topk_indices = np.argsort(pred_similarities, axis=1)[:, -max_k:][:, ::-1]  # (N, max_k)

    # Check if ground truth is in Top-K for each K value
    for k in k_values:
        # For each sample, check if target_index is in topk_indices[:k]
        matches = np.any(topk_indices[:, :k] == target_indices[:, np.newaxis], axis=1)
        results[f'top_{k}'] = (np.sum(matches) / n_samples) * 100

    return results


def test_retrieval(model_name, model_path, model_class, model_kwargs, device='cpu'):
    """Test Top-K retrieval for a single model."""
    print(f"\n{'='*80}")
    print(f"Testing {model_name}: Top-K Retrieval Accuracy")
    print(f"{'='*80}\n")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load data
    train_data = np.load('artifacts/lvm/training_sequences_ctx5.npz')
    wiki_data = np.load('artifacts/lvm/wikipedia_42113_ordered.npz')

    context_sequences = train_data['context_sequences']
    target_vectors = train_data['target_vectors']
    database_vectors = wiki_data['vectors']

    # Use validation split (last 10%)
    split_idx = int(0.9 * len(context_sequences))
    val_contexts = context_sequences[split_idx:]
    val_targets = target_vectors[split_idx:]

    print(f"Validation samples: {len(val_contexts)}")
    print(f"Database size: {len(database_vectors)} vectors")
    print(f"Running inference and retrieval...\n")

    # Predict on validation set (in batches to avoid memory issues)
    batch_size = 64
    all_predictions = []

    start_time = time.time()

    with torch.no_grad():
        for i in range(0, len(val_contexts), batch_size):
            batch = torch.FloatTensor(val_contexts[i:i+batch_size]).to(device)
            preds = model(batch).cpu().numpy()
            all_predictions.append(preds)

    all_predictions = np.vstack(all_predictions)
    inference_time = time.time() - start_time

    # Compute Top-K accuracy
    print("Computing Top-K retrieval accuracy...")
    topk_results = compute_topk_accuracy(all_predictions, val_targets, database_vectors)

    print(f"\n✓ Inference time: {inference_time:.2f}s ({len(val_contexts)/inference_time:.1f} samples/sec)")
    print(f"✓ Top-1:  {topk_results['top_1']:.2f}%")
    print(f"✓ Top-5:  {topk_results['top_5']:.2f}%")
    print(f"✓ Top-10: {topk_results['top_10']:.2f}%")
    print(f"✓ Top-20: {topk_results['top_20']:.2f}%")

    return {
        'model': model_name,
        'top_1': f"{topk_results['top_1']:.2f}%",
        'top_5': f"{topk_results['top_5']:.2f}%",
        'top_10': f"{topk_results['top_10']:.2f}%",
        'top_20': f"{topk_results['top_20']:.2f}%",
        'inference_time': f"{inference_time:.1f}s",
        'samples_per_sec': f"{len(val_contexts)/inference_time:.0f}"
    }


def print_table(data, headers):
    """Simple table printer."""
    widths = [max(len(str(row.get(h, ''))) for row in data + [dict(zip(headers, headers))]) + 2 for h in headers]

    # Header
    print("┌" + "┬".join("─" * (w+2) for w in widths) + "┐")
    print("│ " + " │ ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " │")
    print("├" + "┼".join("─" * (w+2) for w in widths) + "┤")

    # Data rows
    for row in data:
        print("│ " + " │ ".join(str(row.get(h, '')).ljust(w) for h, w in zip(headers, widths)) + " │")

    print("└" + "┴".join("─" * (w+2) for w in widths) + "┘")


def main():
    print("\n" + "█"*80)
    print("█  LVM PHASE 2 TEST: Top-K Retrieval Accuracy".center(80, " ") + "█")
    print("█"*80 + "\n")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}\n")

    models = [
        ('LSTM', 'artifacts/lvm/models/lstm_baseline/best_model.pt', LSTMVectorPredictor, {'input_dim': 768, 'hidden_dim': 512, 'num_layers': 2}),
        ('GRU', 'artifacts/lvm/models/mamba2/best_model.pt', Mamba2VectorPredictor, {'input_dim': 768, 'd_model': 512, 'num_layers': 4}),
        ('Transformer', 'artifacts/lvm/models/transformer/best_model.pt', TransformerVectorPredictor, {'input_dim': 768, 'd_model': 512, 'nhead': 8, 'num_layers': 4}),
    ]

    results = []
    for name, path, model_class, kwargs in models:
        try:
            result = test_retrieval(name, path, model_class, kwargs, device=device)
            results.append(result)
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            results.append({
                'model': name,
                'top_1': 'ERROR',
                'top_5': 'ERROR',
                'top_10': 'ERROR',
                'top_20': 'ERROR',
                'inference_time': 'N/A',
                'samples_per_sec': 'N/A'
            })

    # Print summary table
    print("\n" + "="*80)
    print("TEST 2.1: TOP-K RETRIEVAL ACCURACY SUMMARY")
    print("="*80 + "\n")
    print_table(results, ['model', 'top_1', 'top_5', 'top_10', 'top_20', 'inference_time', 'samples_per_sec'])

    # Save results
    import json
    output_path = Path('artifacts/lvm/evaluation/phase2_retrieval_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")
    print("\n" + "="*80)
    print("CONCLUSION: Test 2.1 Complete")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
