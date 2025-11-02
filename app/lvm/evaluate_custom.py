#!/usr/bin/env python3
"""
Model Evaluation Script
=======================

Evaluates all trained LVM models and compares their performance.

Metrics:
- MSE Loss
- Cosine Similarity
- Top-K Retrieval Accuracy (how often correct next vector is in top K neighbors)
- Inference Speed
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime
import argparse
from tqdm import tqdm
import time

# Import model architectures
from train_unified import VectorSequenceDataset
from models import create_model


def load_model(model_path, device):
    """Load a trained model"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_type = checkpoint.get('model_type', 'amn')
    model_config = checkpoint.get('model_config', {})

    model = create_model(model_type, **model_config)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def cosine_similarity(pred, target):
    """Compute cosine similarity"""
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    return (pred_norm * target_norm).sum(dim=1)


def compute_top_k_accuracy(predictions, targets, all_vectors, k_values=[1, 5, 10]):
    """
    Compute top-K retrieval accuracy.

    For each prediction, find K nearest neighbors in all_vectors
    and check if target is in top K.
    """
    results = {k: 0 for k in k_values}

    pred_norm = predictions / (predictions.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = targets / (targets.norm(dim=1, keepdim=True) + 1e-8)
    all_norm = all_vectors / (all_vectors.norm(dim=1, keepdim=True) + 1e-8)

    for i in range(len(predictions)):
        # Compute similarities to all vectors
        sims = (pred_norm[i:i+1] @ all_norm.T).squeeze()  # [N]

        # Get top K indices
        top_k_indices = torch.topk(sims, max(k_values) + 1).indices  # +1 to handle potential self-match

        # Find target index in all_vectors
        target_sims = (target_norm[i:i+1] @ all_norm.T).squeeze()
        target_idx = torch.argmax(target_sims).item()

        # Check if target is in top K
        for k in k_values:
            if target_idx in top_k_indices[:k]:
                results[k] += 1

    # Convert to percentages
    for k in k_values:
        results[k] = (results[k] / len(predictions)) * 100

    return results


def evaluate_model(model, dataloader, all_vectors, device):
    """Evaluate model on test set"""
    model.eval()

    all_predictions = []
    all_targets = []
    all_losses = []
    all_cosines = []
    inference_times = []

    with torch.no_grad():
        for contexts, targets in tqdm(dataloader, desc="Evaluating"):
            contexts = contexts.to(device)
            targets = targets.to(device)

            # Time inference
            start_time = time.time()
            predictions = model(contexts)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # MSE loss
            loss = nn.functional.mse_loss(predictions, targets, reduction='none').mean(dim=1)
            all_losses.append(loss.cpu())

            # Cosine similarity
            cosines = cosine_similarity(predictions, targets)
            all_cosines.append(cosines.cpu())

            # Store for top-K accuracy
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

    # Concatenate results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_losses = torch.cat(all_losses, dim=0)
    all_cosines = torch.cat(all_cosines, dim=0)

    # Compute metrics
    metrics = {
        'mse_loss': {
            'mean': all_losses.mean().item(),
            'std': all_losses.std().item(),
            'median': all_losses.median().item(),
        },
        'cosine_similarity': {
            'mean': all_cosines.mean().item(),
            'std': all_cosines.std().item(),
            'median': all_cosines.median().item(),
        },
        'inference_time': {
            'mean_ms': np.mean(inference_times) * 1000,
            'std_ms': np.std(inference_times) * 1000,
            'samples_per_second': len(all_predictions) / sum(inference_times),
        }
    }

    # Compute top-K accuracy (on subset for speed)
    print("Computing top-K retrieval accuracy...")
    subset_size = min(1000, len(all_predictions))
    top_k_acc = compute_top_k_accuracy(
        all_predictions[:subset_size].to(device),
        all_targets[:subset_size].to(device),
        all_vectors,
        k_values=[1, 5, 10, 20]
    )
    metrics['top_k_accuracy'] = top_k_acc

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--all-vectors', default='artifacts/lvm/wikipedia_584k_fresh.npz')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--output', default='artifacts/lvm/evaluation_results_custom.json')
    args = parser.parse_args()

    print("=" * 80)
    print("LVM Model Evaluation")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Models: {len(args.models)}")
    print(f"Datasets: {len(args.datasets)}")
    print()

    device = torch.device(args.device)

    # Load all vectors for top-K accuracy
    print("Loading all vectors...")
    all_vectors_data = np.load(args.all_vectors)
    all_vectors = torch.FloatTensor(all_vectors_data['vectors']).to(device)
    print(f"Vector database size: {len(all_vectors)}")
    print()

    results = {}

    for model_path in args.models:
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"⚠️  Model not found: {model_path}")
            continue

        model_name = model_path.parent.name
        results[model_name] = {}

        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*80}")

        # Load model
        print("Loading model...")
        model, checkpoint = load_model(model_path, device)

        # Print model info
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {params:,}")
        print(f"Training epochs: {checkpoint.get('epoch', 'N/A')}")
        print(f"Training val_loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
        print()

        for dataset_path in args.datasets:
            dataset_name = Path(dataset_path).stem
            print(f"--- Evaluating on: {dataset_name} ---")
            
            # Load test data
            dataset = VectorSequenceDataset(dataset_path)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0
            )
            print(f"Test samples: {len(dataset)}")

            # Evaluate
            metrics = evaluate_model(model, dataloader, all_vectors, device)

            # Print results
            print("\nResults:")
            print(f"  MSE Loss: {metrics['mse_loss']['mean']:.6f} ± {metrics['mse_loss']['std']:.6f}")
            print(f"  Cosine Similarity: {metrics['cosine_similarity']['mean']:.4f} ± {metrics['cosine_similarity']['std']:.4f}")
            print(f"  Inference: {metrics['inference_time']['mean_ms']:.2f}ms/batch ({metrics['inference_time']['samples_per_second']:.1f} samples/sec)")
            print(f"  Top-K Retrieval Accuracy:")
            for k, acc in sorted(metrics['top_k_accuracy'].items()):
                print(f"    Top-{k}: {acc:.2f}%")

            # Store results
            results[model_name][dataset_name] = {
                'model_path': str(model_path),
                'parameters': params,
                'metrics': metrics,
                'checkpoint_info': {
                    'epoch': checkpoint.get('epoch', None),
                    'val_loss': checkpoint.get('val_loss', None),
                    'val_cosine': checkpoint.get('val_cosine', None),
                }
            }

    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'evaluated_at': datetime.now().isoformat(),
            'device': str(device),
            'results': results
        }, f, indent=2)

    print(f"Results saved to: {output_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    
    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}")
        print(f"{'Dataset':<40} {'Loss':<10} {'Cosine':<10}")
        print("-" * 60)
        for dataset_name, data in model_results.items():
            loss = data['metrics']['mse_loss']['mean']
            cosine = data['metrics']['cosine_similarity']['mean']
            print(f"{dataset_name:<40} {loss:<10.6f} {cosine:<10.4f}")

    print(f"{'='*80}")


if __name__ == '__main__':
    main()
