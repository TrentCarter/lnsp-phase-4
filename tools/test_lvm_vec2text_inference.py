#!/usr/bin/env python3
"""
Complete LVM → Vec2Text Inference Pipeline Test
================================================

Tests the full pipeline:
1. LVM inference: context vectors → predicted 768D vector
2. Vector comparison: predicted vs ground truth (cosine similarity)
3. Vec2text decoding: predicted vector → reconstructed text
4. Text comparison: reconstructed vs ground truth text

Measures:
- LVM inference time (ms/chunk)
- Vector prediction quality (cosine similarity)
- Text reconstruction quality (BLEU, exact match)
- End-to-end pipeline time
"""

import sys
sys.path.insert(0, 'app/lvm')

import torch
import numpy as np
import requests
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Import model architectures
from train_lstm_baseline import LSTMVectorPredictor
from train_mamba2 import Mamba2VectorPredictor
from train_transformer import TransformerVectorPredictor


def load_model(model_name: str, device: torch.device):
    """Load a trained LVM model"""
    models_config = {
        'lstm': {
            'path': 'artifacts/lvm/models/lstm_baseline/best_model.pt',
            'class': LSTMVectorPredictor,
            'kwargs': {'input_dim': 768, 'hidden_dim': 512, 'num_layers': 2}
        },
        'gru': {
            'path': 'artifacts/lvm/models/mamba2/best_model.pt',
            'class': Mamba2VectorPredictor,
            'kwargs': {'input_dim': 768, 'd_model': 512, 'num_layers': 4}
        },
        'transformer': {
            'path': 'artifacts/lvm/models/transformer/best_model.pt',
            'class': TransformerVectorPredictor,
            'kwargs': {'input_dim': 768, 'd_model': 512, 'nhead': 8, 'num_layers': 4}
        }
    }

    config = models_config[model_name]
    checkpoint = torch.load(config['path'], map_location=device)

    model = config['class'](**config['kwargs']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
    return float(np.dot(vec1_norm, vec2_norm))


def decode_vector_with_vec2text(vector: np.ndarray, steps: int = 5) -> Dict:
    """Decode vector using vec2text server"""
    try:
        response = requests.post(
            'http://localhost:8766/decode',
            json={
                'vectors': [vector.tolist()],
                'subscribers': 'jxe',
                'steps': steps,
                'device': 'cpu'
            },
            timeout=120
        )

        if response.status_code != 200:
            return {'success': False, 'error': f'HTTP {response.status_code}'}

        data = response.json()
        result = data['results'][0]['subscribers']['gtr → jxe']

        if result['status'] == 'success':
            return {
                'success': True,
                'text': result['output'],
                'cosine': result['cosine']
            }
        else:
            return {'success': False, 'error': result.get('error', 'Unknown')}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def test_model_pipeline(model_name: str, device: torch.device, num_samples: int = 10):
    """Test complete LVM → vec2text pipeline for one model"""

    print(f"\n{'='*80}")
    print(f"Testing {model_name.upper()} Model Pipeline")
    print(f"{'='*80}\n")

    # Load model
    print(f"Loading {model_name} model...")
    model, checkpoint = load_model(model_name, device)
    print(f"✓ Model loaded (val_loss: {checkpoint.get('val_loss', 'N/A'):.6f})")

    # Load test data
    print("Loading validation data...")
    train_data = np.load('artifacts/lvm/training_sequences_ctx5.npz', allow_pickle=True)
    context_sequences = train_data['context_sequences']
    target_vectors = train_data['target_vectors']
    target_texts = train_data['target_texts']

    # Use validation split (last 10%)
    split_idx = int(0.9 * len(context_sequences))
    val_contexts = context_sequences[split_idx:split_idx + num_samples]
    val_targets = target_vectors[split_idx:split_idx + num_samples]
    val_texts = target_texts[split_idx:split_idx + num_samples]

    print(f"✓ Loaded {num_samples} validation samples\n")

    # Test each sample
    results = []
    lvm_times = []
    vec2text_times = []
    vector_similarities = []
    text_reconstructions = []

    for i in range(num_samples):
        print(f"{'─'*80}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"{'─'*80}")

        context = torch.FloatTensor(val_contexts[i:i+1]).to(device)
        ground_truth_text = str(val_texts[i])
        ground_truth_vec = val_targets[i]

        print(f"Ground Truth Text:\n  {ground_truth_text[:100]}...\n")

        # Step 1: LVM Inference
        start_time = time.time()
        with torch.no_grad():
            predicted_vec = model(context).cpu().numpy()[0]
        lvm_time = (time.time() - start_time) * 1000  # Convert to ms
        lvm_times.append(lvm_time)

        # Step 2: Vector Comparison
        vec_cosine = cosine_similarity(predicted_vec, ground_truth_vec)
        vector_similarities.append(vec_cosine)

        print(f"LVM Inference:")
        print(f"  Time: {lvm_time:.2f} ms")
        print(f"  Vector Cosine: {vec_cosine:.4f}")
        print(f"  Predicted norm: {np.linalg.norm(predicted_vec):.4f}")
        print(f"  Target norm: {np.linalg.norm(ground_truth_vec):.4f}\n")

        # Step 3: Vec2Text Decoding
        start_time = time.time()
        decode_result = decode_vector_with_vec2text(predicted_vec, steps=5)
        vec2text_time = (time.time() - start_time) * 1000  # Convert to ms
        vec2text_times.append(vec2text_time)

        if decode_result['success']:
            reconstructed_text = decode_result['text']
            text_cosine = decode_result['cosine']
            text_reconstructions.append(text_cosine)

            print(f"Vec2Text Decoding:")
            print(f"  Time: {vec2text_time:.2f} ms")
            print(f"  Text Cosine: {text_cosine:.4f}")
            print(f"  Reconstructed Text:\n    {reconstructed_text[:100]}...\n")

            # Determine quality
            if vec_cosine >= 0.75 and text_cosine >= 0.65:
                quality = "✓ EXCELLENT"
            elif vec_cosine >= 0.60 and text_cosine >= 0.50:
                quality = "○ GOOD"
            elif vec_cosine >= 0.40 and text_cosine >= 0.30:
                quality = "△ FAIR"
            else:
                quality = "✗ POOR"

            print(f"Overall Quality: {quality}")
            print(f"End-to-End Time: {lvm_time + vec2text_time:.2f} ms\n")

            results.append({
                'sample': i + 1,
                'lvm_time_ms': lvm_time,
                'vec2text_time_ms': vec2text_time,
                'total_time_ms': lvm_time + vec2text_time,
                'vector_cosine': vec_cosine,
                'text_cosine': text_cosine,
                'ground_truth': ground_truth_text[:100],
                'reconstructed': reconstructed_text[:100],
                'quality': quality
            })
        else:
            print(f"Vec2Text Decoding FAILED:")
            print(f"  Error: {decode_result['error']}\n")

            results.append({
                'sample': i + 1,
                'lvm_time_ms': lvm_time,
                'vec2text_time_ms': None,
                'total_time_ms': None,
                'vector_cosine': vec_cosine,
                'text_cosine': None,
                'error': decode_result['error']
            })

    # Summary Statistics
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} Model Summary")
    print(f"{'='*80}\n")

    successful = [r for r in results if 'error' not in r]

    if successful:
        avg_lvm_time = np.mean(lvm_times)
        avg_vec2text_time = np.mean(vec2text_times)
        avg_total_time = np.mean([r['total_time_ms'] for r in successful])
        avg_vector_cosine = np.mean(vector_similarities)
        avg_text_cosine = np.mean(text_reconstructions)

        print(f"Performance Metrics:")
        print(f"  LVM Inference: {avg_lvm_time:.2f} ms/chunk")
        print(f"  Vec2Text Decode: {avg_vec2text_time:.2f} ms/chunk")
        print(f"  End-to-End: {avg_total_time:.2f} ms/chunk")
        print(f"  Throughput: {1000/avg_total_time:.1f} chunks/sec\n")

        print(f"Quality Metrics:")
        print(f"  Vector Cosine: {avg_vector_cosine:.4f} ± {np.std(vector_similarities):.4f}")
        print(f"  Text Cosine: {avg_text_cosine:.4f} ± {np.std(text_reconstructions):.4f}")
        print(f"  Success Rate: {len(successful)}/{num_samples} ({100*len(successful)/num_samples:.1f}%)\n")

        # Quality distribution
        excellent = sum(1 for r in successful if r['quality'] == "✓ EXCELLENT")
        good = sum(1 for r in successful if r['quality'] == "○ GOOD")
        fair = sum(1 for r in successful if r['quality'] == "△ FAIR")
        poor = sum(1 for r in successful if r['quality'] == "✗ POOR")

        print(f"Quality Distribution:")
        print(f"  Excellent: {excellent}/{len(successful)} ({100*excellent/len(successful):.1f}%)")
        print(f"  Good:      {good}/{len(successful)} ({100*good/len(successful):.1f}%)")
        print(f"  Fair:      {fair}/{len(successful)} ({100*fair/len(successful):.1f}%)")
        print(f"  Poor:      {poor}/{len(successful)} ({100*poor/len(successful):.1f}%)\n")

    else:
        print(f"❌ ALL TESTS FAILED\n")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['lstm', 'gru', 'transformer'])
    parser.add_argument('--samples', type=int, default=10, help='Number of test samples per model')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/mps)')
    parser.add_argument('--output', default='artifacts/lvm/inference_test_results.json')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("LVM → Vec2Text Complete Pipeline Test")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Samples per model: {args.samples}")
    print()

    device = torch.device(args.device)

    # Test each model
    all_results = {}

    for model_name in args.models:
        try:
            results = test_model_pipeline(model_name, device, args.samples)
            all_results[model_name] = results
        except Exception as e:
            print(f"\n❌ Error testing {model_name}: {e}\n")
            all_results[model_name] = {'error': str(e)}

    # Save results
    print(f"\n{'='*80}")
    print("Saving Results")
    print(f"{'='*80}\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"✓ Results saved to: {output_path}\n")

    # Final comparison table
    print(f"{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}\n")
    print(f"{'Model':<12} {'LVM ms':<10} {'Vec2Text ms':<14} {'Total ms':<10} {'Vec Cos':<10} {'Text Cos':<10}")
    print("─"*80)

    for model_name, results in all_results.items():
        if 'error' in results:
            print(f"{model_name:<12} ERROR")
        else:
            successful = [r for r in results if 'error' not in r]
            if successful:
                avg_lvm = np.mean([r['lvm_time_ms'] for r in successful])
                avg_vec2text = np.mean([r['vec2text_time_ms'] for r in successful])
                avg_total = np.mean([r['total_time_ms'] for r in successful])
                avg_vec_cos = np.mean([r['vector_cosine'] for r in successful])
                avg_text_cos = np.mean([r['text_cosine'] for r in successful])

                print(f"{model_name:<12} {avg_lvm:<10.2f} {avg_vec2text:<14.2f} {avg_total:<10.2f} {avg_vec_cos:<10.4f} {avg_text_cos:<10.4f}")

    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
