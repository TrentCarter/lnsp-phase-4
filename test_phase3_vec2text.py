#!/usr/bin/env python3
"""
Test 3.1: Vec2Text Integration
================================

Tests full pipeline: context vectors → LVM prediction → vec2text decoding → text comparison.
"""

import sys
sys.path.insert(0, 'app/lvm')

import torch
import numpy as np
import subprocess
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# Import actual model architectures
from train_lstm_baseline import LSTMVectorPredictor
from train_mamba2 import Mamba2VectorPredictor
from train_transformer import TransformerVectorPredictor


def decode_vector_to_text(vector, backend='jxe', steps=1):
    """Decode a vector using vec2text FastAPI server (port 8766)."""
    import requests

    try:
        # Call vec2text server /decode endpoint
        response = requests.post(
            'http://localhost:8766/decode',
            json={
                'vectors': [vector.tolist()],
                'subscribers': backend,
                'steps': steps,
                'device': 'cpu'
            },
            timeout=60
        )

        if response.status_code != 200:
            return f"ERROR: HTTP {response.status_code}"

        data = response.json()

        # Extract decoded text from response
        if 'results' in data and len(data['results']) > 0:
            result = data['results'][0]
            if 'subscribers' in result:
                # Format: {"gtr → jxe": {"output": "text", "cosine": 0.56}}
                key = f"gtr → {backend}"
                if key in result['subscribers']:
                    return result['subscribers'][key].get('output', 'ERROR: No output')

        return "ERROR: Unexpected response format"

    except requests.exceptions.Timeout:
        return "ERROR: Request timeout (>60s)"
    except Exception as e:
        return f"ERROR: {str(e)}"


def test_vec2text_pipeline(model_name, model_path, model_class, model_kwargs, n_samples=10, device='cpu'):
    """Test full vec2text pipeline for a single model."""
    print(f"\n{'='*80}")
    print(f"Testing {model_name}: Vec2Text Integration")
    print(f"{'='*80}\n")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load data
    train_data = np.load('artifacts/lvm/training_sequences_ctx5.npz', allow_pickle=True)
    wiki_data = np.load('artifacts/lvm/wikipedia_42113_ordered.npz', allow_pickle=True)

    context_sequences = train_data['context_sequences']
    target_vectors = train_data['target_vectors']
    if 'target_texts' in train_data.files:
        target_texts = train_data['target_texts']
    else:
        target_texts = np.array([''] * len(target_vectors))

    # Use validation split (last 10%)
    split_idx = int(0.9 * len(context_sequences))
    val_contexts = context_sequences[split_idx:split_idx + n_samples]
    val_targets = target_vectors[split_idx:split_idx + n_samples]
    val_texts = target_texts[split_idx:split_idx + n_samples]

    print(f"Testing {n_samples} samples...")
    print(f"Backends: JXE (vec2text inversion)\n")

    successes = 0
    failures = 0
    semantic_sims = []

    # Test each sample
    for i in range(n_samples):
        context = torch.FloatTensor(val_contexts[i:i+1]).to(device)
        ground_truth_vec = val_targets[i]
        ground_truth_text = str(val_texts[i])

        # LVM prediction
        with torch.no_grad():
            pred_raw, pred_cos = model(context, return_raw=True)
        predicted_raw = pred_raw.cpu().numpy()[0]
        predicted_cos = pred_cos.cpu().numpy()[0]

        # Decode both vectors
        print(f"[{i+1}/{n_samples}] Decoding predicted vector...", end=' ')
        pred_text = decode_vector_to_text(predicted_raw, backend='jxe', steps=1)

        if pred_text.startswith('ERROR'):
            print(f"FAIL - {pred_text}")
            failures += 1
            continue

        # Compute semantic similarity (cosine between vectors)
        vec_similarity = cos_sim([predicted_cos], [ground_truth_vec])[0][0]
        semantic_sims.append(vec_similarity)

        print(f"OK (cos={vec_similarity:.3f})")
        print(f"   Ground truth: {ground_truth_text[:60]}...")
        print(f"   Prediction:   {pred_text[:60]}...")
        successes += 1

    # Results
    avg_semantic_sim = np.mean(semantic_sims) if semantic_sims else 0.0
    success_rate = (successes / n_samples) * 100

    print(f"\n✓ Successful decodings: {successes}/{n_samples} ({success_rate:.1f}%)")
    print(f"✓ Average cosine similarity: {avg_semantic_sim:.4f}")

    return {
        'model': model_name,
        'samples_tested': n_samples,
        'successful_decodings': successes,
        'failed_decodings': failures,
        'success_rate': f"{success_rate:.1f}%",
        'avg_cosine': f"{avg_semantic_sim:.4f}"
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
    print("█  LVM PHASE 3 TEST: Vec2Text Integration Pipeline".center(80, " ") + "█")
    print("█"*80 + "\n")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Note: Vec2text runs on CPU (required by vec2text backends)\n")

    models = [
        ('LSTM', 'artifacts/lvm/models/lstm_baseline/best_model.pt', LSTMVectorPredictor, {'input_dim': 768, 'hidden_dim': 512, 'num_layers': 2}),
        ('GRU', 'artifacts/lvm/models/mamba2/best_model.pt', Mamba2VectorPredictor, {'input_dim': 768, 'd_model': 512, 'num_layers': 4}),
        ('Transformer', 'artifacts/lvm/models/transformer/best_model.pt', TransformerVectorPredictor, {'input_dim': 768, 'd_model': 512, 'nhead': 8, 'num_layers': 4}),
    ]

    # Test all models (vec2text server is fast)
    print("✓ Using vec2text FastAPI server (port 8766) for fast decoding\n")

    results = []
    for name, path, model_class, kwargs in models:  # All models
        try:
            result = test_vec2text_pipeline(name, path, model_class, kwargs, n_samples=3, device=device)
            results.append(result)
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            results.append({
                'model': name,
                'samples_tested': 0,
                'successful_decodings': 0,
                'failed_decodings': 0,
                'success_rate': 'ERROR',
                'avg_cosine': 'N/A'
            })

    # Print summary table
    print("\n" + "="*80)
    print("TEST 3.1: VEC2TEXT INTEGRATION SUMMARY")
    print("="*80 + "\n")
    print_table(results, ['model', 'samples_tested', 'successful_decodings', 'failed_decodings', 'success_rate', 'avg_cosine'])

    # Save results
    output_path = Path('artifacts/lvm/evaluation/phase3_vec2text_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")
    print("\n" + "="*80)
    print("CONCLUSION: Test 3.1 Complete - Vec2Text Integration Pipeline")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
