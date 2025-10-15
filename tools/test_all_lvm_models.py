#!/usr/bin/env python3
"""
Test All Retrained LVM Models with Vec2Text
==========================================

Tests LSTM and GRU models (with L2 normalization) against vec2text reconstruction.
Expected: High cosine similarity (0.65-0.85) instead of broken output (0.05-0.11).
"""

import sys
sys.path.insert(0, 'app/lvm')

import torch
import numpy as np
import requests
from pathlib import Path

from train_lstm_baseline import LSTMVectorPredictor
from train_mamba2 import Mamba2VectorPredictor

print("\n" + "="*80)
print("LVM MODELS → VEC2TEXT PIPELINE TEST (WITH L2 NORMALIZATION)")
print("="*80 + "\n")

# Load test data
train_data = np.load('artifacts/lvm/training_sequences_ctx5.npz', allow_pickle=True)
context_sequences = train_data['context_sequences']
target_vectors = train_data['target_vectors']
if 'target_texts' in train_data.files:
    target_texts = train_data['target_texts']
else:
    target_texts = np.array([''] * len(target_vectors))

# Use validation samples
split_idx = int(0.9 * len(context_sequences))
val_contexts = context_sequences[split_idx:split_idx+5]
val_targets = target_vectors[split_idx:split_idx+5]
val_texts = target_texts[split_idx:split_idx+5]

device = torch.device('cpu')  # Use CPU for consistency

models = [
    ('LSTM', 'artifacts/lvm/models/lstm_baseline/best_model.pt', LSTMVectorPredictor,
     {'input_dim': 768, 'hidden_dim': 512, 'num_layers': 2}),
    ('GRU', 'artifacts/lvm/models/mamba2/best_model.pt', Mamba2VectorPredictor,
     {'input_dim': 768, 'd_model': 512, 'num_layers': 4}),
]

for model_name, model_path, model_class, kwargs in models:
    print(f"{'='*80}")
    print(f"Testing {model_name} Model")
    print(f"{'='*80}\n")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = model_class(**kwargs).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ {model_name} model loaded\n")

    # Test 5 samples
    results = []

    for i in range(5):
        context = torch.FloatTensor(val_contexts[i:i+1]).to(device)
        ground_truth_text = str(val_texts[i])
        ground_truth_vec = val_targets[i]

        # Get LVM prediction
        with torch.no_grad():
            pred_raw, pred_cos = model(context, return_raw=True)
        predicted_raw = pred_raw.cpu().numpy()[0]
        predicted_vec = pred_cos.cpu().numpy()[0]

        # Check L2 norm (should be 1.0 now!)
        pred_norm_raw = np.linalg.norm(predicted_raw)
        pred_norm = np.linalg.norm(predicted_vec)
        target_norm = np.linalg.norm(ground_truth_vec)

        print(f"Sample {i+1}/5")
        print("─"*80)
        print(f"Ground Truth Text:\n  {ground_truth_text[:80]}...\n")
        print(f"Vector Norms:")
        print(f"  Predicted raw: {pred_norm_raw:.4f}")
        print(f"  Predicted normed: {pred_norm:.4f} (should be ~1.0)")
        print(f"  Target:    {target_norm:.4f}\n")

        # Decode predicted vector via vec2text server
        try:
            response = requests.post(
                'http://localhost:8766/decode',
                json={
                    'vectors': [predicted_raw.tolist()],
                    'subscribers': 'jxe',
                    'steps': 5,
                    'device': 'cpu'
                },
                timeout=120
            )

            if response.status_code != 200:
                print(f"✗ HTTP Error {response.status_code}\n")
                results.append({'sample': i+1, 'error': f"HTTP {response.status_code}"})
                continue

            data = response.json()
            result = data['results'][0]['subscribers']['gtr → jxe']

            if result['status'] != 'success':
                print(f"✗ Decode Error: {result.get('error', 'Unknown')}\n")
                results.append({'sample': i+1, 'error': result.get('error', 'Unknown')})
                continue

            predicted_text = result['output']
            cosine = result['cosine']

            print(f"Vec2Text Reconstructed Text:\n  {predicted_text[:80]}...\n")
            print(f"Cosine Similarity: {cosine:.4f}")

            # Determine if this is good or broken
            if cosine >= 0.65:
                print(f"✓ EXCELLENT! Vec2text working correctly (expected 0.65-0.85)\n")
            elif cosine >= 0.30:
                print(f"⚠️  MODERATE - Better than before but could be improved\n")
            else:
                print(f"✗ BROKEN - Still producing nonsense (same as LayerNorm issue)\n")

            results.append({
                'sample': i+1,
                'cosine': cosine,
                'pred_norm_raw': pred_norm_raw,
                'pred_norm': pred_norm,
                'target_norm': target_norm,
                'ground_truth': ground_truth_text[:50],
                'prediction': predicted_text[:50]
            })

        except Exception as e:
            print(f"✗ Error: {e}\n")
            results.append({'sample': i+1, 'error': str(e)})

    # Summary statistics
    print(f"{'='*80}")
    print(f"{model_name} Model Summary")
    print(f"{'='*80}\n")

    successful = [r for r in results if 'cosine' in r]
    if successful:
        avg_cosine = np.mean([r['cosine'] for r in successful])
        avg_pred_norm_raw = np.mean([r['pred_norm_raw'] for r in successful])
        avg_pred_norm = np.mean([r['pred_norm'] for r in successful])
        avg_target_norm = np.mean([r['target_norm'] for r in successful])

        print(f"Successful reconstructions: {len(successful)}/5")
        print(f"Average cosine similarity: {avg_cosine:.4f}")
        print(f"Average predicted raw norm: {avg_pred_norm_raw:.4f}")
        print(f"Average predicted normed: {avg_pred_norm:.4f} (should be 1.0)")
        print(f"Average target norm: {avg_target_norm:.4f}")
        print()

        if avg_cosine >= 0.65:
            print(f"✅ {model_name} MODEL WORKING PERFECTLY WITH VEC2TEXT!")
            print(f"   L2 normalization fix successful!\n")
        elif avg_cosine >= 0.30:
            print(f"⚠️  {model_name} MODEL IMPROVED BUT NEEDS MORE WORK\n")
        else:
            print(f"❌ {model_name} MODEL STILL BROKEN - L2 normalization may not be working\n")
    else:
        print(f"❌ ALL TESTS FAILED FOR {model_name} MODEL\n")

    print()

print("="*80)
print("Testing Complete")
print("="*80 + "\n")
