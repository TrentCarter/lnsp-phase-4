#!/usr/bin/env python3
"""Test if ground truth vectors decode properly through vec2text"""

import numpy as np
import requests

# Load test data
train_data = np.load('artifacts/lvm/training_sequences_ctx5.npz', allow_pickle=True)
target_vectors = train_data['target_vectors']
target_texts = train_data['target_texts']

# Use validation samples
split_idx = int(0.9 * len(target_vectors))
val_targets = target_vectors[split_idx:split_idx+3]
val_texts = target_texts[split_idx:split_idx+3]

print("\n" + "="*80)
print("GROUND TRUTH VECTORS → VEC2TEXT TEST")
print("="*80 + "\n")

for i in range(3):
    ground_truth_vec = val_targets[i]
    ground_truth_text = str(val_texts[i])

    print(f"Sample {i+1}/3")
    print("─"*80)
    print(f"Original Text:\n  {ground_truth_text[:80]}...\n")
    print(f"Vector Norm: {np.linalg.norm(ground_truth_vec):.4f}\n")

    # Decode ground truth vector via vec2text
    response = requests.post(
        'http://localhost:8766/decode',
        json={
            'vectors': [ground_truth_vec.tolist()],
            'subscribers': 'jxe',
            'steps': 5,
            'device': 'cpu'
        },
        timeout=120
    )

    data = response.json()
    result = data['results'][0]['subscribers']['gtr → jxe']

    reconstructed_text = result['output']
    cosine = result['cosine']

    print(f"Vec2Text Reconstructed:\n  {reconstructed_text[:80]}...\n")
    print(f"Cosine: {cosine:.4f}\n")

    if cosine >= 0.65:
        print("✓ GOOD - Vec2text working correctly\n")
    else:
        print("✗ BAD - Vec2text not working even on ground truth\n")

print("="*80 + "\n")
