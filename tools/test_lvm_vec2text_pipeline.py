#!/usr/bin/env python3
"""
Direct Test: LVM Predictions → Vec2Text Reconstruction
"""

import sys
sys.path.insert(0, 'app/lvm')

import torch
import numpy as np
import requests
from pathlib import Path

# Import GRU model
from train_mamba2 import Mamba2VectorPredictor

print("\n" + "="*80)
print("LVM → VEC2TEXT PIPELINE TEST")
print("="*80 + "\n")

# Load GRU model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"LVM Device: {device}")

checkpoint = torch.load('artifacts/lvm/models/mamba2/best_model.pt', map_location=device)
model = Mamba2VectorPredictor(input_dim=768, d_model=512, num_layers=4).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ GRU model loaded\n")

# Load test data
train_data = np.load('artifacts/lvm/training_sequences_ctx5.npz', allow_pickle=True)
context_sequences = train_data['context_sequences']
target_vectors = train_data['target_vectors']
target_texts = train_data['target_texts']

# Use validation samples
split_idx = int(0.9 * len(context_sequences))
val_contexts = context_sequences[split_idx:split_idx+5]
val_targets = target_vectors[split_idx:split_idx+5]
val_texts = target_texts[split_idx:split_idx+5]

print(f"Testing 5 validation samples\n")
print("="*80 + "\n")

for i in range(5):
    context = torch.FloatTensor(val_contexts[i:i+1]).to(device)
    ground_truth_text = str(val_texts[i])

    # Get LVM prediction
    with torch.no_grad():
        predicted_vec = model(context).cpu().numpy()[0]

    print(f"Sample {i+1}/5")
    print("─"*80)
    print(f"Ground Truth Text:\n  {ground_truth_text[:100]}...\n")

    # Decode predicted vector via vec2text server
    response = requests.post(
        'http://localhost:8766/decode',
        json={
            'vectors': [predicted_vec.tolist()],
            'subscribers': 'jxe',
            'steps': 5,
            'device': 'cpu'
        },
        timeout=120
    )

    if response.status_code != 200:
        print(f"✗ HTTP Error {response.status_code}\n")
        continue

    data = response.json()
    result = data['results'][0]['subscribers']['gtr → jxe']

    if result['status'] != 'success':
        print(f"✗ Decode Error: {result.get('error', 'Unknown')}\n")
        continue

    predicted_text = result['output']
    cosine = result['cosine']

    print(f"LVM Predicted Text (via vec2text):\n  {predicted_text[:100]}...\n")
    print(f"Cosine Similarity: {cosine:.4f}")
    print()

print("="*80)
print("Test Complete")
print("="*80 + "\n")
