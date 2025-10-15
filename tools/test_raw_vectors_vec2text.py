#!/usr/bin/env python3
"""Test raw training vectors directly through vec2text"""
import numpy as np
import requests

VEC2TEXT_URL = "http://127.0.0.1:8766"

def decode_vector(vector):
    response = requests.post(
        f"{VEC2TEXT_URL}/decode",
        json={"vectors": [vector.tolist()], "steps": 1, "subscribers": "jxe"},
        timeout=60
    )
    result = response.json()
    return result["results"][0]["subscribers"]["gtr → jxe"]

# Load training data
npz = np.load("artifacts/lvm/training_sequences_ctx5.npz")
texts_npz = np.load("artifacts/lvm/wikipedia_42113_ordered.npz")

targets = npz['target_vectors']
texts = texts_npz['texts']

print("="*80)
print("Raw Training Vectors → Vec2Text Test")
print("="*80)

# Test 5 vectors at different positions
indices = [100, 2000, 4000, 6000, 8000]

for i, idx in enumerate(indices, 1):
    if idx >= len(targets):
        continue
    
    vector = targets[idx]
    ground_truth = texts[idx + 5]  # +5 because target is 5 positions ahead
    
    # Decode
    result = decode_vector(vector)
    decoded_text = result["output"]
    cosine = result["cosine"]
    
    print(f"\nExample {i} (idx={idx})")
    print(f"  Ground Truth: {ground_truth[:80]}")
    print(f"  Vec2Text Out: {decoded_text[:80]}")
    print(f"  Cosine:       {cosine:.4f}")
    print()

