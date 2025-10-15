#!/usr/bin/env python3
"""Test GTR-T5 API embeddings directly with vec2text"""

import requests
import numpy as np

print("\n" + "="*80)
print("TESTING GTR-T5 API → VEC2TEXT (DIRECT)")
print("="*80 + "\n")

test_text = "The Earth is the third planet from the Sun."
print(f"Test text: {test_text}\n")

# Get embedding from GTR-T5 API
print("1. Getting embedding from GTR-T5 API...")
response = requests.post(
    'http://localhost:8767/embed',
    json={"texts": [test_text]},
    timeout=30
)

embedding = response.json()['embeddings'][0]
vec_array = np.array(embedding, dtype=np.float32)

print(f"   Dimension: {len(embedding)}")
print(f"   Norm: {np.linalg.norm(vec_array):.4f}")
print(f"   First 10 values: {vec_array[:10]}")
print()

# Test with vec2text
print("2. Decoding through vec2text...")
response = requests.post(
    'http://localhost:8766/decode',
    json={
        'vectors': [embedding],
        'subscribers': 'jxe',
        'steps': 5,
        'device': 'cpu'
    },
    timeout=120
)

result = response.json()['results'][0]['subscribers']['gtr → jxe']

print(f"   Original:      {test_text}")
print(f"   Reconstructed: {result['output']}")
print(f"   Cosine:        {result['cosine']:.4f}")
print()

if result['cosine'] >= 0.65:
    print("✅ GTR-T5 API is working correctly!")
else:
    print("❌ GTR-T5 API is BROKEN - not producing valid embeddings")

print("\n" + "="*80 + "\n")
