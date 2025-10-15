#!/usr/bin/env python3
"""Test GTR-T5 directly with sentence-transformers (bypassing API) → vec2text"""
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

print("\n" + "="*80)
print("TESTING SENTENCE-TRANSFORMERS GTR-T5 DIRECTLY → VEC2TEXT")
print("="*80 + "\n")

# Load GTR-T5 directly
print("Loading GTR-T5 model directly...")
model = SentenceTransformer('sentence-transformers/gtr-t5-base', device='cpu')

test_texts = [
    "The Earth is round",
    "Water is essential for life",
    "Python is a programming language"
]

print("\nTest texts:")
for i, text in enumerate(test_texts, 1):
    print(f"  {i}. '{text}'")

# Get embeddings directly from sentence-transformers
print("\nGenerating embeddings with sentence-transformers...")
embeddings = model.encode(test_texts, normalize_embeddings=True, show_progress_bar=False)
embeddings = np.asarray(embeddings, dtype=np.float32)

print(f"\nEmbedding shape: {embeddings.shape}")
print(f"Sample vector norm: {np.linalg.norm(embeddings[0]):.6f}")

# Test with vec2text
print("\n" + "-"*80)
print("TESTING WITH VEC2TEXT")
print("-"*80)

passed = 0
failed = 0

for i, (text, vec) in enumerate(zip(test_texts, embeddings), 1):
    # Send to vec2text (API expects vectors as a list)
    response = requests.post(
        'http://localhost:8766/decode',
        json={'vectors': [vec.tolist()], 'steps': 1, 'subscribers': 'jxe,ielab'}
    )
    result = response.json()

    # Get first result (we sent 1 vector)
    if result.get('results'):
        decoded_text = result['results'][0].get('ielab', {}).get('decoded_text', 'ERROR')
        vec2text_vec_data = result['results'][0].get('ielab', {}).get('decoded_vector', [0]*768)
    else:
        decoded_text = 'ERROR'
        vec2text_vec_data = [0]*768

    # Get vec2text's re-encoding
    vec2text_vec = np.array(vec2text_vec_data, dtype=np.float32)
    cosine = np.dot(vec, vec2text_vec) / (np.linalg.norm(vec) * np.linalg.norm(vec2text_vec) + 1e-10)

    print(f"\n{i}. Original: '{text}'")
    print(f"   Decoded:  '{decoded_text}'")
    print(f"   Cosine:   {cosine:.4f}")

    if cosine < 0.5:
        print("   ⚠️  FAILED: Cosine too low (expected 0.65-0.85)")
        failed += 1
    elif cosine >= 0.65:
        print("   ✅ PASSED: Good reconstruction")
        passed += 1
    else:
        print("   ⚠️  MARGINAL: Cosine below optimal")
        failed += 1

print("\n" + "="*80)
print("VERDICT:")
if passed == len(test_texts):
    print("✅ GTR-T5 sentence-transformers → vec2text WORKS!")
    print("   Problem is likely in the API wrapper or vectorizer configuration.")
else:
    print(f"❌ GTR-T5 sentence-transformers → vec2text FAILS! ({failed}/{len(test_texts)} failed)")
    print("   This indicates a fundamental incompatibility between GTR-T5 and vec2text.")
    print("\n   CRITICAL: Vec2text was likely trained on a DIFFERENT embedding model!")
    print("   Check vec2text documentation for which encoder it expects.")
print("="*80 + "\n")
