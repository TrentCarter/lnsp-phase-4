#!/usr/bin/env python3
"""
Test Database Vectors with Vec2Text Decoder
============================================
Loads vectors from regenerated NPZ and tests vec2text decoding quality.
Tests first 10 and last 10 vectors.
"""
import numpy as np
import requests
import sys
from pathlib import Path

VEC2TEXT_DECODER_URL = "http://127.0.0.1:8766"
VEC2TEXT_ENCODER_URL = "http://127.0.0.1:8767"

def compute_cosine(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def test_vector(idx, original_text, vector):
    """Test a single vector with vec2text decoder."""
    try:
        # Decode vector to text
        response = requests.post(
            f"{VEC2TEXT_DECODER_URL}/decode",
            json={"vectors": [vector.tolist()], "steps": 1, "subscribers": "jxe"},
            timeout=30
        )
        response.raise_for_status()
        decoded_text = response.json()["decoded_texts"][0]

        # Re-encode decoded text
        enc_response = requests.post(
            f"{VEC2TEXT_ENCODER_URL}/embed",
            json={"texts": [decoded_text]},
            timeout=30
        )
        enc_response.raise_for_status()
        decoded_vec = np.array(enc_response.json()["embeddings"][0])

        # Compute cosine similarity
        cosine = compute_cosine(vector, decoded_vec)

        # Display results
        status = "✅" if cosine > 0.65 else "❌"
        print(f"\n{'='*80}")
        print(f"Test #{idx} {status} (cosine: {cosine:.4f})")
        print(f"{'='*80}")
        print(f"Original:  {original_text[:120]}...")
        print(f"Decoded:   {decoded_text[:120]}...")
        print(f"Vector norm: {np.linalg.norm(vector):.6f}")

        return cosine >= 0.65

    except Exception as e:
        print(f"\n❌ Test #{idx} FAILED: {e}")
        return False

def main():
    print("="*80)
    print("Testing Database Vectors with Vec2Text Decoder")
    print("="*80)

    # Load NPZ file
    npz_path = Path("artifacts/lvm/wikipedia_42113_ordered.npz")
    if not npz_path.exists():
        print(f"❌ NPZ file not found: {npz_path}")
        sys.exit(1)

    data = np.load(npz_path)
    texts = data["texts"]
    vectors = data["vectors"]

    print(f"\n✓ Loaded {len(texts)} texts and {len(vectors)} vectors")
    print(f"  Vector shape: {vectors.shape}")
    print(f"  Vector dtype: {vectors.dtype}")

    # Test first 10
    print("\n" + "="*80)
    print("Testing FIRST 10 vectors")
    print("="*80)

    first_results = []
    for i in range(min(10, len(vectors))):
        result = test_vector(i, texts[i], vectors[i])
        first_results.append(result)

    # Test last 10
    print("\n" + "="*80)
    print("Testing LAST 10 vectors")
    print("="*80)

    last_results = []
    start_idx = max(0, len(vectors) - 10)
    for i in range(start_idx, len(vectors)):
        result = test_vector(i, texts[i], vectors[i])
        last_results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    first_passed = sum(first_results)
    last_passed = sum(last_results)
    total_passed = first_passed + last_passed
    total_tests = len(first_results) + len(last_results)

    print(f"\nFirst 10: {first_passed}/{len(first_results)} passed")
    print(f"Last 10:  {last_passed}/{len(last_results)} passed")
    print(f"Total:    {total_passed}/{total_tests} passed ({100*total_passed/total_tests:.1f}%)")

    if total_passed == total_tests:
        print("\n✅ SUCCESS: All vectors decode correctly with vec2text!")
        print("   Training data is ready for LVM training.")
    else:
        print(f"\n⚠️  WARNING: {total_tests - total_passed} vectors failed to decode properly")
        print("   May need to re-check vec2text encoder/decoder compatibility")

    print("="*80)

if __name__ == "__main__":
    main()
