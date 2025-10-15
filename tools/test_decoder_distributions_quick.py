#!/usr/bin/env python3
"""
Task 1.3: Decoder A/B/C Test (Quick Version)
============================================

Tests whether vec2text decoder is sensitive to L2 normalization.
Uses 2 samples for faster execution.

Test A: decoder(encoder(x)) → should reconstruct well
Test B: decoder(L2(encoder(x))) → if this fails, decoder expects raw distribution

Expected Result:
- Test A: Good reconstruction (baseline)
- Test B: Gibberish/poor reconstruction (SMOKING GUN for root cause)
"""

import numpy as np
import torch
import requests
import json
from typing import List


def encode_text_gtr(text: str) -> np.ndarray:
    """
    Encode text using GTR-T5 on port 8767 (vec2text-compatible encoder)
    Returns: 768D vector (RAW distribution, not L2-normalized)
    """
    response = requests.post(
        "http://127.0.0.1:8767/embed",
        json={"texts": [text]}
    )
    if response.status_code != 200:
        raise RuntimeError(f"GTR encoder failed: {response.text}")

    embeddings = response.json()["embeddings"]
    return np.array(embeddings[0], dtype=np.float32)


def decode_vector_vec2text(vector: np.ndarray, steps: int = 1) -> str:
    """
    Decode vector using vec2text on port 8766
    Returns: reconstructed text
    """
    response = requests.post(
        "http://127.0.0.1:8766/decode",
        json={
            "vectors": [vector.tolist()],
            "num_steps": steps,
            "subscriber": "ielab"  # Use ielab decoder
        }
    )
    if response.status_code != 200:
        raise RuntimeError(f"Vec2text decoder failed: {response.text}")

    result = response.json()
    return result["decoded_texts"][0]


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Force L2 norm = 1.0 (this is what breaks the decoder)"""
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return vector
    return vector / norm


def main():
    print("=" * 80)
    print("Task 1.3: Decoder A/B/C Test (QUICK VERSION - 2 samples)")
    print("=" * 80)
    print()

    # Test samples (reduced to 2)
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require large amounts of training data."
    ]

    print("Testing on 2 text samples for faster results...")
    print()

    results = {
        'test_a': [],  # encoder → decoder (baseline)
        'test_b': [],  # encoder → L2 → decoder (broken?)
    }

    for i, text in enumerate(test_texts, 1):
        print(f"{'='*80}")
        print(f"Sample {i}/2")
        print(f"{'='*80}")
        print(f"Original Text: {text}")
        print()

        # Encode with GTR-T5
        print("Encoding with GTR-T5...")
        vector_raw = encode_text_gtr(text)

        # L2 normalize
        vector_l2 = l2_normalize(vector_raw)

        print(f"Vector Statistics:")
        print(f"  Raw norm: {np.linalg.norm(vector_raw):.4f}")
        print(f"  L2 norm:  {np.linalg.norm(vector_l2):.4f}")
        print(f"  Raw mean: {vector_raw.mean():.6f}, std: {vector_raw.std():.6f}")
        print(f"  L2 mean:  {vector_l2.mean():.6f}, std: {vector_l2.std():.6f}")
        print()

        # Test A: decoder(encoder(x))
        print("TEST A: decoder(encoder(x)) [BASELINE - should work]")
        try:
            decoded_a = decode_vector_vec2text(vector_raw, steps=1)
            print(f"  Decoded: {decoded_a}")

            # Simple similarity check
            original_words = set(text.lower().split())
            decoded_words = set(decoded_a.lower().split())
            overlap = len(original_words & decoded_words) / len(original_words)
            print(f"  Word overlap: {overlap:.2%}")

            results['test_a'].append({
                'text': text,
                'decoded': decoded_a,
                'overlap': overlap
            })
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            results['test_a'].append({'text': text, 'decoded': None, 'error': str(e)})
        print()

        # Test B: decoder(L2(encoder(x)))
        print("TEST B: decoder(L2(encoder(x))) [SMOKING GUN - should fail]")
        try:
            decoded_b = decode_vector_vec2text(vector_l2, steps=1)
            print(f"  Decoded: {decoded_b}")

            original_words = set(text.lower().split())
            decoded_words = set(decoded_b.lower().split())
            overlap = len(original_words & decoded_words) / len(original_words)
            print(f"  Word overlap: {overlap:.2%}")

            results['test_b'].append({
                'text': text,
                'decoded': decoded_b,
                'overlap': overlap
            })
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            results['test_b'].append({'text': text, 'decoded': None, 'error': str(e)})
        print()

    print("=" * 80)
    print("Summary Results")
    print("=" * 80)

    # Test A results
    test_a_overlaps = [r['overlap'] for r in results['test_a'] if 'overlap' in r]
    print(f"\nTEST A (baseline): decoder(encoder(x))")
    print(f"  Average word overlap: {np.mean(test_a_overlaps):.2%}")
    print(f"  Quality: ", end="")
    if np.mean(test_a_overlaps) > 0.5:
        print("✅ GOOD (decoder works with raw distribution)")
    else:
        print("❌ POOR (unexpected - baseline should work)")

    # Test B results
    test_b_overlaps = [r['overlap'] for r in results['test_b'] if 'overlap' in r]
    print(f"\nTEST B (smoking gun): decoder(L2(encoder(x)))")
    print(f"  Average word overlap: {np.mean(test_b_overlaps):.2%}")
    print(f"  Quality: ", end="")
    if np.mean(test_b_overlaps) < 0.3:
        print("❌ POOR (CONFIRMED: L2 normalization breaks decoder!)")
        print("\n🔥 SMOKING GUN FOUND:")
        print("  - Vec2text decoder expects RAW encoder distribution")
        print("  - L2 normalization collapses variance decoder needs")
        print("  - This is why LVM outputs (L2-normalized) decode to gibberish")
    elif np.mean(test_b_overlaps) > 0.5:
        print("✅ GOOD (unexpected - decoder robust to normalization)")
    else:
        print("⚠️ OKAY (marginal degradation)")

    print("\n" + "=" * 80)
    print("Key Insight:")
    print("=" * 80)
    print()

    degradation = (np.mean(test_a_overlaps) - np.mean(test_b_overlaps)) / np.mean(test_a_overlaps)
    print(f"Degradation from L2 normalization: {degradation:.1%}")

    if degradation > 0.5:
        print("\n✅ DIAGNOSIS CONFIRMED:")
        print("  1. Vec2text decoder is trained on RAW GTR-T5 output distribution")
        print("  2. L2 normalization destroys per-dimension variance decoder relies on")
        print("  3. LVM must output TWO heads:")
        print("     - y_dec: RAW distribution for vec2text (port 8766)")
        print("     - y_cos: L2-normalized for retrieval/cosine metrics")
        print()
        print("NEXT STEPS:")
        print("  → Proceed to Task 2.1: Split output heads")
        print("  → Add moment-matching loss to preserve distribution")
        print("  → Retrain with y_dec matching GTR-T5 statistics")
    else:
        print("\n⚠️ UNEXPECTED RESULT:")
        print("  Vec2text decoder appears robust to L2 normalization.")
        print("  Need to investigate other causes of mode collapse.")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
