#!/usr/bin/env python3
"""
Critical Test: Can training data target vectors be decoded by vec2text?

This determines if the problem is:
1. Training data encoded with wrong encoder → targets decode to gibberish
2. Model prediction issue → targets decode correctly, predictions don't
"""

import numpy as np
import requests
import sys

def decode_vector(vector):
    """Decode vector using vec2text"""
    response = requests.post(
        "http://localhost:8766/decode",
        json={
            "vectors": [vector.tolist()],
            "subscribers": "ielab",
            "steps": 1,
            "device": "cpu"
        },
        timeout=30
    )
    if response.status_code == 200:
        result = response.json()
        return result["results"][0]["subscribers"]["gtr → ielab"]["output"]
    return None

def main():
    print("=" * 80)
    print("Critical Test: Training Data Target Vector Decoding")
    print("=" * 80)
    print()

    # Load training data
    print("Loading training data...")
    data = np.load('artifacts/lvm/training_sequences_ctx5.npz', allow_pickle=True)

    target_vectors = data['target_vectors']
    target_texts = data['target_texts']

    print(f"✓ Loaded {len(target_vectors)} target vectors")
    print()

    # Sample 10 random targets
    num_samples = 10
    indices = np.random.choice(len(target_vectors), num_samples, replace=False)

    print(f"Testing {num_samples} random target vectors...")
    print()

    results = []

    for i, idx in enumerate(indices):
        print(f"Sample {i+1}/{num_samples} (idx: {idx})")

        target_vec = target_vectors[idx]
        expected_text = target_texts[idx]

        print(f"  Expected text: '{expected_text}'")

        # Decode the target vector
        decoded = decode_vector(target_vec)

        if decoded is None:
            print(f"  ✗ Decode failed")
            continue

        print(f"  Decoded text:  '{decoded}'")

        # Check if decoded matches expected
        if decoded.strip().lower() == expected_text.strip().lower():
            match = "✓ EXACT MATCH"
        elif expected_text.lower() in decoded.lower() or decoded.lower() in expected_text.lower():
            match = "○ Partial match"
        else:
            match = "✗ No match (gibberish)"

        print(f"  Result: {match}")
        print()

        results.append({
            'expected': expected_text,
            'decoded': decoded,
            'match': match
        })

    # Summary
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    exact_matches = sum(1 for r in results if "EXACT" in r['match'])
    partial_matches = sum(1 for r in results if "Partial" in r['match'])
    no_matches = sum(1 for r in results if "No match" in r['match'])

    print(f"Exact matches:   {exact_matches}/{num_samples}")
    print(f"Partial matches: {partial_matches}/{num_samples}")
    print(f"No matches:      {no_matches}/{num_samples}")
    print()

    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()

    if exact_matches >= 8:
        print("✓ TRAINING DATA IS GOOD")
        print("  → Target vectors decode correctly to expected text")
        print("  → Problem is model prediction, not training data")
        print()
        print("Recommended fix:")
        print("  - Add cycle consistency loss (Phase 3)")
        print("  - OR use more training data")
        print("  - OR try different architecture")
    elif exact_matches + partial_matches >= 5:
        print("⚠ TRAINING DATA PARTIALLY CORRECT")
        print("  → Some vectors decode correctly, some don't")
        print("  → Possible mixed encoder sources in training data")
        print()
        print("Recommended fix:")
        print("  - Re-extract training data with consistent encoder")
    else:
        print("✗ TRAINING DATA IS CORRUPTED")
        print("  → Target vectors decode to gibberish")
        print("  → Training data was encoded with WRONG encoder")
        print("  → Current encoder and training data are incompatible")
        print()
        print("CRITICAL FIX REQUIRED:")
        print("  1. Re-extract ALL training vectors using current GTR-T5 encoder")
        print("  2. Verify round-trip works: text → encode → decode → text")
        print("  3. Retrain models with corrected data")
        print()
        print("Without this fix, NO amount of model tuning will work!")

if __name__ == "__main__":
    main()
