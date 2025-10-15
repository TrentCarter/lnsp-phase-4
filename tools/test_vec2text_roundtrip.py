#!/usr/bin/env python3
"""
Vec2Text Round-Trip Test
========================

Test if vec2text works at all by doing:
1. Original text → GTR-T5 embedding
2. GTR-T5 embedding → JXE vec2text
3. GTR-T5 embedding → IELab vec2text

This bypasses the LVM to isolate vec2text issues.
"""

import requests
import numpy as np
from sentence_transformers import SentenceTransformer

def test_roundtrip(text, backend='jxe'):
    """Test text → GTR-T5 → vec2text → text."""

    # Step 1: Encode with GTR-T5
    print(f"\n{'='*80}")
    print(f"Testing {backend.upper()} vec2text")
    print(f"{'='*80}\n")
    print(f"Original text:\n  {text}\n")

    model = SentenceTransformer('sentence-transformers/gtr-t5-base')
    vector = model.encode([text])[0]
    print(f"✓ GTR-T5 encoded to {len(vector)}D vector\n")

    # Step 2: Decode with vec2text
    print(f"Decoding with {backend}...\n")

    response = requests.post(
        'http://localhost:8766/decode',
        json={
            'vectors': [vector.tolist()],
            'subscribers': backend,
            'steps': 1,
            'device': 'cpu'
        },
        timeout=60
    )

    if response.status_code != 200:
        print(f"✗ HTTP Error {response.status_code}")
        return

    data = response.json()

    # Extract result
    if 'results' in data and len(data['results']) > 0:
        result = data['results'][0]
        if 'subscribers' in result:
            key = f"gtr → {backend}"
            if key in result['subscribers']:
                output = result['subscribers'][key].get('output', 'ERROR')
                cosine = result['subscribers'][key].get('cosine', 0.0)

                print(f"Reconstructed text:\n  {output}\n")
                print(f"Vector cosine similarity: {cosine:.4f}")

                # Simple text comparison
                orig_words = set(text.lower().split())
                recon_words = set(output.lower().split())
                overlap = len(orig_words & recon_words)
                print(f"Word overlap: {overlap}/{len(orig_words)} words ({100*overlap/len(orig_words):.1f}%)")
                return output

    print("✗ Failed to decode")
    return None


def main():
    print("\n" + "█"*80)
    print("█  VEC2TEXT ROUND-TRIP TEST (GTR-T5 → Vec2Text)".center(80, " ") + "█")
    print("█"*80 + "\n")

    # Test with multiple example texts
    test_texts = [
        "Water reflects light very differently from typical terrestrial materials.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can predict the next vector in a sequence.",
        "Wikipedia is a free online encyclopedia created by volunteers.",
        "The Eiffel Tower is located in Paris, France."
    ]

    for text in test_texts:
        # Test JXE
        jxe_result = test_roundtrip(text, backend='jxe')

        # Test IELab
        ielab_result = test_roundtrip(text, backend='ielab')

        print("\n" + "─"*80 + "\n")


if __name__ == '__main__':
    main()
