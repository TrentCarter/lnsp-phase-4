#!/usr/bin/env python3
"""
Test Fresh Encode → Decode Cycle
==================================

Tests if the vec2text ecosystem is working by doing a fresh round-trip:
Text → Port 8767 Encoder → Port 8766 Decoder → Text
"""

import requests
import time

test_texts = [
    "Water reflects light very differently from typical terrestrial materials.",
    "Clouds keep Earth cool by reflecting sunlight.",
    "Trees also impact climate in extremely complicated ways through evapotranspiration."
]

print("\n" + "="*80)
print("Fresh Encode → Decode Round-Trip Test")
print("="*80 + "\n")

for i, text in enumerate(test_texts, 1):
    print(f"Test {i}/3")
    print(f"{'─'*80}")
    print(f"Original Text:\n  {text}\n")

    # Step 1: Encode via port 8767
    try:
        encode_response = requests.post(
            'http://localhost:8767/embed',
            json={'texts': [text], 'normalize': True},
            timeout=10
        )

        if encode_response.status_code != 200:
            print(f"✗ Encoding failed: HTTP {encode_response.status_code}\n")
            continue

        encode_data = encode_response.json()
        vector = encode_data['embeddings'][0]

        print(f"Encoding:")
        print(f"  Dimension: {encode_data['dimension']}")
        print(f"  Encoder: {encode_data['encoder']}\n")

    except Exception as e:
        print(f"✗ Encoding error: {e}\n")
        continue

    # Step 2: Decode via port 8766
    try:
        decode_response = requests.post(
            'http://localhost:8766/decode',
            json={
                'vectors': [vector],
                'subscribers': 'jxe',
                'steps': 1,
                'device': 'cpu'
            },
            timeout=30
        )

        if decode_response.status_code != 200:
            print(f"✗ Decoding failed: HTTP {decode_response.status_code}\n")
            continue

        decode_data = decode_response.json()
        result = decode_data['results'][0]['subscribers']['gtr → jxe']

        if result['status'] == 'success':
            reconstructed = result['output']
            cosine = result['cosine']

            print(f"Decoding:")
            print(f"  Cosine: {cosine:.4f}")
            print(f"  Reconstructed:\n    {reconstructed}\n")

            if cosine >= 0.65:
                print(f"✅ EXCELLENT - Vec2text ecosystem working\n")
            elif cosine >= 0.50:
                print(f"⚠️  MODERATE - Acceptable but could be better\n")
            else:
                print(f"❌ POOR - Vec2text ecosystem broken\n")

        else:
            print(f"✗ Decode error: {result.get('error', 'Unknown')}\n")

    except Exception as e:
        print(f"✗ Decoding error: {e}\n")

print("="*80 + "\n")
