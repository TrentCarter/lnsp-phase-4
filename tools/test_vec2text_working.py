#!/usr/bin/env python3
"""Quick test showing vec2text works correctly with CPU-only mode"""

import requests

test_texts = [
    "Water reflects light very differently from typical terrestrial materials.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can predict the next vector in a sequence.",
    "Wikipedia is a free online encyclopedia created by volunteers.",
    "The Eiffel Tower is located in Paris, France."
]

print("\n" + "="*80)
print("VEC2TEXT ROUND-TRIP TEST (CPU-ONLY MODE)")
print("="*80 + "\n")

for text in test_texts:
    response = requests.post(
        'http://localhost:8766/encode-decode',
        json={"texts": [text], "subscribers": "jxe,ielab", "steps": 5},
        timeout=120
    )

    if response.status_code != 200:
        print(f"ERROR: {response.status_code}")
        continue

    data = response.json()['results'][0]
    jxe_result = data['subscribers']['gtr → jxe']
    ielab_result = data['subscribers']['gtr → ielab']

    print(f"{'─'*80}")
    print(f"INPUT:\n  {text}\n")
    print(f"JXE OUTPUT:\n  {jxe_result['output']}")
    print(f"  Cosine: {jxe_result['cosine']:.4f}\n")
    print(f"IELAB OUTPUT:\n  {ielab_result['output']}")
    print(f"  Cosine: {ielab_result['cosine']:.4f}\n")

print(f"{'='*80}\n")
