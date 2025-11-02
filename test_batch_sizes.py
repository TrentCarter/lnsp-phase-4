#!/usr/bin/env python3
"""
Test if batching improves MPS performance

Tests batch sizes: 1, 5, 10, 20 vectors at once
"""

import requests
import time
import numpy as np

test_phrases = [
    "Artificial intelligence is a branch of computer science.",
    "Airplanes fly through the air using aerodynamic principles.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The Earth orbits around the Sun once every year.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Shakespeare wrote many famous plays and sonnets.",
    "Mount Everest is the highest mountain on Earth.",
    "DNA contains the genetic instructions for all living organisms.",
    "The speed of light is approximately 300,000 kilometers per second.",
    "Beethoven composed nine symphonies during his lifetime."
]

batch_sizes = [1, 5, 10]

print("🚀 Testing Batch Size Impact on MPS Performance")
print("="*80)

for batch_size in batch_sizes:
    print(f"\n📊 Batch Size: {batch_size}")

    # Prepare batch
    texts = test_phrases[:batch_size]

    # Encode all texts
    encode_start = time.time()
    encode_resp = requests.post(
        "http://localhost:7003/encode",
        json={"texts": texts},
        timeout=60
    )
    vectors = encode_resp.json()["embeddings"]
    encode_time = (time.time() - encode_start) * 1000

    # Decode all vectors at once (batched)
    decode_start = time.time()
    decode_resp = requests.post(
        "http://localhost:7004/decode",
        json={"vectors": vectors, "subscriber": "ielab", "steps": 3},
        timeout=120
    )
    decoded_texts = decode_resp.json()["results"]
    decode_time = (time.time() - decode_start) * 1000

    total_time = encode_time + decode_time
    per_item = total_time / batch_size

    print(f"  Encode: {encode_time:>7.0f}ms ({encode_time/batch_size:>5.0f}ms per item)")
    print(f"  Decode: {decode_time:>7.0f}ms ({decode_time/batch_size:>5.0f}ms per item)")
    print(f"  Total:  {total_time:>7.0f}ms ({per_item:>5.0f}ms per item)")
    print(f"  Throughput: {batch_size / (total_time/1000):.2f} items/sec")

print("\n" + "="*80)
print("💡 Analysis:")
print("If per-item time decreases with batch size, MPS benefits from batching!")
print("If per-item time stays constant, batching doesn't help (sequential bottleneck)")
print("="*80)
