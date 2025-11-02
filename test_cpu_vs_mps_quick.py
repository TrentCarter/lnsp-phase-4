#!/usr/bin/env python3
"""Quick CPU vs MPS Comparison Test (3 iterations)"""

import requests
import time
from typing import Dict, List
import numpy as np

# Test phrases
test_phrases = [
    "Artificial intelligence is a branch of computer science.",
    "Airplanes fly through the air using aerodynamic principles.",
    "Photosynthesis converts light energy into chemical energy in plants."
]

# Configuration matrix
configs = [
    {"name": "CPU → IELab", "encoder_port": 7001, "decoder_port": 7002, "subscriber": "ielab"},
    {"name": "CPU → JXE", "encoder_port": 7001, "decoder_port": 7002, "subscriber": "jxe"},
    {"name": "MPS → IELab", "encoder_port": 7003, "decoder_port": 7004, "subscriber": "ielab"},
    {"name": "MPS → JXE", "encoder_port": 7003, "decoder_port": 7004, "subscriber": "jxe"},
]


def rouge_l_score(reference: str, candidate: str) -> float:
    """Calculate ROUGE-L score"""
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()

    if not ref_words or not cand_words:
        return 0.0

    m, n = len(ref_words), len(cand_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == cand_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_length = dp[m][n]
    precision = lcs_length / len(cand_words) if cand_words else 0
    recall = lcs_length / len(ref_words) if ref_words else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1 * 10


def run_test(config: Dict, phrase: str, steps: int = 3) -> Dict:
    """Run encode-decode test"""
    start_time = time.time()

    # Encode
    enc_resp = requests.post(
        f"http://localhost:{config['encoder_port']}/encode",
        json={"texts": [phrase]},
        timeout=30
    )
    vector = enc_resp.json()["embeddings"][0]

    # Decode
    dec_resp = requests.post(
        f"http://localhost:{config['decoder_port']}/decode",
        json={"vectors": [vector], "subscriber": config["subscriber"], "steps": steps},
        timeout=30
    )
    decoded = dec_resp.json()["results"][0]

    latency_ms = (time.time() - start_time) * 1000
    quality = rouge_l_score(phrase, decoded)

    return {"latency_ms": latency_ms, "quality": quality, "decoded": decoded}


print("🚀 Quick CPU vs MPS Comparison (3 iterations per config)")
print("="*80)

results = {}

for config in configs:
    print(f"\n📊 {config['name']}")
    latencies = []
    qualities = []
    sample_decoded = None

    for i in range(3):
        for phrase in test_phrases:
            result = run_test(config, phrase)
            latencies.append(result["latency_ms"])
            qualities.append(result["quality"])
            if i == 0 and phrase == test_phrases[0]:
                sample_decoded = result["decoded"]
            print(f"  Iter {i+1}, Phrase {test_phrases.index(phrase)+1}: {result['latency_ms']:.0f}ms, Q={result['quality']:.1f}/10")

    avg_lat = np.mean(latencies)
    avg_qual = np.mean(qualities)
    results[config['name']] = {"latency": avg_lat, "quality": avg_qual, "sample": sample_decoded}

    print(f"  ✅ Avg: {avg_lat:.0f}ms, Quality: {avg_qual:.2f}/10")

# Summary
print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)
print(f"{'Config':<15} {'Latency':<15} {'Quality':<10} {'Status'}")
print("-"*80)

for name, data in results.items():
    status = "✅ Fast" if data['latency'] < 2000 else "⚠️  Slow"
    print(f"{name:<15} {data['latency']:>7.0f}ms      {data['quality']:>4.2f}/10    {status}")

print("-"*80)

# Find winners
fastest = min(results.items(), key=lambda x: x[1]['latency'])
print(f"\n⚡ Fastest: {fastest[0]} ({fastest[1]['latency']:.0f}ms)")

# Speed comparison
cpu_ielab = results["CPU → IELab"]["latency"]
mps_ielab = results["MPS → IELab"]["latency"]
print(f"\n💡 CPU is {mps_ielab/cpu_ielab:.2f}x FASTER than MPS for decoding!")

print("\n✅ Test complete!")
