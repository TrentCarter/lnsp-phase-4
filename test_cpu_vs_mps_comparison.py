#!/usr/bin/env python3
"""
Comprehensive CPU vs MPS Comparison Test

Tests all 4 configurations:
1. CPU Encoder (7001) + CPU Decoder (7002) with IELab
2. CPU Encoder (7001) + CPU Decoder (7002) with JXE
3. MPS Encoder (7003) + MPS Decoder (7004) with IELab
4. MPS Encoder (7003) + MPS Decoder (7004) with JXE

Runs 10x iterations to average out loading time and get reliable performance metrics.
"""

import requests
import time
from typing import Dict, List
import numpy as np


# Test phrases (same as before)
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
    """Calculate ROUGE-L score (basic implementation)"""
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()

    if not ref_words or not cand_words:
        return 0.0

    # Find longest common subsequence
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
    return f1 * 10  # Scale to 0-10


def encode_text(text: str, encoder_port: int) -> List[float]:
    """Encode text using specified encoder port"""
    response = requests.post(
        f"http://localhost:{encoder_port}/encode",
        json={"texts": [text]},
        timeout=30
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def decode_vector(vector: List[float], decoder_port: int, subscriber: str, steps: int = 3) -> str:
    """Decode vector using specified decoder port and subscriber"""
    response = requests.post(
        f"http://localhost:{decoder_port}/decode",
        json={
            "vectors": [vector],
            "subscriber": subscriber,
            "steps": steps
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["results"][0]


def run_single_test(config: Dict, phrase: str, steps: int = 3) -> Dict:
    """Run a single encode-decode test"""
    start_time = time.time()

    # Encode
    vector = encode_text(phrase, config["encoder_port"])

    # Decode
    decoded = decode_vector(vector, config["decoder_port"], config["subscriber"], steps)

    latency_ms = (time.time() - start_time) * 1000
    quality = rouge_l_score(phrase, decoded)

    return {
        "original": phrase,
        "decoded": decoded,
        "latency_ms": latency_ms,
        "quality": quality
    }


def run_comprehensive_test(iterations: int = 10, steps: int = 3):
    """Run comprehensive test across all configurations"""
    print(f"🚀 Running Comprehensive CPU vs MPS Comparison")
    print(f"   Iterations: {iterations} per configuration")
    print(f"   Steps: {steps}")
    print(f"   Test phrases: {len(test_phrases)}")
    print()

    results = {}

    for config in configs:
        config_name = config["name"]
        print(f"📊 Testing {config_name}...")

        config_results = {
            "latencies": [],
            "qualities": [],
            "samples": []
        }

        # Run multiple iterations to average out variance
        for iteration in range(iterations):
            for phrase in test_phrases:
                try:
                    result = run_single_test(config, phrase, steps)
                    config_results["latencies"].append(result["latency_ms"])
                    config_results["qualities"].append(result["quality"])

                    # Save first sample for display
                    if iteration == 0:
                        config_results["samples"].append({
                            "original": result["original"],
                            "decoded": result["decoded"],
                            "quality": result["quality"]
                        })

                    print(f"   Iteration {iteration+1}/{iterations}, Phrase {test_phrases.index(phrase)+1}: "
                          f"{result['latency_ms']:.0f}ms, Quality {result['quality']:.2f}/10")

                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    continue

        # Calculate statistics
        config_results["avg_latency"] = np.mean(config_results["latencies"])
        config_results["std_latency"] = np.std(config_results["latencies"])
        config_results["avg_quality"] = np.mean(config_results["qualities"])

        results[config_name] = config_results
        print(f"   ✅ Avg Latency: {config_results['avg_latency']:.0f}ms ± {config_results['std_latency']:.0f}ms")
        print(f"   ✅ Avg Quality: {config_results['avg_quality']:.2f}/10")
        print()

    return results


def print_summary(results: Dict):
    """Print comprehensive summary of results"""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)

    print("\n🎯 Performance Comparison Table")
    print("-" * 80)
    print(f"{'Configuration':<20} {'Avg Latency (ms)':<20} {'Avg Quality':<15} {'Status':<10}")
    print("-" * 80)

    for config_name, config_results in results.items():
        avg_latency = config_results['avg_latency']
        avg_quality = config_results['avg_quality']

        status = "✅ Good" if avg_quality >= 4.0 else "⚠️  Check"

        print(f"{config_name:<20} {avg_latency:>8.0f} ± {config_results['std_latency']:.0f}ms {avg_quality:>8.2f}/10      {status}")

    print("-" * 80)

    # Find best configurations
    print("\n🏆 Best Configurations")
    print("-" * 80)

    best_latency_config = min(results.items(), key=lambda x: x[1]['avg_latency'])
    best_quality_config = max(results.items(), key=lambda x: x[1]['avg_quality'])

    print(f"⚡ Fastest: {best_latency_config[0]} ({best_latency_config[1]['avg_latency']:.0f}ms)")
    print(f"🎯 Best Quality: {best_quality_config[0]} ({best_quality_config[1]['avg_quality']:.2f}/10)")

    # Sample outputs
    print("\n📝 Sample Outputs (First Iteration)")
    print("-" * 80)

    for config_name, config_results in results.items():
        print(f"\n{config_name}:")
        for sample in config_results['samples'][:2]:  # Show first 2 samples
            print(f"  Original:  {sample['original']}")
            print(f"  Decoded:   {sample['decoded']}")
            print(f"  Quality:   {sample['quality']:.2f}/10")
            print()

    # Comparison insights
    print("\n💡 Key Insights")
    print("-" * 80)

    # Compare CPU vs MPS for same decoder
    cpu_ielab = results.get("CPU → IELab")
    mps_ielab = results.get("MPS → IELab")

    if cpu_ielab and mps_ielab:
        speedup = cpu_ielab['avg_latency'] / mps_ielab['avg_latency']
        if speedup > 1:
            print(f"✅ MPS is {speedup:.2f}x FASTER than CPU for IELab decoder")
        else:
            print(f"⚠️  CPU is {1/speedup:.2f}x FASTER than MPS for IELab decoder")

    cpu_jxe = results.get("CPU → JXE")
    mps_jxe = results.get("MPS → JXE")

    if cpu_jxe and mps_jxe:
        speedup = cpu_jxe['avg_latency'] / mps_jxe['avg_latency']
        if speedup > 1:
            print(f"✅ MPS is {speedup:.2f}x FASTER than CPU for JXE decoder")
        else:
            print(f"⚠️  CPU is {1/speedup:.2f}x FASTER than MPS for JXE decoder")

    print("\n✅ Test completed successfully!")
    print("="*80)


if __name__ == "__main__":
    # Run comprehensive test with 10 iterations
    results = run_comprehensive_test(iterations=10, steps=3)

    # Print summary
    print_summary(results)
