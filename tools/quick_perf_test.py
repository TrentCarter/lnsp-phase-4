#!/usr/bin/env python3
"""
Quick Performance Test: Measure real-time metrics for all TMD-LS lane specialists
Tests: First token latency, tokens/sec, total time, output quality
"""

import requests
import json
import time
from typing import Dict, Tuple

# Models on their ports
MODELS = [
    ("llama3.1:8b", 11434, "Llama 3.1 8B"),
    ("tinyllama:1.1b", 11435, "TinyLlama 1.1B"),
    ("phi3:mini", 11436, "Phi3 Mini 3.8B"),
    ("granite3-moe:1b", 11437, "Granite3 MoE 1B"),
]

# Test prompt
TEST_PROMPT = "Explain quantum computing in 2-3 sentences."

def test_model(model: str, port: int, name: str) -> Dict:
    """Test a model and return performance metrics"""
    url = f"http://localhost:{port}/api/generate"
    payload = {
        "model": model,
        "prompt": TEST_PROMPT,
        "stream": False
    }

    try:
        # Time the request
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=60)
        end_time = time.time()

        response.raise_for_status()
        data = response.json()

        # Extract metrics
        total_duration_ns = data.get('total_duration', 0)
        load_duration_ns = data.get('load_duration', 0)
        prompt_eval_duration_ns = data.get('prompt_eval_duration', 0)
        eval_duration_ns = data.get('eval_duration', 0)

        eval_count = data.get('eval_count', 0)
        prompt_eval_count = data.get('prompt_eval_count', 0)

        response_text = data.get('response', '')[:100]  # First 100 chars

        # Convert to seconds
        total_sec = total_duration_ns / 1e9
        load_sec = load_duration_ns / 1e9
        prompt_eval_sec = prompt_eval_duration_ns / 1e9
        eval_sec = eval_duration_ns / 1e9

        # Calculate metrics
        tokens_per_sec = eval_count / eval_sec if eval_sec > 0 else 0

        # First token latency = load + prompt eval time
        first_token_latency = load_sec + prompt_eval_sec

        # Wall clock time
        wall_clock_sec = end_time - start_time

        return {
            'success': True,
            'model': model,
            'name': name,
            'port': port,
            'tokens_per_sec': tokens_per_sec,
            'first_token_latency': first_token_latency,
            'total_time': total_sec,
            'wall_clock_time': wall_clock_sec,
            'eval_count': eval_count,
            'prompt_eval_count': prompt_eval_count,
            'load_time': load_sec,
            'prompt_eval_time': prompt_eval_sec,
            'eval_time': eval_sec,
            'response_preview': response_text
        }

    except Exception as e:
        return {
            'success': False,
            'model': model,
            'name': name,
            'port': port,
            'error': str(e)
        }

def main():
    print("=" * 80)
    print("QUICK PERFORMANCE TEST: TMD-LS Lane Specialists")
    print("=" * 80)
    print(f"Test Prompt: '{TEST_PROMPT}'")
    print()

    results = []

    for model, port, name in MODELS:
        print(f"Testing {name} on port {port}...")
        result = test_model(model, port, name)
        results.append(result)

        if result['success']:
            print(f"  ✅ Complete: {result['tokens_per_sec']:.2f} tok/s")
        else:
            print(f"  ❌ Failed: {result['error']}")
        print()

    # Display results table
    print("=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print()

    # Sort by tokens/sec
    successful = [r for r in results if r['success']]
    successful.sort(key=lambda x: x['tokens_per_sec'], reverse=True)

    print(f"{'Rank':<6} {'Model':<20} {'Port':<6} {'Tok/s':<10} {'1st Token':<12} {'Total Time':<12}")
    print("-" * 80)

    for i, r in enumerate(successful, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{emoji} #{i:<4} {r['name']:<20} {r['port']:<6} "
              f"{r['tokens_per_sec']:>7.2f}   {r['first_token_latency']:>9.3f}s   {r['total_time']:>9.3f}s")

    print()
    print("=" * 80)
    print("DETAILED METRICS")
    print("=" * 80)
    print()

    for r in successful:
        print(f"{r['name']} (Port {r['port']})")
        print(f"  Generation Speed:     {r['tokens_per_sec']:.2f} tok/s")
        print(f"  First Token Latency:  {r['first_token_latency']:.3f}s")
        print(f"  Total Time:           {r['total_time']:.3f}s")
        print(f"  Wall Clock Time:      {r['wall_clock_time']:.3f}s")
        print(f"  Tokens Generated:     {r['eval_count']}")
        print(f"  Prompt Tokens:        {r['prompt_eval_count']}")
        print(f"  Load Time:            {r['load_time']:.3f}s")
        print(f"  Prompt Eval Time:     {r['prompt_eval_time']:.3f}s")
        print(f"  Generation Time:      {r['eval_time']:.3f}s")
        print(f"  Response Preview:     {r['response_preview']}...")
        print()

    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()

    if len(successful) >= 2:
        fastest = successful[0]
        baseline = next((r for r in successful if 'llama3.1' in r['model']), successful[-1])

        speedup = fastest['tokens_per_sec'] / baseline['tokens_per_sec']
        latency_improvement = baseline['first_token_latency'] / fastest['first_token_latency']

        print(f"🏆 Fastest Model: {fastest['name']}")
        print(f"   • Generation: {fastest['tokens_per_sec']:.2f} tok/s")
        print(f"   • Speedup: {speedup:.2f}x vs {baseline['name']}")
        print(f"   • First Token: {fastest['first_token_latency']:.3f}s ({latency_improvement:.2f}x faster)")
        print()

        print(f"🐢 Baseline: {baseline['name']}")
        print(f"   • Generation: {baseline['tokens_per_sec']:.2f} tok/s")
        print(f"   • First Token: {baseline['first_token_latency']:.3f}s")
        print()

    # Average first token latency
    avg_first_token = sum(r['first_token_latency'] for r in successful) / len(successful)
    print(f"📊 Average First Token Latency: {avg_first_token:.3f}s")
    print(f"📊 Average Generation Speed: {sum(r['tokens_per_sec'] for r in successful) / len(successful):.2f} tok/s")
    print()

    print("=" * 80)

if __name__ == "__main__":
    main()
