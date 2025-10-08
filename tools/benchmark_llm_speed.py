#!/usr/bin/env python3
"""
LLM Speed Benchmark: Llama 3.1:8b vs TinyLlama 1.1b
Tests tokens/sec for TMD-LS lane specialist architecture
"""

import requests
import json
import time
from typing import Dict, List, Tuple

# Test prompts of varying lengths
PROMPTS = [
    ("Short (3 tokens)", "What is AI?"),
    ("Medium (~10 tokens)", "Explain the concept of machine learning in detail."),
    ("Long (~50 tokens)",
     "Write a comprehensive explanation of quantum computing, including its principles, "
     "applications, and current limitations. Cover quantum superposition, entanglement, "
     "and how quantum gates differ from classical logic gates."),
]

def benchmark_model(model: str, port: int, prompt: str, prompt_name: str) -> Tuple[float, Dict]:
    """Benchmark a single model with a prompt."""
    print(f"Testing {model} on port {port}")
    print(f"Prompt: {prompt_name}")
    print("-" * 70)

    url = f"http://localhost:{port}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        # Extract timing data (in nanoseconds)
        total_duration = data.get('total_duration', 0)
        eval_duration = data.get('eval_duration', 0)
        prompt_eval_duration = data.get('prompt_eval_duration', 0)
        eval_count = data.get('eval_count', 0)
        prompt_eval_count = data.get('prompt_eval_count', 0)

        # Convert to seconds
        total_sec = total_duration / 1e9
        eval_sec = eval_duration / 1e9
        prompt_eval_sec = prompt_eval_duration / 1e9

        # Calculate tokens/sec
        tokens_per_sec = eval_count / eval_sec if eval_sec > 0 else 0
        prompt_tokens_per_sec = prompt_eval_count / prompt_eval_sec if prompt_eval_sec > 0 else 0

        print(f"  Total time: {total_sec:.3f}s")
        print(f"  Generation time: {eval_sec:.3f}s")
        print(f"  Prompt eval time: {prompt_eval_sec:.3f}s")
        print(f"  Tokens generated: {eval_count}")
        print(f"  Prompt tokens: {prompt_eval_count}")
        print(f"  ⚡ Generation speed: {tokens_per_sec:.2f} tokens/sec")
        print(f"  📖 Prompt processing: {prompt_tokens_per_sec:.2f} tokens/sec")
        print()

        return tokens_per_sec, data

    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        return 0, {}

def main():
    print("=" * 70)
    print("LLM SPEED BENCHMARK: Llama 3.1:8b vs TinyLlama 1.1b")
    print("=" * 70)
    print()

    # Check server availability
    print("Checking server availability...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        print("✅ Llama server running on port 11434")
    except:
        print("❌ Error: Ollama server not running on port 11434")
        print("Start with: ollama serve")
        return

    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print()

    # Store results
    llama_speeds = []
    tinyllama_speeds = []

    # Run benchmarks
    for i, (prompt_name, prompt) in enumerate(PROMPTS, 1):
        print("━" * 70)
        print(f"TEST {i}/{len(PROMPTS)}: {prompt_name}")
        print("━" * 70)
        print()

        print("🔹 LLAMA 3.1:8B")
        llama_speed, _ = benchmark_model("llama3.1:8b", 11434, prompt, prompt_name)
        llama_speeds.append(llama_speed)

        print("🔸 TINYLLAMA 1.1B")
        tinyllama_speed, _ = benchmark_model("tinyllama:1.1b", 11434, prompt, prompt_name)
        tinyllama_speeds.append(tinyllama_speed)

        # Calculate speedup
        if llama_speed > 0 and tinyllama_speed > 0:
            speedup = tinyllama_speed / llama_speed
            print(f"⚡ TinyLlama is {speedup:.2f}x faster for this test")
        print()

    # Calculate averages
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    llama_avg = sum(s for s in llama_speeds if s > 0) / len([s for s in llama_speeds if s > 0]) if any(s > 0 for s in llama_speeds) else 0
    tinyllama_avg = sum(s for s in tinyllama_speeds if s > 0) / len([s for s in tinyllama_speeds if s > 0]) if any(s > 0 for s in tinyllama_speeds) else 0

    print("Average Generation Speed:")
    print(f"  🔹 Llama 3.1:8b    : {llama_avg:.2f} tokens/sec")
    print(f"  🔸 TinyLlama 1.1b  : {tinyllama_avg:.2f} tokens/sec")
    print()

    if llama_avg > 0 and tinyllama_avg > 0:
        overall_speedup = tinyllama_avg / llama_avg
        print(f"⚡ Overall: TinyLlama is {overall_speedup:.2f}x faster")
        print()

        # Compare to PRD claims
        print("PRD_TMD-LS.md Claims:")
        print("  • Llama ~200-300 tokens/sec")
        print("  • TinyLlama ~600-800 tokens/sec")
        print()
        print("Actual Results:")
        print(f"  • Llama: {llama_avg:.2f} tokens/sec")
        print(f"  • TinyLlama: {tinyllama_avg:.2f} tokens/sec")
        print()

        # Validation
        if 200 <= llama_avg <= 400:
            print("✅ Llama 3.1:8b within expected range")
        else:
            print(f"⚠️  Llama 3.1:8b outside expected range (200-400 tok/s)")

        if 500 <= tinyllama_avg <= 1000:
            print("✅ TinyLlama within expected range")
        else:
            print(f"⚠️  TinyLlama outside expected range (500-1000 tok/s)")

    print()
    print("=" * 70)
    import platform
    print(f"Hardware: {platform.processor()}")
    print(f"Platform: {platform.system()} {platform.release()}")
    import datetime
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
