#!/usr/bin/env python3
"""
Comprehensive LLM Speed Benchmark: All TMD-LS Lane Specialist Models
Tests: Llama 3.1:8b, TinyLlama 1.1b, Phi3:mini, Granite3-MoE 1b
"""

import requests
import json
import time
from typing import Dict, List, Tuple

# Models to benchmark
MODELS = [
    ("llama3.1:8b", "Llama 3.1 8B", "General purpose, complex reasoning"),
    ("tinyllama:1.1b", "TinyLlama 1.1B", "Ultra-fast specialist"),
    ("phi3:mini", "Phi3 Mini 3.8B", "Precision, code generation"),
    ("granite3-moe:1b", "Granite3 MoE 1B", "IBM MoE, low latency"),
]

# Test prompts of varying lengths
PROMPTS = [
    ("Short (3 tokens)", "What is AI?"),
    ("Medium (~10 tokens)", "Explain the concept of machine learning in detail."),
    ("Long (~50 tokens)",
     "Write a comprehensive explanation of quantum computing, including its principles, "
     "applications, and current limitations. Cover quantum superposition, entanglement, "
     "and how quantum gates differ from classical logic gates."),
]

def benchmark_model(model: str, port: int, prompt: str, prompt_name: str) -> Tuple[float, float, Dict]:
    """Benchmark a single model with a prompt."""
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
        eval_sec = eval_duration / 1e9
        prompt_eval_sec = prompt_eval_duration / 1e9

        # Calculate tokens/sec
        tokens_per_sec = eval_count / eval_sec if eval_sec > 0 else 0
        prompt_tokens_per_sec = prompt_eval_count / prompt_eval_sec if prompt_eval_sec > 0 else 0

        return tokens_per_sec, prompt_tokens_per_sec, data

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0, 0, {}

def main():
    print("=" * 80)
    print("COMPREHENSIVE LLM SPEED BENCHMARK: TMD-LS Lane Specialists")
    print("=" * 80)
    print()

    # Check server availability
    print("Checking server availability...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        print("✅ Ollama server running on port 11434")
    except:
        print("❌ Error: Ollama server not running on port 11434")
        print("Start with: ollama serve")
        return

    print()

    # Store all results
    all_results = {model[0]: [] for model in MODELS}
    all_prompt_speeds = {model[0]: [] for model in MODELS}

    # Run benchmarks
    for i, (prompt_name, prompt) in enumerate(PROMPTS, 1):
        print("=" * 80)
        print(f"TEST {i}/{len(PROMPTS)}: {prompt_name}")
        print("=" * 80)
        print()

        for model_id, model_name, description in MODELS:
            print(f"🔹 {model_name.upper()}")
            print(f"   {description}")
            print(f"   Model: {model_id}")
            print("-" * 80)

            gen_speed, prompt_speed, data = benchmark_model(model_id, 11434, prompt, prompt_name)

            if gen_speed > 0:
                eval_count = data.get('eval_count', 0)
                print(f"  ⚡ Generation: {gen_speed:.2f} tok/s ({eval_count} tokens)")
                print(f"  📖 Prompt processing: {prompt_speed:.2f} tok/s")
                all_results[model_id].append(gen_speed)
                all_prompt_speeds[model_id].append(prompt_speed)
            else:
                print(f"  ❌ Benchmark failed")

            print()

    # Calculate and display summary
    print("=" * 80)
    print("SUMMARY - AVERAGE GENERATION SPEED")
    print("=" * 80)
    print()

    results_table = []
    for model_id, model_name, description in MODELS:
        speeds = [s for s in all_results[model_id] if s > 0]
        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            results_table.append((model_name, model_id, avg_speed))
        else:
            results_table.append((model_name, model_id, 0))

    # Sort by speed (fastest first)
    results_table.sort(key=lambda x: x[2], reverse=True)

    print(f"{'Rank':<6} {'Model':<25} {'Avg Speed':<15} {'Speedup vs Llama'}")
    print("-" * 80)

    llama_speed = next((r[2] for r in results_table if "Llama 3.1" in r[0]), 1)

    for rank, (name, model_id, speed) in enumerate(results_table, 1):
        if speed > 0:
            speedup = speed / llama_speed if llama_speed > 0 else 1.0
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{emoji} #{rank:<4} {name:<25} {speed:>8.2f} tok/s   {speedup:>5.2f}x")
        else:
            print(f"  #{rank:<4} {name:<25} {'FAILED':>15}")

    print()
    print("=" * 80)
    print("DETAILED BREAKDOWN")
    print("=" * 80)
    print()

    for model_id, model_name, description in MODELS:
        speeds = all_results[model_id]
        if speeds and any(s > 0 for s in speeds):
            avg = sum(s for s in speeds if s > 0) / len([s for s in speeds if s > 0])
            print(f"{model_name}:")
            print(f"  Average: {avg:.2f} tok/s")
            for i, speed in enumerate(speeds, 1):
                if speed > 0:
                    print(f"  Test {i}: {speed:.2f} tok/s")
            print()

    print("=" * 80)
    print("TMD-LS LANE ASSIGNMENTS (RECOMMENDED)")
    print("=" * 80)
    print()

    # Sort results for lane assignment
    sorted_models = sorted(results_table, key=lambda x: x[2], reverse=True)

    print("Based on performance characteristics:\n")
    for name, model_id, speed in sorted_models:
        if "TinyLlama" in name or "Granite" in name:
            print(f"🏎️  HIGH-SPEED LANES (L1, L2, L6):")
            print(f"    {name} ({speed:.0f} tok/s)")
            print(f"    → Fact retrieval, simple extraction, batch ingestion")
            print()
        elif "Phi3" in name:
            print(f"⚙️  PRECISION LANES (L3, L5):")
            print(f"    {name} ({speed:.0f} tok/s)")
            print(f"    → Code generation, structured output, schema validation")
            print()
        elif "Llama 3.1" in name:
            print(f"🧠 REASONING LANES (L4):")
            print(f"    {name} ({speed:.0f} tok/s)")
            print(f"    → Complex reasoning, detailed narrative, Echo Loop validation")
            print()

    print("=" * 80)
    import platform
    print(f"Hardware: {platform.processor()}")
    print(f"Platform: {platform.system()} {platform.release()}")
    import datetime
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
