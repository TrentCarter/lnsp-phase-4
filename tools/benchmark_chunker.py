#!/usr/bin/env python3
"""
Quick benchmark to test chunker performance
Runs 5 consecutive tests to identify warm-up issues
"""

import requests
import time
import json
from statistics import mean, stdev

CHUNKER_URL = "http://localhost:8001"

TEST_TEXT = """Photosynthesis is the process by which plants convert sunlight into chemical energy using chlorophyll. This amazing process occurs in specialized organelles called chloroplasts, which are found in plant cells and some algae."""

def test_chunker(mode="semantic", breakpoint_threshold=50, min_chunk_size=20):
    """Single test of chunker"""
    start = time.time()

    response = requests.post(
        f"{CHUNKER_URL}/chunk",
        json={
            "text": TEST_TEXT,
            "mode": mode,
            "breakpoint_threshold": breakpoint_threshold,
            "min_chunk_size": min_chunk_size,
            "max_chunk_size": 320
        },
        timeout=10
    )

    latency_ms = (time.time() - start) * 1000

    if response.status_code == 200:
        data = response.json()
        return {
            "success": True,
            "latency_ms": latency_ms,
            "num_chunks": data.get("total_chunks", 0),
            "server_time_ms": data.get("processing_time_ms", 0)
        }
    else:
        return {
            "success": False,
            "latency_ms": latency_ms,
            "error": f"HTTP {response.status_code}"
        }

def run_benchmark(n=5, mode="semantic"):
    """Run N tests and analyze results"""
    print(f"\n{'='*80}")
    print(f"CHUNKER PERFORMANCE BENCHMARK - {mode.upper()} MODE")
    print(f"{'='*80}\n")

    print(f"Running {n} consecutive tests...\n")

    results = []
    for i in range(1, n + 1):
        result = test_chunker(mode=mode)
        results.append(result)

        if result["success"]:
            print(f"Test {i}:")
            print(f"  Total latency:  {result['latency_ms']:.2f}ms")
            print(f"  Server time:    {result['server_time_ms']:.2f}ms")
            print(f"  Network+JSON:   {result['latency_ms'] - result['server_time_ms']:.2f}ms")
            print(f"  Chunks created: {result['num_chunks']}")
            print()
        else:
            print(f"Test {i}: ❌ FAILED - {result['error']}\n")

    # Analysis
    successful = [r for r in results if r["success"]]

    if successful:
        total_latencies = [r["latency_ms"] for r in successful]
        server_times = [r["server_time_ms"] for r in successful]

        print(f"\n{'='*80}")
        print("ANALYSIS")
        print(f"{'='*80}\n")

        print(f"Total Latency (client-side):")
        print(f"  Min:     {min(total_latencies):.2f}ms")
        print(f"  Max:     {max(total_latencies):.2f}ms")
        print(f"  Mean:    {mean(total_latencies):.2f}ms")
        if len(total_latencies) > 1:
            print(f"  StdDev:  {stdev(total_latencies):.2f}ms")
        print()

        print(f"Server Processing Time:")
        print(f"  Min:     {min(server_times):.2f}ms")
        print(f"  Max:     {max(server_times):.2f}ms")
        print(f"  Mean:    {mean(server_times):.2f}ms")
        if len(server_times) > 1:
            print(f"  StdDev:  {stdev(server_times):.2f}ms")
        print()

        # Check for warm-up effect
        if len(successful) >= 2:
            first = total_latencies[0]
            rest_mean = mean(total_latencies[1:])

            if first > rest_mean * 1.5:
                print(f"⚠️  WARM-UP DETECTED:")
                print(f"   First request: {first:.2f}ms")
                print(f"   Subsequent:    {rest_mean:.2f}ms")
                print(f"   Speedup:       {first / rest_mean:.1f}x slower on first run")
            else:
                print(f"✅ NO WARM-UP ISSUES - Consistent performance")

        print()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark chunker performance")
    parser.add_argument("--n", type=int, default=5, help="Number of tests to run")
    parser.add_argument("--mode", default="semantic", choices=["simple", "semantic", "proposition", "hybrid"])

    args = parser.parse_args()

    # Check health first
    try:
        response = requests.get(f"{CHUNKER_URL}/health", timeout=2)
        if response.status_code != 200:
            print(f"❌ Chunker not healthy: HTTP {response.status_code}")
            exit(1)
    except Exception as e:
        print(f"❌ Chunker not reachable: {e}")
        print(f"\nStart it with:")
        print(f"  ./.venv/bin/uvicorn app.api.chunking:app --host 127.0.0.1 --port 8001")
        exit(1)

    run_benchmark(n=args.n, mode=args.mode)
