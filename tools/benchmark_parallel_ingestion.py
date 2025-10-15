#!/usr/bin/env python3
"""
Benchmark parallel vs sequential chunk ingestion.

Tests the performance improvement from parallel processing of chunks
with the ingestion API.

Expected Results:
- Sequential (10 chunks): ~10-12 seconds
- Parallel (10 chunks):   ~1-2 seconds  (5-10x speedup)

Usage:
    # Make sure Ollama is running with concurrency enabled:
    export OLLAMA_NUM_PARALLEL=10
    ollama serve &

    # Run benchmark
    ./tools/benchmark_parallel_ingestion.py
"""

import os
import sys
import time
import json
import requests
from pathlib import Path

# Configuration
INGEST_API_URL = "http://localhost:8004"
NUM_CHUNKS = 10
TEST_DATASET_SOURCE = "benchmark_test"

# Sample chunks for benchmarking
SAMPLE_CHUNKS = [
    "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose.",
    "The Eiffel Tower was built in 1889 for the World's Fair and stands 330 meters tall.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "The human brain contains approximately 86 billion neurons that communicate via synapses.",
    "Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales.",
    "The Great Barrier Reef is the world's largest coral reef system, stretching over 2,300 kilometers.",
    "DNA replication is a fundamental process where genetic information is copied before cell division.",
    "The Industrial Revolution began in Britain in the late 18th century and transformed manufacturing.",
    "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
    "Antibiotics work by targeting specific bacterial processes while leaving human cells unharmed."
]

def check_service_health():
    """Check if ingestion API is running"""
    try:
        response = requests.get(f"{INGEST_API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def run_ingestion_test(chunks, skip_cpesh=True, test_name="Test"):
    """
    Run ingestion and measure performance.

    Args:
        chunks: List of chunk texts to ingest
        skip_cpesh: Skip CPESH extraction (faster, for benchmarking)
        test_name: Name for this test run

    Returns:
        dict with timing results
    """
    print(f"\n{'=' * 80}")
    print(f"{test_name}")
    print(f"{'=' * 80}")

    # Prepare request payload
    payload = {
        "chunks": [
            {"text": chunk_text, "source_document": "benchmark", "chunk_index": idx}
            for idx, chunk_text in enumerate(chunks)
        ],
        "dataset_source": TEST_DATASET_SOURCE,
        "batch_id": f"{test_name.lower().replace(' ', '_')}_{int(time.time())}",
        "skip_cpesh": skip_cpesh
    }

    print(f"Chunks to ingest: {len(chunks)}")
    print(f"CPESH: {'Disabled (fast mode)' if skip_cpesh else 'Enabled'}")
    print()
    print("Starting ingestion...")

    # Time the request
    start_time = time.perf_counter()
    try:
        response = requests.post(
            f"{INGEST_API_URL}/ingest",
            json=payload,
            timeout=120  # 2 minutes max
        )
        response.raise_for_status()
        elapsed_time = time.perf_counter() - start_time
    except requests.RequestException as e:
        print(f"❌ Ingestion failed: {e}")
        return None

    # Parse response
    result = response.json()

    # Display results
    print(f"✅ Ingestion completed in {elapsed_time:.2f}s")
    print()
    print(f"Results:")
    print(f"  Total chunks:    {result['total_chunks']}")
    print(f"  Successful:      {result['successful']}")
    print(f"  Failed:          {result['failed']}")
    print(f"  Server time:     {result['processing_time_ms']:.1f}ms")
    print(f"  Client time:     {elapsed_time * 1000:.1f}ms")
    print(f"  Per chunk:       {elapsed_time / len(chunks) * 1000:.1f}ms")
    print()

    # Timing breakdown from individual chunks
    if result['results']:
        first_result = result['results'][0]
        if 'timings_ms' in first_result and first_result['timings_ms']:
            print("Per-chunk timing breakdown (first chunk):")
            for step, ms in first_result['timings_ms'].items():
                print(f"  {step:20s}: {ms:.1f}ms")
            print()

    return {
        "test_name": test_name,
        "num_chunks": len(chunks),
        "elapsed_time_s": elapsed_time,
        "per_chunk_ms": elapsed_time / len(chunks) * 1000,
        "server_time_ms": result['processing_time_ms'],
        "successful": result['successful'],
        "failed": result['failed']
    }

def main():
    print("=" * 80)
    print("Parallel Ingestion Performance Benchmark")
    print("=" * 80)
    print()

    # Check if API is running
    print("Checking ingestion API health...")
    if not check_service_health():
        print(f"❌ Ingestion API not responding at {INGEST_API_URL}")
        print()
        print("Please start the API first:")
        print("  ./.venv/bin/uvicorn app.api.ingest_chunks:app --host 127.0.0.1 --port 8004")
        print()
        sys.exit(1)

    print(f"✅ Ingestion API is healthy at {INGEST_API_URL}")
    print()

    # Check Ollama
    print("Checking Ollama LLM endpoint...")
    try:
        ollama_response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if ollama_response.status_code == 200:
            print("✅ Ollama is running at http://localhost:11434")
        else:
            print("⚠️  Ollama may not be running correctly")
    except:
        print("❌ Ollama is not running at http://localhost:11434")
        print()
        print("Please start Ollama with parallel support:")
        print("  export OLLAMA_NUM_PARALLEL=10")
        print("  ollama serve")
        print()
        sys.exit(1)

    print()
    print("=" * 80)
    print("Running Benchmarks")
    print("=" * 80)

    # Test 1: Sequential (disable parallel processing)
    print()
    print("⚠️  NOTE: To test sequential vs parallel, you need to:")
    print("    1. Start API with LNSP_ENABLE_PARALLEL=false for sequential test")
    print("    2. Restart API with LNSP_ENABLE_PARALLEL=true for parallel test")
    print()
    print("For now, testing with current API configuration...")
    print()

    # Run benchmark
    chunks_to_test = SAMPLE_CHUNKS[:NUM_CHUNKS]
    result = run_ingestion_test(chunks_to_test, skip_cpesh=True, test_name="Parallel Ingestion Test")

    if result:
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print()
        print(f"Chunks ingested:     {result['num_chunks']}")
        print(f"Total time:          {result['elapsed_time_s']:.2f}s")
        print(f"Per chunk:           {result['per_chunk_ms']:.1f}ms")
        print(f"Throughput:          {result['num_chunks'] / result['elapsed_time_s']:.2f} chunks/s")
        print()

        # Expected performance
        print("Expected Performance:")
        print(f"  Sequential:  ~10-12s for {NUM_CHUNKS} chunks (1 chunk/s)")
        print(f"  Parallel:    ~1-2s for {NUM_CHUNKS} chunks (5-10 chunks/s)")
        print()

        if result['per_chunk_ms'] < 200:
            print("🎉 Excellent! Parallel processing is working (~10x speedup)")
        elif result['per_chunk_ms'] < 500:
            print("✅ Good! Some parallelization is happening (~2-5x speedup)")
        else:
            print("⚠️  Slow performance. Check:")
            print("    - Is LNSP_ENABLE_PARALLEL=true?")
            print("    - Is OLLAMA_NUM_PARALLEL=10 set for Ollama?")
            print("    - Is Ollama actually serving requests concurrently?")

    print()
    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
