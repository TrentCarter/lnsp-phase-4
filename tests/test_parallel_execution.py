#!/usr/bin/env python3
"""
Test parallel execution of Programmer Pool via Manager-Code-01.

This test:
1. Submits multiple concurrent tasks to Manager-Code-01
2. Verifies multiple Programmers execute simultaneously
3. Measures speedup vs sequential execution
4. Validates receipts and metrics
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import httpx


# Configuration
MANAGER_CODE_01_URL = "http://localhost:6141"
NUM_TASKS = 3  # Start with 3 tasks for faster testing
TIMEOUT = 120  # seconds


async def submit_task(
    client: httpx.AsyncClient,
    task_id: int,
    job_card: Dict[str, Any]
) -> Dict[str, Any]:
    """Submit a single task to Manager-Code-01 and wait for completion."""
    start_time = time.time()

    print(f"[Task {task_id}] Submitting: {job_card['task'][:50]}")

    try:
        # Submit job card to Manager
        response = await client.post(
            f"{MANAGER_CODE_01_URL}/submit",
            json={"job_card": job_card},
            timeout=5.0
        )
        response.raise_for_status()
        result = response.json()

        job_card_id = result.get("job_card_id")
        print(f"[Task {task_id}] Submitted - Job Card ID: {job_card_id}")

        # Poll for completion
        max_polls = 40  # 40 polls * 3s = 120s total
        for poll in range(max_polls):
            await asyncio.sleep(3)

            status_response = await client.get(
                f"{MANAGER_CODE_01_URL}/status/{job_card_id}",
                timeout=5.0
            )
            status_response.raise_for_status()
            status = status_response.json()

            state = status.get("state", "unknown")

            if state in ["completed", "failed"]:
                elapsed = time.time() - start_time
                success = state == "completed"
                print(f"[Task {task_id}] {state.upper()} in {elapsed:.2f}s")

                return {
                    "task_id": task_id,
                    "elapsed": elapsed,
                    "result": status,
                    "success": success
                }

        # Timeout
        elapsed = time.time() - start_time
        print(f"[Task {task_id}] TIMEOUT after {elapsed:.2f}s")
        return {
            "task_id": task_id,
            "elapsed": elapsed,
            "error": "Timeout waiting for completion",
            "success": False
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Task {task_id}] Failed in {elapsed:.2f}s - Error: {e}")
        return {
            "task_id": task_id,
            "elapsed": elapsed,
            "error": str(e),
            "success": False
        }


async def test_parallel_execution():
    """Test parallel execution with multiple concurrent tasks."""

    print("=" * 80)
    print("PARALLEL EXECUTION TEST")
    print("=" * 80)

    # Create test job cards - simple file modifications
    job_cards = []
    for i in range(NUM_TASKS):
        job_card = {
            "id": f"test-job-{i+1}",
            "task": f"Add a comment '# Parallel test {i+1}' to the top of a new test file",
            "goal": f"Create tests/parallel_test_{i+1}.py with a simple function",
            "entry_file": f"tests/parallel_test_{i+1}.py",
            "requirements": [
                f"Create tests/parallel_test_{i+1}.py",
                f"Add a comment '# Parallel test {i+1}'",
                "Add a simple test function"
            ],
            "constraints": ["Keep it simple", "No external dependencies"],
            "llm_config": {
                "provider": "ollama",
                "model": "qwen2.5-coder:7b",
                "max_tokens": 2000
            },
            "priority": "normal",
            "estimated_complexity": "low"
        }
        job_cards.append(job_card)

    # Test 1: Parallel Execution
    print(f"\n[TEST 1] Submitting {NUM_TASKS} tasks in PARALLEL")
    print("-" * 80)

    parallel_start = time.time()

    async with httpx.AsyncClient() as client:
        # Submit all tasks concurrently
        parallel_results = await asyncio.gather(*[
            submit_task(client, i+1, job_card)
            for i, job_card in enumerate(job_cards)
        ])

    parallel_elapsed = time.time() - parallel_start

    print("-" * 80)
    print(f"[TEST 1] Total parallel time: {parallel_elapsed:.2f}s")

    # Test 2: Sequential Execution (for comparison)
    print(f"\n[TEST 2] Submitting {NUM_TASKS} tasks SEQUENTIALLY")
    print("-" * 80)

    sequential_start = time.time()
    sequential_results = []

    async with httpx.AsyncClient() as client:
        for i, job_card in enumerate(job_cards):
            result = await submit_task(client, i+1, job_card)
            sequential_results.append(result)

    sequential_elapsed = time.time() - sequential_start

    print("-" * 80)
    print(f"[TEST 2] Total sequential time: {sequential_elapsed:.2f}s")

    # Analysis
    print("\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)

    # Success rates
    parallel_success = sum(1 for r in parallel_results if r["success"])
    sequential_success = sum(1 for r in sequential_results if r["success"])

    print(f"\nSuccess Rate:")
    print(f"  Parallel:   {parallel_success}/{NUM_TASKS} ({100*parallel_success/NUM_TASKS:.1f}%)")
    print(f"  Sequential: {sequential_success}/{NUM_TASKS} ({100*sequential_success/NUM_TASKS:.1f}%)")

    # Timing analysis
    speedup = sequential_elapsed / parallel_elapsed if parallel_elapsed > 0 else 0

    print(f"\nTiming:")
    print(f"  Parallel:   {parallel_elapsed:.2f}s")
    print(f"  Sequential: {sequential_elapsed:.2f}s")
    print(f"  Speedup:    {speedup:.2f}x")

    # Per-task timing
    avg_parallel = sum(r["elapsed"] for r in parallel_results) / len(parallel_results)
    avg_sequential = sum(r["elapsed"] for r in sequential_results) / len(sequential_results)

    print(f"\nAverage Task Duration:")
    print(f"  Parallel:   {avg_parallel:.2f}s")
    print(f"  Sequential: {avg_sequential:.2f}s")

    # Programmer utilization check
    print(f"\nProgrammer Utilization:")
    print(f"  Expected concurrent: {min(NUM_TASKS, 10)} Programmers")
    print(f"  Theoretical speedup: {min(NUM_TASKS, 10):.1f}x")
    print(f"  Actual speedup:      {speedup:.2f}x")
    print(f"  Efficiency:          {100*speedup/min(NUM_TASKS, 10):.1f}%")

    # Check logs for concurrent execution
    print(f"\n[INFO] Check Programmer logs at artifacts/logs/programmer_*.log")
    print(f"[INFO] Look for overlapping timestamps to verify parallel execution")

    # Validate receipts
    receipts_dir = Path("artifacts/programmer_receipts")
    if receipts_dir.exists():
        receipts = list(receipts_dir.glob("*.jsonl"))
        print(f"\n[INFO] Generated {len(receipts)} receipt files")
        if receipts:
            latest = max(receipts, key=lambda p: p.stat().st_mtime)
            print(f"[INFO] Latest receipt: {latest.name}")

    # Final verdict
    print("\n" + "=" * 80)
    if speedup >= 2.0:
        print("✅ PASS - Parallel execution achieving significant speedup (>2x)")
    elif speedup >= 1.5:
        print("⚠️  PARTIAL - Some parallelization detected (>1.5x), but below target")
    else:
        print("❌ FAIL - No significant parallel speedup detected (<1.5x)")
    print("=" * 80)

    return {
        "parallel_elapsed": parallel_elapsed,
        "sequential_elapsed": sequential_elapsed,
        "speedup": speedup,
        "parallel_success_rate": parallel_success / NUM_TASKS,
        "sequential_success_rate": sequential_success / NUM_TASKS
    }


async def test_programmer_pool_status():
    """Check Programmer Pool status before and after test."""

    print("\n" + "=" * 80)
    print("PROGRAMMER POOL STATUS")
    print("=" * 80)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{MANAGER_CODE_01_URL}/health")
            response.raise_for_status()
            health = response.json()

            print(f"\nManager-Code-01 Health:")
            print(f"  Status: {health.get('status', 'unknown')}")
            print(f"  Using Programmer Pool: {health.get('using_programmer_pool', False)}")

            # Try to get pool stats if available
            if "programmer_pool_stats" in health:
                stats = health["programmer_pool_stats"]
                print(f"\nProgrammer Pool Stats:")
                print(f"  Total Programmers: {stats.get('total_programmers', 0)}")
                print(f"  Idle:   {stats.get('idle', 0)}")
                print(f"  Busy:   {stats.get('busy', 0)}")
                print(f"  Failed: {stats.get('failed', 0)}")

        except Exception as e:
            print(f"⚠️  Could not get pool status: {e}")


async def main():
    """Run all tests."""

    # Check pool status
    await test_programmer_pool_status()

    # Run parallel execution test
    results = await test_parallel_execution()

    # Save results
    results_file = Path("artifacts/parallel_execution_test_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "num_tasks": NUM_TASKS,
            **results
        }, f, indent=2)

    print(f"\n[INFO] Results saved to {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
