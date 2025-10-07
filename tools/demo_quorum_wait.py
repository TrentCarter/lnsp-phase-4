#!/usr/bin/env python3
"""
Demo: Quorum Wait for Multi-Concept Retrieval

Shows how quorum_wait improves latency by not waiting for slow stragglers.

Scenario:
- User query: "machine learning training optimization techniques"
- 5 concepts extracted
- 3 concepts retrieve fast (50ms each)
- 2 concepts retrieve slow (500ms each)

Without quorum: Wait for ALL → 500ms total
With quorum (70%): Wait for 3/5 + 250ms grace → ~300ms total

Usage:
    python tools/demo_quorum_wait.py
"""

import asyncio
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lvm.quorum_wait import quorum_wait, QuorumResult


# Mock retrieval tasks
async def mock_retrieve_concept(concept: str, delay_ms: float, found: bool = True) -> dict:
    """Simulate concept retrieval with variable latency."""
    await asyncio.sleep(delay_ms / 1000.0)

    if not found:
        return {
            "concept": concept,
            "results": [],
            "confidence": 0.0,
            "latency_ms": delay_ms
        }

    return {
        "concept": concept,
        "results": [
            f"Result 1 for {concept}",
            f"Result 2 for {concept}",
        ],
        "confidence": 0.85,
        "latency_ms": delay_ms
    }


async def demo_without_quorum():
    """Show OLD behavior: wait for all concepts."""
    print("\n" + "="*60)
    print("DEMO 1: Without Quorum (wait for ALL)")
    print("="*60)

    concepts = [
        ("machine learning", 50),
        ("training", 50),
        ("optimization", 50),
        ("techniques", 500),  # SLOW
        ("neural networks", 500),  # SLOW
    ]

    print(f"\nQuery: 'machine learning training optimization techniques'")
    print(f"Concepts: {len(concepts)}")
    print(f"Strategy: Wait for ALL concepts\n")

    start = time.time()

    # Create tasks
    tasks = [
        asyncio.create_task(mock_retrieve_concept(concept, delay))
        for concept, delay in concepts
    ]

    # Wait for ALL (old behavior)
    print("Waiting for all concepts...")
    results = await asyncio.gather(*tasks)

    elapsed = (time.time() - start) * 1000

    print(f"\n✓ All {len(results)} concepts retrieved")
    print(f"⏱️  Total latency: {elapsed:.0f}ms")
    print(f"📊 Bottleneck: Slowest concept determines latency")

    return results, elapsed


async def demo_with_quorum():
    """Show NEW behavior: quorum wait with grace period."""
    print("\n" + "="*60)
    print("DEMO 2: With Quorum Wait (70% + 250ms grace)")
    print("="*60)

    concepts = [
        ("machine learning", 50),
        ("training", 50),
        ("optimization", 50),
        ("techniques", 500),  # SLOW
        ("neural networks", 500),  # SLOW
    ]

    print(f"\nQuery: 'machine learning training optimization techniques'")
    print(f"Concepts: {len(concepts)}")
    print(f"Strategy: Quorum Q=70% (4/5) + 250ms grace window\n")

    start = time.time()

    # Create tasks
    tasks = [
        asyncio.create_task(mock_retrieve_concept(concept, delay))
        for concept, delay in concepts
    ]

    # Quorum wait (new behavior)
    print("Waiting for quorum (70%)...")
    result = await quorum_wait(tasks, grace_window_sec=0.25, quorum_pct=0.70)

    elapsed = (time.time() - start) * 1000

    print(f"\n✓ Quorum met: {len(result.ready_predictions)}/{result.metrics['N']} concepts retrieved")
    print(f"⏱️  Total latency: {elapsed:.0f}ms")
    print(f"📊 Metrics: {result.metrics}")

    if result.metrics['quorum_met']:
        print(f"✅ SUCCESS: Proceeded with {len(result.ready_predictions)} results")
    else:
        print(f"⚠️  WARNING: Quorum not met, proceeding with partials")

    return result.ready_predictions, elapsed


async def demo_comparison():
    """Side-by-side comparison."""
    print("\n" + "="*60)
    print("COMPARISON: Quorum Wait vs Wait-For-All")
    print("="*60)

    # Run both demos
    _, latency_all = await demo_without_quorum()
    results_quorum, latency_quorum = await demo_with_quorum()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    speedup = (latency_all - latency_quorum) / latency_all * 100

    print(f"\nWait-for-all latency:  {latency_all:.0f}ms")
    print(f"Quorum wait latency:   {latency_quorum:.0f}ms")
    print(f"Speedup:               {speedup:.1f}%")
    print(f"\nResults quality:       {len(results_quorum)}/5 concepts (acceptable for 70% threshold)")

    print("\n💡 Key Insight:")
    print("   Quorum wait reduces p95 latency by not blocking on slow stragglers,")
    print("   while still maintaining high recall (3-4 out of 5 concepts).")


async def demo_tunable_parameters():
    """Show how to tune quorum parameters."""
    print("\n" + "="*60)
    print("DEMO 3: Tunable Parameters")
    print("="*60)

    concepts = [
        ("ai", 30),
        ("ml", 40),
        ("dl", 50),
        ("nlp", 400),  # slow
        ("cv", 500),   # slow
    ]

    configs = [
        {"quorum_pct": 0.60, "grace_window_sec": 0.1, "name": "Aggressive (60% + 100ms)"},
        {"quorum_pct": 0.70, "grace_window_sec": 0.25, "name": "Balanced (70% + 250ms)"},
        {"quorum_pct": 0.80, "grace_window_sec": 0.5, "name": "Conservative (80% + 500ms)"},
    ]

    print("\nTesting different quorum configurations:\n")

    for config in configs:
        tasks = [
            asyncio.create_task(mock_retrieve_concept(concept, delay))
            for concept, delay in concepts
        ]

        start = time.time()
        result = await quorum_wait(
            tasks,
            quorum_pct=config["quorum_pct"],
            grace_window_sec=config["grace_window_sec"]
        )
        elapsed = (time.time() - start) * 1000

        print(f"{config['name']}:")
        print(f"  Latency:  {elapsed:.0f}ms")
        print(f"  Results:  {len(result.ready_predictions)}/5")
        print(f"  Quorum:   {'✓ met' if result.metrics['quorum_met'] else '✗ not met'}")
        print()

    print("💡 Tuning Guidance:")
    print("   - Lower Q + short grace = lower latency, risk missing results")
    print("   - Higher Q + long grace = higher latency, more complete results")
    print("   - Production default: Q=70%, grace=250ms (balances both)")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("QUORUM WAIT DEMO")
    print("Tiny Bite #4 of LVM Inference Pipeline")
    print("="*60)

    await demo_comparison()
    await demo_tunable_parameters()

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Integrate quorum_wait into production retrieval pipeline")
    print("  2. Monitor metrics: quorum_met_rate, avg_latency, avg_results_count")
    print("  3. Tune parameters based on p95 latency targets")
    print()


if __name__ == "__main__":
    asyncio.run(main())
