#!/usr/bin/env python3
"""
Benchmark TMD Extraction: TinyLlama 1.1B vs Llama 3.1 8B

Tests accuracy and speed of different LLM models for TMD extraction.
Helps determine if we can safely use smaller/faster models.

Usage:
    LNSP_LLM_ENDPOINT=http://localhost:11434 ./tools/benchmark_tmd_models.py
"""

import os
import sys
import time
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
import statistics

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.tmd_router import route_concept


@dataclass
class TestCase:
    """Test case for TMD extraction."""
    text: str
    expected_domain: int
    expected_task: int
    category: str  # For grouping results


# Test cases covering different domains/tasks
# CORRECT DOMAIN MAPPING (from configs/llm_prompts/llm_prompts_master.json):
# 0=Science, 1=Mathematics, 2=Technology, 3=Engineering, 4=Medicine,
# 5=Psychology, 6=Philosophy, 7=History, 8=Literature, 9=Art,
# 10=Economics, 11=Law, 12=Politics, 13=Education, 14=Environment, 15=Software
TEST_CASES = [
    # Scientific concepts (domain=0)
    TestCase(
        text="The water cycle involves evaporation, condensation, and precipitation",
        expected_domain=0,  # Science (or 14=Environment)
        expected_task=8,    # Summarization (or 3=Causal Inference)
        category="science"
    ),
    TestCase(
        text="Photosynthesis converts light energy into chemical energy in plants",
        expected_domain=0,  # Science
        expected_task=3,    # Causal Inference
        category="science"
    ),
    TestCase(
        text="DNA replication occurs during the S phase of the cell cycle",
        expected_domain=4,  # Medicine/Biology
        expected_task=0,    # Fact Retrieval
        category="science"
    ),

    # Software/Tech concepts (domain=15 or 2)
    TestCase(
        text="A binary search tree maintains sorted order with O(log n) lookup time",
        expected_domain=15,  # Software (or 2=Technology)
        expected_task=1,     # Definition Matching
        category="tech"
    ),
    TestCase(
        text="REST APIs use HTTP methods like GET, POST, PUT, and DELETE",
        expected_domain=15,  # Software
        expected_task=5,     # Entity Recognition
        category="tech"
    ),
    TestCase(
        text="Python list comprehensions provide concise syntax for creating lists",
        expected_domain=15,  # Software
        expected_task=1,     # Definition Matching
        category="tech"
    ),

    # Mathematical concepts (domain=1)
    TestCase(
        text="The quadratic formula solves equations of the form ax² + bx + c = 0",
        expected_domain=1,  # Mathematics
        expected_task=1,    # Definition Matching
        category="math"
    ),
    TestCase(
        text="Prime numbers are divisible only by 1 and themselves",
        expected_domain=1,  # Mathematics
        expected_task=1,    # Definition Matching
        category="math"
    ),

    # Historical/Social concepts (domain=7 or 12)
    TestCase(
        text="The Industrial Revolution transformed manufacturing in the 18th century",
        expected_domain=7,  # History
        expected_task=0,    # Fact Retrieval
        category="history"
    ),
    TestCase(
        text="Democracy is a system of government where citizens vote for leaders",
        expected_domain=12,  # Politics (or 7=History)
        expected_task=1,     # Definition Matching
        category="history"
    ),
]


@dataclass
class BenchmarkResult:
    """Results for a single model."""
    model_name: str
    total_tests: int
    domain_correct: int
    task_correct: int
    both_correct: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    total_time_s: float
    results: List[Dict]

    @property
    def domain_accuracy(self) -> float:
        return self.domain_correct / self.total_tests if self.total_tests > 0 else 0.0

    @property
    def task_accuracy(self) -> float:
        return self.task_correct / self.total_tests if self.total_tests > 0 else 0.0

    @property
    def combined_accuracy(self) -> float:
        return self.both_correct / self.total_tests if self.total_tests > 0 else 0.0


def test_model(model_name: str, test_cases: List[TestCase]) -> BenchmarkResult:
    """Test TMD extraction with a specific model."""

    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    endpoint = os.getenv("LNSP_LLM_ENDPOINT", "http://localhost:11434")

    # WARM-UP: Run 3 queries to eliminate cold start latency
    print(f"\nWarming up model (3 queries)...")
    warmup_text = "Machine learning is a subset of artificial intelligence"
    for i in range(3):
        try:
            route_concept(
                concept_text=warmup_text,
                use_cache=False,
                llm_endpoint=endpoint,
                llm_model=model_name
            )
            print(f"  Warm-up {i+1}/3 complete")
        except Exception as e:
            print(f"  Warm-up {i+1}/3 failed: {e}")
    print(f"✓ Model warmed up, starting benchmark...\n")

    results = []
    latencies = []
    domain_correct = 0
    task_correct = 0
    both_correct = 0

    start_total = time.time()

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}/{len(test_cases)}: {test_case.category}")
        print(f"  Text: {test_case.text[:60]}...")

        # Measure latency
        start = time.time()
        try:
            result = route_concept(
                concept_text=test_case.text,
                use_cache=False,  # Disable cache for accurate benchmarking
                llm_endpoint=endpoint,
                llm_model=model_name
            )
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

            # Check accuracy
            domain_match = result['domain_code'] == test_case.expected_domain
            task_match = result['task_code'] == test_case.expected_task

            if domain_match:
                domain_correct += 1
            if task_match:
                task_correct += 1
            if domain_match and task_match:
                both_correct += 1

            result_record = {
                'test_num': i,
                'category': test_case.category,
                'text': test_case.text,
                'expected_domain': test_case.expected_domain,
                'expected_task': test_case.expected_task,
                'actual_domain': result['domain_code'],
                'actual_task': result['task_code'],
                'domain_match': domain_match,
                'task_match': task_match,
                'latency_ms': latency_ms
            }
            results.append(result_record)

            # Print result
            domain_status = "✓" if domain_match else "✗"
            task_status = "✓" if task_match else "✗"
            print(f"  Domain: {result['domain_code']} (expected {test_case.expected_domain}) {domain_status}")
            print(f"  Task: {result['task_code']} (expected {test_case.expected_task}) {task_status}")
            print(f"  Latency: {latency_ms:.1f}ms")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'test_num': i,
                'category': test_case.category,
                'error': str(e)
            })

    total_time_s = time.time() - start_total

    return BenchmarkResult(
        model_name=model_name,
        total_tests=len(test_cases),
        domain_correct=domain_correct,
        task_correct=task_correct,
        both_correct=both_correct,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0,
        min_latency_ms=min(latencies) if latencies else 0,
        max_latency_ms=max(latencies) if latencies else 0,
        total_time_s=total_time_s,
        results=results
    )


def print_comparison(results: List[BenchmarkResult]):
    """Print comparison table."""

    print("\n" + "="*80)
    print("BENCHMARK RESULTS: TMD EXTRACTION MODEL COMPARISON")
    print("="*80)

    # Accuracy comparison
    print("\n📊 ACCURACY COMPARISON:")
    print("-" * 80)
    print(f"{'Model':<20} {'Domain':<15} {'Task':<15} {'Both Correct':<15}")
    print("-" * 80)

    for result in results:
        print(f"{result.model_name:<20} "
              f"{result.domain_accuracy*100:>6.1f}% ({result.domain_correct}/{result.total_tests})"
              f"     {result.task_accuracy*100:>6.1f}% ({result.task_correct}/{result.total_tests})"
              f"     {result.combined_accuracy*100:>6.1f}% ({result.both_correct}/{result.total_tests})")

    # Speed comparison
    print("\n⚡ SPEED COMPARISON:")
    print("-" * 80)
    print(f"{'Model':<20} {'Avg Latency':<15} {'Min':<12} {'Max':<12} {'Total Time':<12}")
    print("-" * 80)

    for result in results:
        print(f"{result.model_name:<20} "
              f"{result.avg_latency_ms:>8.1f} ms     "
              f"{result.min_latency_ms:>6.1f} ms   "
              f"{result.max_latency_ms:>6.1f} ms   "
              f"{result.total_time_s:>6.1f} s")

    # Speedup calculation
    if len(results) == 2:
        baseline = results[0]
        optimized = results[1]
        speedup = baseline.avg_latency_ms / optimized.avg_latency_ms if optimized.avg_latency_ms > 0 else 0
        time_saved = baseline.total_time_s - optimized.total_time_s

        print(f"\n🚀 PERFORMANCE IMPROVEMENT:")
        print(f"   Speedup: {speedup:.1f}x faster")
        print(f"   Time saved: {time_saved:.1f}s ({time_saved/baseline.total_time_s*100:.0f}% faster)")
        print(f"   Latency reduction: {baseline.avg_latency_ms - optimized.avg_latency_ms:.1f}ms")

    # Recommendation
    print("\n💡 RECOMMENDATION:")
    print("-" * 80)

    if len(results) == 2:
        baseline = results[0]
        optimized = results[1]

        # Check if optimized model maintains >90% accuracy
        accuracy_threshold = 0.9
        if optimized.combined_accuracy >= accuracy_threshold:
            print(f"✅ RECOMMENDED: Use {optimized.model_name}")
            print(f"   • Maintains high accuracy ({optimized.combined_accuracy*100:.1f}%)")
            print(f"   • Significantly faster ({speedup:.1f}x speedup)")
            print(f"   • Will reduce pipeline latency by ~{time_saved/baseline.total_time_s*100:.0f}%")
        elif optimized.combined_accuracy >= 0.7:
            print(f"⚠️  CONDITIONAL: {optimized.model_name} may be acceptable")
            print(f"   • Moderate accuracy ({optimized.combined_accuracy*100:.1f}%)")
            print(f"   • Much faster ({speedup:.1f}x speedup)")
            print(f"   • Consider testing with more diverse examples")
        else:
            print(f"❌ NOT RECOMMENDED: Stick with {baseline.model_name}")
            print(f"   • {optimized.model_name} accuracy too low ({optimized.combined_accuracy*100:.1f}%)")
            print(f"   • Speed gain not worth accuracy loss")

    print("-" * 80)


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save benchmark results to JSON."""

    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'models': [
            {
                'model_name': r.model_name,
                'total_tests': r.total_tests,
                'domain_accuracy': r.domain_accuracy,
                'task_accuracy': r.task_accuracy,
                'combined_accuracy': r.combined_accuracy,
                'avg_latency_ms': r.avg_latency_ms,
                'min_latency_ms': r.min_latency_ms,
                'max_latency_ms': r.max_latency_ms,
                'total_time_s': r.total_time_s,
                'results': r.results
            }
            for r in results
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")


def main():
    """Main benchmark runner."""

    print("🔬 TMD EXTRACTION MODEL BENCHMARK")
    print("="*80)
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"LLM endpoint: {os.getenv('LNSP_LLM_ENDPOINT', 'http://localhost:11434')}")

    # Test both models
    models_to_test = [
        "llama3.1:8b",      # Baseline (current)
        "tinyllama:1.1b",   # Optimized (proposed)
    ]

    all_results = []

    for model_name in models_to_test:
        result = test_model(model_name, TEST_CASES)
        all_results.append(result)

    # Print comparison
    print_comparison(all_results)

    # Save results
    output_path = "artifacts/tmd_model_benchmark.json"
    os.makedirs("artifacts", exist_ok=True)
    save_results(all_results, output_path)

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
