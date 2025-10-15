#!/usr/bin/env python3
"""
Vec2Text Performance Benchmark
Tests text→768D and 768D→text performance to identify cold start issues
"""

import asyncio
import json
import statistics
import time
from typing import Dict, List, Tuple

import aiohttp
import numpy as np


class Vec2TextBenchmark:
    def __init__(self):
        self.embedding_url = "http://localhost:8767"
        self.decoding_url = "http://localhost:8766"
        self.test_texts = [
            "Photosynthesis is the process by which plants convert sunlight into chemical energy",
            "Machine learning is a subset of artificial intelligence that enables computers to learn",
            "The theory of relativity revolutionized our understanding of space and time",
            "Climate change is one of the most pressing challenges facing humanity today",
            "Quantum computing promises to solve problems that are intractable for classical computers"
        ] * 10  # 50 test samples

    async def benchmark_text_to_embedding(self, num_requests: int = 100) -> Dict:
        """Benchmark text → 768D embedding performance"""
        print(f"🔄 Benchmarking text → 768D encoding ({num_requests} requests)...")

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_requests):
                text = self.test_texts[i % len(self.test_texts)]
                tasks.append(self._single_embedding_request(session, text, i))

            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            # Process results
            latencies = []
            successful_requests = 0

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"❌ Request {i} failed: {result}")
                else:
                    latencies.append(result['latency'])
                    successful_requests += 1

            # Calculate statistics
            if latencies:
                avg_latency = statistics.mean(latencies)
                median_latency = statistics.median(latencies)
                p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                throughput = successful_requests / total_time * 1000  # requests per second
            else:
                avg_latency = median_latency = p95_latency = min_latency = max_latency = throughput = 0

            return {
                'total_requests': num_requests,
                'successful_requests': successful_requests,
                'total_time': total_time,
                'throughput_rps': throughput,
                'latency_ms': {
                    'avg': avg_latency,
                    'median': median_latency,
                    'p95': p95_latency,
                    'min': min_latency,
                    'max': max_latency
                }
            }

    async def benchmark_embedding_to_text(self, num_requests: int = 100) -> Dict:
        """Benchmark 768D → text decoding performance"""
        print(f"🔄 Benchmarking 768D → text decoding ({num_requests} requests)...")

        # First get embeddings for all test texts
        embeddings = await self._get_test_embeddings()

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_requests):
                embedding = embeddings[i % len(embeddings)]
                tasks.append(self._single_decoding_request(session, embedding, i))

            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            # Process results
            latencies = []
            successful_requests = 0

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"❌ Request {i} failed: {result}")
                else:
                    latencies.append(result['latency'])
                    successful_requests += 1

            # Calculate statistics
            if latencies:
                avg_latency = statistics.mean(latencies)
                median_latency = statistics.median(latencies)
                p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                throughput = successful_requests / total_time * 1000  # requests per second
            else:
                avg_latency = median_latency = p95_latency = min_latency = max_latency = throughput = 0

            return {
                'total_requests': num_requests,
                'successful_requests': successful_requests,
                'total_time': total_time,
                'throughput_rps': throughput,
                'latency_ms': {
                    'avg': avg_latency,
                    'median': median_latency,
                    'p95': p95_latency,
                    'min': min_latency,
                    'max': max_latency
                }
            }

    async def _get_test_embeddings(self) -> List[List[float]]:
        """Get embeddings for test texts"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for text in self.test_texts[:10]:  # Get embeddings for first 10 texts
                tasks.append(self._single_embedding_request(session, text))

            results = await asyncio.gather(*tasks)
            return [result['embedding'] for result in results]

    async def _single_embedding_request(self, session: aiohttp.ClientSession, text: str, request_id: int = 0) -> Dict:
        """Make a single embedding request and measure latency"""
        payload = {
            "texts": [text],
            "normalize": True
        }

        start_time = time.time()
        try:
            async with session.post(f"{self.embedding_url}/embed", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    latency = (time.time() - start_time) * 1000
                    return {
                        'success': True,
                        'latency': latency,
                        'embedding': result['embeddings'][0]
                    }
                else:
                    error_text = await response.text()
                    latency = (time.time() - start_time) * 1000
                    return {
                        'success': False,
                        'latency': latency,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {
                'success': False,
                'latency': latency,
                'error': str(e)
            }

    async def _single_decoding_request(self, session: aiohttp.ClientSession, embedding: List[float], request_id: int = 0) -> Dict:
        """Make a single decoding request and measure latency"""
        payload = {
            "vectors": [embedding],
            "subscribers": "jxe",
            "steps": 1,
            "device": "cpu"
        }

        start_time = time.time()
        try:
            async with session.post(f"{self.decoding_url}/decode", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    latency = (time.time() - start_time) * 1000
                    return {
                        'success': True,
                        'latency': latency,
                        'decoded_text': result['results'][0]['jxe']['decoded_text']
                    }
                else:
                    error_text = await response.text()
                    latency = (time.time() - start_time) * 1000
                    return {
                        'success': False,
                        'latency': latency,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {
                'success': False,
                'latency': latency,
                'error': str(e)
            }

    async def run_full_benchmark(self, num_requests: int = 50) -> Dict:
        """Run complete benchmark suite"""
        print("🚀 Starting Vec2Text Performance Benchmark")
        print("=" * 60)

        # Test encoding (text → 768D)
        encoding_results = await self.benchmark_text_to_embedding(num_requests)

        print("\n" + "=" * 60)

        # Test decoding (768D → text)
        decoding_results = await self.benchmark_embedding_to_text(num_requests)

        print("\n" + "=" * 60)
        print("📊 BENCHMARK RESULTS SUMMARY")
        print("=" * 60)

        print("📝 Text → 768D Encoding (Port 8767):")
        print(f"   Throughput: {encoding_results['throughput_rps']".1f"} requests/sec")
        print(f"   Avg Latency: {encoding_results['latency_ms']['avg']".1f"}ms")
        print(f"   Median Latency: {encoding_results['latency_ms']['median']".1f"}ms")
        print(f"   P95 Latency: {encoding_results['latency_ms']['p95']".1f"}ms")
        print(f"   Success Rate: {encoding_results['successful_requests']}/{encoding_results['total_requests']}")

        print("\n🔤 768D → Text Decoding (Port 8766):")
        print(f"   Throughput: {decoding_results['throughput_rps']".1f"} requests/sec")
        print(f"   Avg Latency: {decoding_results['latency_ms']['avg']".1f"}ms")
        print(f"   Median Latency: {decoding_results['latency_ms']['median']".1f"}ms")
        print(f"   P95 Latency: {decoding_results['latency_ms']['p95']".1f"}ms")
        print(f"   Success Rate: {decoding_results['successful_requests']}/{decoding_results['total_requests']}")

        # Performance comparison
        encoding_throughput = encoding_results['throughput_rps']
        decoding_throughput = decoding_results['throughput_rps']

        if encoding_throughput > 0 and decoding_throughput > 0:
            slowdown_factor = encoding_throughput / decoding_throughput
            print(f"\n⚡ Performance Analysis:")
            print(f"   Decoding is {slowdown_factor".1f"}x slower than encoding")
            print(f"   Encoding handles {encoding_throughput".1f"} req/sec vs {decoding_throughput".1f"} req/sec for decoding")

        return {
            'encoding': encoding_results,
            'decoding': decoding_results
        }


async def main():
    import sys

    num_requests = 50
    if len(sys.argv) > 1:
        try:
            num_requests = int(sys.argv[1])
        except ValueError:
            print("Usage: python vec2text_benchmark.py [num_requests]")
            sys.exit(1)

    benchmark = Vec2TextBenchmark()
    results = await benchmark.run_full_benchmark(num_requests)

    # Save results to JSON file
    with open('/tmp/vec2text_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: /tmp/vec2text_benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
