#!/usr/bin/env python3
"""
Benchmark Ingestion Pipeline
Measures: Time, RAM, Functionality, Resource Usage
"""

import json
import time
import psutil
import requests
from pathlib import Path
from typing import Dict, List
import sys

class IngestionBenchmark:
    def __init__(self, api_url: str = "http://localhost:8004"):
        self.api_url = api_url
        self.process = psutil.Process()
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "peak_memory_mb": 0,
            "avg_memory_mb": 0,
            "concepts_ingested": 0,
            "chains_processed": 0,
            "errors": [],
            "latencies": [],
            "cpesh_count": 0,
            "tmd_count": 0,
            "vector_count": 0,
        }
        self.memory_samples = []

    def measure_memory(self):
        """Sample current memory usage"""
        mem = self.process.memory_info().rss / 1024 / 1024  # MB
        self.memory_samples.append(mem)
        if mem > self.metrics["peak_memory_mb"]:
            self.metrics["peak_memory_mb"] = mem

    def ingest_chain(self, chain_data: Dict) -> Dict:
        """Ingest a single chain and measure"""
        self.measure_memory()

        start = time.time()
        try:
            # Convert chain to individual concepts
            concepts = chain_data.get("concepts", [])
            source = chain_data.get("source", "unknown")

            results = []
            # Build chunks array for batch ingestion
            chunks = []
            for idx, concept_text in enumerate(concepts):
                chunks.append({
                    "text": concept_text,
                    "chunk_index": idx,
                    "source_document": chain_data.get("chain_id"),
                    "metadata": {
                        "chain_id": chain_data.get("chain_id"),
                        "position": idx,
                        "total": len(concepts)
                    },
                    "parent_cpe_ids": [],  # Will be linked by API
                    "child_cpe_ids": []
                })

            # Single API call for entire chain
            payload = {
                "chunks": chunks,
                "dataset_source": f"ontology-{source}",
                "batch_id": chain_data.get("chain_id"),
                "skip_cpesh": True  # Disable CPESH for speed
            }

            response = requests.post(
                f"{self.api_url}/ingest",
                json=payload,
                timeout=60  # Longer timeout for batch
            )

            if response.status_code == 200:
                result = response.json()
                results.append(result)

                # Track ingested concepts (batch result)
                ingested = result.get("successful", 0)
                self.metrics["concepts_ingested"] += ingested

                # Track component counts (from batch result)
                if result.get("results"):
                    for item in result["results"]:
                        if item.get("cpesh_extracted"):
                            self.metrics["cpesh_count"] += 1
                        if item.get("tmd_extracted"):
                            self.metrics["tmd_count"] += 1
                        if item.get("vector_created"):
                            self.metrics["vector_count"] += 1
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                self.metrics["errors"].append(error_msg)

            latency = time.time() - start
            self.metrics["latencies"].append(latency)
            self.metrics["chains_processed"] += 1

            return {"success": True, "results": results, "latency": latency}

        except Exception as e:
            self.metrics["errors"].append(str(e))
            return {"success": False, "error": str(e)}

    def run_benchmark(self, input_file: Path) -> Dict:
        """Run full benchmark on input file"""
        print(f"📊 Starting ingestion benchmark: {input_file}")
        print(f"API: {self.api_url}")
        print()

        # Check API health
        try:
            health = requests.get(f"{self.api_url}/health", timeout=5)
            if health.status_code != 200:
                print(f"❌ API not healthy: {health.status_code}")
                return self.metrics
        except Exception as e:
            print(f"❌ Cannot reach API: {e}")
            return self.metrics

        self.metrics["start_time"] = time.time()

        # Process chains
        with open(input_file) as f:
            for line_num, line in enumerate(f, 1):
                chain = json.loads(line.strip())
                result = self.ingest_chain(chain)

                if line_num % 10 == 0:
                    print(f"  Processed {line_num} chains, {self.metrics['concepts_ingested']} concepts...")

        self.metrics["end_time"] = time.time()
        self.metrics["duration_seconds"] = self.metrics["end_time"] - self.metrics["start_time"]

        # Calculate average memory
        if self.memory_samples:
            self.metrics["avg_memory_mb"] = sum(self.memory_samples) / len(self.memory_samples)

        # Print summary
        self.print_summary()

        return self.metrics

    def print_summary(self):
        """Print benchmark summary"""
        print()
        print("=" * 60)
        print("📈 BENCHMARK RESULTS")
        print("=" * 60)
        print(f"⏱️  Duration: {self.metrics['duration_seconds']:.2f}s")
        print(f"📦 Chains processed: {self.metrics['chains_processed']}")
        print(f"🔤 Concepts ingested: {self.metrics['concepts_ingested']}")
        print()
        print(f"💾 Peak memory: {self.metrics['peak_memory_mb']:.1f} MB")
        print(f"💾 Avg memory: {self.metrics['avg_memory_mb']:.1f} MB")
        print()
        print(f"✅ CPESH extracted: {self.metrics['cpesh_count']} ({self.metrics['cpesh_count']/max(self.metrics['concepts_ingested'],1)*100:.1f}%)")
        print(f"✅ TMD extracted: {self.metrics['tmd_count']} ({self.metrics['tmd_count']/max(self.metrics['concepts_ingested'],1)*100:.1f}%)")
        print(f"✅ Vectors created: {self.metrics['vector_count']} ({self.metrics['vector_count']/max(self.metrics['concepts_ingested'],1)*100:.1f}%)")
        print()

        if self.metrics['latencies']:
            avg_latency = sum(self.metrics['latencies']) / len(self.metrics['latencies'])
            print(f"⚡ Avg latency per chain: {avg_latency:.3f}s")
            print(f"⚡ Throughput: {self.metrics['concepts_ingested']/self.metrics['duration_seconds']:.1f} concepts/sec")

        if self.metrics['errors']:
            print()
            print(f"❌ Errors ({len(self.metrics['errors'])}):")
            for err in self.metrics['errors'][:5]:  # Show first 5
                print(f"   - {err}")

        print("=" * 60)

    def save_metrics(self, output_file: Path):
        """Save metrics to JSON"""
        with open(output_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"💾 Metrics saved to: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_ingestion.py <input_file.jsonl> [api_url]")
        print("Example: python benchmark_ingestion.py test_data/swo_10_samples.jsonl")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8004"

    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        sys.exit(1)

    # Run benchmark
    benchmark = IngestionBenchmark(api_url)
    metrics = benchmark.run_benchmark(input_file)

    # Save metrics
    output_file = input_file.parent / f"{input_file.stem}_metrics.json"
    benchmark.save_metrics(output_file)


if __name__ == "__main__":
    main()
