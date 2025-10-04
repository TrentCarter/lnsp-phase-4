#!/usr/bin/env python3
"""Check TMD benchmark results"""
import json
import sys

results_file = "RAG/results/llm_tmd_oct4.jsonl"

try:
    with open(results_file) as f:
        for line in f:
            data = json.loads(line)
            if 'summary' in data:
                name = data['name']
                metrics = data['metrics']
                print(f"{name}:")
                print(f"  P@1:  {metrics['p_at_1']:.4f}")
                print(f"  P@5:  {metrics['p_at_5']:.4f}")
                print(f"  P@10: {metrics['p_at_10']:.4f}")
                print(f"  MRR:  {metrics['mrr']:.4f}")
                print()
except FileNotFoundError:
    print(f"Results file not found: {results_file}")
    print("Run the benchmark first with: bash run_tmd_benchmark.sh")
    sys.exit(1)
