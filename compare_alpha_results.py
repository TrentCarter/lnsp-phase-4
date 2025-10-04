#!/usr/bin/env python3
"""Compare TMD alpha parameter tuning results."""
import json
import sys
from pathlib import Path

results_dir = Path("RAG/results")

# Alpha values tested
alphas = [0.2, 0.3, 0.4, 0.5, 0.6]

print("=" * 80)
print("TMD ALPHA PARAMETER TUNING RESULTS")
print("=" * 80)
print()
print("Alpha = TMD weight (1-alpha = vector weight)")
print()
print(f"{'Alpha':<8} {'P@1':<8} {'P@5':<8} {'P@10':<8} {'MRR':<8} {'nDCG':<8}")
print("-" * 80)

best_p1 = (0, 0.0)
best_p5 = (0, 0.0)
best_mrr = (0, 0.0)

for alpha in alphas:
    result_file = results_dir / f"tmd_alpha_{alpha}_oct4.jsonl"

    if not result_file.exists():
        print(f"{alpha:<8} Results file not found: {result_file}")
        continue

    with open(result_file) as f:
        for line in f:
            data = json.loads(line)
            if 'summary' in data:
                metrics = data['metrics']
                p1 = metrics['p_at_1']
                p5 = metrics['p_at_5']
                p10 = metrics['p_at_10']
                mrr = metrics['mrr']
                ndcg = metrics.get('ndcg', 0.0)

                print(f"{alpha:<8.1f} {p1:<8.4f} {p5:<8.4f} {p10:<8.4f} {mrr:<8.4f} {ndcg:<8.4f}")

                if p1 > best_p1[1]:
                    best_p1 = (alpha, p1)
                if p5 > best_p5[1]:
                    best_p5 = (alpha, p5)
                if mrr > best_mrr[1]:
                    best_mrr = (alpha, mrr)

print("-" * 80)
print()
print("BEST RESULTS:")
print(f"  Best P@1:  alpha={best_p1[0]:.1f} ({best_p1[1]:.4f})")
print(f"  Best P@5:  alpha={best_p5[0]:.1f} ({best_p5[1]:.4f})")
print(f"  Best MRR:  alpha={best_mrr[0]:.1f} ({best_mrr[1]:.4f})")
print()
print("=" * 80)
