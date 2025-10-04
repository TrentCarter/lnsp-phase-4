#!/usr/bin/env python3
"""Compare baseline vec vs TMD re-ranking."""
import json
from pathlib import Path

def compute_metrics_from_file(filepath, backend_filter=None):
    """Compute metrics from results file."""
    results = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if backend_filter and data.get('backend') != backend_filter:
                    continue
                results.append(data)

    if not results:
        return None

    total = len(results)
    def get_rank(r):
        rank = r.get('gold_rank')
        return 999 if rank is None else rank

    p_at_1 = sum(1 for r in results if get_rank(r) == 1) / total
    p_at_5 = sum(1 for r in results if get_rank(r) <= 5) / total
    p_at_10 = sum(1 for r in results if get_rank(r) <= 10) / total

    mrr = sum(1.0/get_rank(r) for r in results if get_rank(r) < 999) / total

    return {
        "p_at_1": p_at_1,
        "p_at_5": p_at_5,
        "p_at_10": p_at_10,
        "mrr": mrr,
        "count": total,
    }

# Compute baseline (vec from comprehensive_200.jsonl)
baseline = compute_metrics_from_file(
    "RAG/results/comprehensive_200.jsonl",
    backend_filter="vec"
)

# Compute TMD from earlier test (tmd_200_oct4.jsonl)
tmd_old = compute_metrics_from_file("RAG/results/tmd_200_oct4.jsonl")

# Compute TMD from alpha tuning (alpha=0.3)
tmd_new = compute_metrics_from_file("RAG/results/tmd_alpha_0.3_oct4.jsonl")

print("=" * 80)
print("BASELINE vs TMD RE-RANKING COMPARISON")
print("=" * 80)
print()

if baseline:
    print(f"Baseline vecRAG (comprehensive_200.jsonl, backend=vec):")
    print(f"  P@1:  {baseline['p_at_1']*100:.1f}%  P@5:  {baseline['p_at_5']*100:.1f}%  " +
          f"P@10: {baseline['p_at_10']*100:.1f}%  MRR: {baseline['mrr']:.4f}  ({baseline['count']} queries)")
print()

if tmd_old:
    print(f"TMD re-rank (tmd_200_oct4.jsonl - earlier test):")
    print(f"  P@1:  {tmd_old['p_at_1']*100:.1f}%  P@5:  {tmd_old['p_at_5']*100:.1f}%  " +
          f"P@10: {tmd_old['p_at_10']*100:.1f}%  MRR: {tmd_old['mrr']:.4f}  ({tmd_old['count']} queries)")
print()

if tmd_new:
    print(f"TMD re-rank (tmd_alpha_0.3_oct4.jsonl - alpha tuning):")
    print(f"  P@1:  {tmd_new['p_at_1']*100:.1f}%  P@5:  {tmd_new['p_at_5']*100:.1f}%  " +
          f"P@10: {tmd_new['p_at_10']*100:.1f}%  MRR: {tmd_new['mrr']:.4f}  ({tmd_new['count']} queries)")
print()

# Compare all three
if baseline and tmd_new:
    print("FINDINGS:")
    print(f"1. Baseline vs TMD alpha tuning:")
    delta_p1 = (tmd_new['p_at_1'] - baseline['p_at_1']) * 100
    delta_p5 = (tmd_new['p_at_5'] - baseline['p_at_5']) * 100
    print(f"   P@1: {delta_p1:+.1f}pp  P@5: {delta_p5:+.1f}pp")

if baseline and tmd_old:
    print(f"2. Baseline vs TMD earlier test:")
    delta_p1 = (tmd_old['p_at_1'] - baseline['p_at_1']) * 100
    delta_p5 = (tmd_old['p_at_5'] - baseline['p_at_5']) * 100
    print(f"   P@1: {delta_p1:+.1f}pp  P@5: {delta_p5:+.1f}pp")

if tmd_old and tmd_new:
    print(f"3. TMD earlier test vs TMD alpha tuning:")
    delta_p1 = (tmd_old['p_at_1'] - tmd_new['p_at_1']) * 100
    delta_p5 = (tmd_old['p_at_5'] - tmd_new['p_at_5']) * 100
    print(f"   P@1: {delta_p1:+.1f}pp  P@5: {delta_p5:+.1f}pp")
    print(f"   → Different tests or same queries?")

print()
print("=" * 80)
