#!/usr/bin/env python3
"""
GraphRAG Iteration Tracker - Track GraphRAG improvements over time.

Maintains a log of GraphRAG experiments and their performance metrics.

Usage:
    python tools/graphrag_tracker.py add --name "Fix 10x edge expansion" --p1 0.60 --p5 0.84 --notes "Reduced Neo4j edge expansion"
    python tools/graphrag_tracker.py list
    python tools/graphrag_tracker.py compare
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
TRACKER_FILE = ROOT / "artifacts" / "graphrag_iterations.jsonl"


def ensure_tracker_file():
    """Ensure tracker file exists."""
    TRACKER_FILE.parent.mkdir(exist_ok=True)
    if not TRACKER_FILE.exists():
        TRACKER_FILE.write_text("")


def add_iteration(
    name: str,
    p_at_1: float,
    p_at_5: float,
    p_at_10: float = None,
    mrr: float = None,
    ndcg: float = None,
    latency: float = None,
    notes: str = "",
    git_commit: str = "",
):
    """Add a new GraphRAG iteration to the tracker."""
    ensure_tracker_file()

    iteration = {
        "timestamp": datetime.now().isoformat(),
        "name": name,
        "metrics": {
            "p_at_1": p_at_1,
            "p_at_5": p_at_5,
            "p_at_10": p_at_10,
            "mrr": mrr,
            "ndcg": ndcg,
        },
        "latency_ms": latency,
        "notes": notes,
        "git_commit": git_commit,
    }

    with open(TRACKER_FILE, "a") as f:
        f.write(json.dumps(iteration) + "\n")

    print(f"✅ Added iteration: {name}")
    print(f"   P@1: {p_at_1*100:.1f}%  P@5: {p_at_5*100:.1f}%")


def list_iterations():
    """List all GraphRAG iterations."""
    ensure_tracker_file()

    iterations = []
    with open(TRACKER_FILE) as f:
        for line in f:
            if line.strip():
                iterations.append(json.loads(line))

    if not iterations:
        print("No GraphRAG iterations recorded yet.")
        return

    print("=" * 120)
    print(f"{'GRAPHRAG ITERATION HISTORY':^120}")
    print("=" * 120)
    print()
    print(f"{'#':<4} {'Date':<12} {'Name':<30} {'P@1':>8} {'P@5':>8} {'P@10':>8} {'MRR':>8} {'Latency':>10}")
    print("-" * 120)

    for i, it in enumerate(iterations, 1):
        date = datetime.fromisoformat(it['timestamp']).strftime('%Y-%m-%d')
        name = it['name'][:29]
        metrics = it['metrics']

        p1 = f"{metrics['p_at_1']*100:.1f}%" if metrics.get('p_at_1') else "N/A"
        p5 = f"{metrics['p_at_5']*100:.1f}%" if metrics.get('p_at_5') else "N/A"
        p10 = f"{metrics['p_at_10']*100:.1f}%" if metrics.get('p_at_10') else "N/A"
        mrr = f"{metrics['mrr']:.4f}" if metrics.get('mrr') else "N/A"
        latency = f"{it.get('latency_ms', 0):.1f}ms" if it.get('latency_ms') else "N/A"

        print(f"{i:<4} {date:<12} {name:<30} {p1:>8} {p5:>8} {p10:>8} {mrr:>8} {latency:>10}")

        if it.get('notes'):
            print(f"     Notes: {it['notes']}")

    print("-" * 120)
    print(f"\nTotal iterations: {len(iterations)}")


def compare_iterations():
    """Compare GraphRAG iterations and show improvements."""
    ensure_tracker_file()

    iterations = []
    with open(TRACKER_FILE) as f:
        for line in f:
            if line.strip():
                iterations.append(json.loads(line))

    if len(iterations) < 2:
        print("Need at least 2 iterations to compare.")
        return

    print("=" * 100)
    print(f"{'GRAPHRAG ITERATION COMPARISON':^100}")
    print("=" * 100)
    print()

    # Compare first vs last
    first = iterations[0]
    last = iterations[-1]

    print(f"First iteration:  {first['name']} ({datetime.fromisoformat(first['timestamp']).strftime('%Y-%m-%d')})")
    print(f"Latest iteration: {last['name']} ({datetime.fromisoformat(last['timestamp']).strftime('%Y-%m-%d')})")
    print()

    print(f"{'Metric':<10} {'First':>12} {'Latest':>12} {'Δ Absolute':>12} {'Δ Relative':>12} {'Trend':>8}")
    print("-" * 100)

    metrics_to_compare = [
        ('P@1', 'p_at_1'),
        ('P@5', 'p_at_5'),
        ('P@10', 'p_at_10'),
        ('MRR', 'mrr'),
    ]

    for metric_name, metric_key in metrics_to_compare:
        first_val = first['metrics'].get(metric_key)
        last_val = last['metrics'].get(metric_key)

        if first_val is None or last_val is None:
            continue

        abs_delta = last_val - first_val
        rel_delta = (abs_delta / first_val * 100) if first_val > 0 else 0

        if metric_name.startswith('P@'):
            first_str = f"{first_val*100:.1f}%"
            last_str = f"{last_val*100:.1f}%"
            abs_str = f"+{abs_delta*100:.1f}pp" if abs_delta >= 0 else f"{abs_delta*100:.1f}pp"
        else:
            first_str = f"{first_val:.4f}"
            last_str = f"{last_val:.4f}"
            abs_str = f"+{abs_delta:.4f}" if abs_delta >= 0 else f"{abs_delta:.4f}"

        rel_str = f"+{rel_delta:.1f}%" if rel_delta >= 0 else f"{rel_delta:.1f}%"

        if abs_delta > 0:
            trend = "📈 UP"
        elif abs_delta < 0:
            trend = "📉 DOWN"
        else:
            trend = "➡️  SAME"

        print(f"{metric_name:<10} {first_str:>12} {last_str:>12} {abs_str:>12} {rel_str:>12} {trend:>8}")

    print("-" * 100)
    print()

    # Show iteration-by-iteration improvements
    print("ITERATION-BY-ITERATION P@5 PROGRESS")
    print("-" * 100)

    for i, it in enumerate(iterations):
        p5 = it['metrics'].get('p_at_5', 0)
        date = datetime.fromisoformat(it['timestamp']).strftime('%Y-%m-%d')

        # Calculate delta from previous
        if i > 0:
            prev_p5 = iterations[i-1]['metrics'].get('p_at_5', 0)
            delta = p5 - prev_p5
            delta_str = f"(+{delta*100:.1f}pp)" if delta >= 0 else f"({delta*100:.1f}pp)"
        else:
            delta_str = "(baseline)"

        print(f"{i+1}. [{date}] {it['name']:<40} P@5: {p5*100:.1f}% {delta_str}")

    print("-" * 100)


def main():
    parser = argparse.ArgumentParser(description="GraphRAG Iteration Tracker")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new iteration')
    add_parser.add_argument('--name', required=True, help='Iteration name')
    add_parser.add_argument('--p1', type=float, required=True, help='P@1 score (0.0-1.0)')
    add_parser.add_argument('--p5', type=float, required=True, help='P@5 score (0.0-1.0)')
    add_parser.add_argument('--p10', type=float, help='P@10 score (0.0-1.0)')
    add_parser.add_argument('--mrr', type=float, help='MRR score')
    add_parser.add_argument('--ndcg', type=float, help='nDCG score')
    add_parser.add_argument('--latency', type=float, help='Average latency in ms')
    add_parser.add_argument('--notes', default='', help='Additional notes')
    add_parser.add_argument('--commit', default='', help='Git commit hash')

    # List command
    subparsers.add_parser('list', help='List all iterations')

    # Compare command
    subparsers.add_parser('compare', help='Compare iterations')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'add':
        add_iteration(
            name=args.name,
            p_at_1=args.p1,
            p_at_5=args.p5,
            p_at_10=args.p10,
            mrr=args.mrr,
            ndcg=args.ndcg,
            latency=args.latency,
            notes=args.notes,
            git_commit=args.commit,
        )
    elif args.command == 'list':
        list_iterations()
    elif args.command == 'compare':
        compare_iterations()


if __name__ == "__main__":
    main()
