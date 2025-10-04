#!/usr/bin/env python3
"""
RAG Performance Dashboard - Real-time monitoring of RAG backend metrics.

Usage:
    python tools/rag_dashboard.py [--watch] [--compare]

Options:
    --watch     Continuously update metrics (refresh every 5s)
    --compare   Show historical comparison
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "RAG" / "results"


def load_latest_results() -> Dict[str, Dict]:
    """Load the most recent benchmark results for each backend."""
    results = {}

    # Define key result files to track
    result_files = {
        "vecRAG": "comprehensive_200.jsonl",
        "TMD_rerank": "tmd_200_oct4.jsonl",
        "GraphRAG": "graphrag_after_fix.jsonl",
        "BM25": "comprehensive_200.jsonl",
        "Lexical": "comprehensive_200.jsonl",
    }

    for backend, filename in result_files.items():
        filepath = RESULTS_DIR / filename
        if not filepath.exists():
            continue

        try:
            with open(filepath) as f:
                for line in f:
                    data = json.loads(line)
                    if 'summary' in data:
                        # Match backend name
                        if backend == "vecRAG" and data['name'] == 'vec':
                            results['vecRAG'] = data
                        elif backend == "TMD_rerank" and data['name'] == 'vec_tmd_rerank':
                            results['TMD_rerank'] = data
                        elif backend == "GraphRAG" and 'graphrag' in data['name']:
                            results['GraphRAG'] = data
                        elif backend == "BM25" and data['name'] == 'bm25':
                            results['BM25'] = data
                        elif backend == "Lexical" and data['name'] == 'lex':
                            results['Lexical'] = data
        except Exception as e:
            print(f"Warning: Failed to load {filename}: {e}", file=sys.stderr)

    return results


def load_alpha_tuning_results() -> List[Tuple[float, Dict]]:
    """Load alpha tuning results if available."""
    results = []
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6]

    for alpha in alphas:
        filepath = RESULTS_DIR / f"tmd_alpha_{alpha}_oct4.jsonl"
        if not filepath.exists():
            continue

        try:
            with open(filepath) as f:
                for line in f:
                    data = json.loads(line)
                    if 'summary' in data:
                        results.append((alpha, data))
                        break
        except Exception:
            continue

    return results


def format_metric(value: float, is_percentage: bool = True) -> str:
    """Format metric with appropriate precision."""
    if is_percentage:
        return f"{value*100:5.1f}%"
    else:
        return f"{value:.4f}"


def print_dashboard_header():
    """Print dashboard header with timestamp."""
    print("=" * 100)
    print(f"{'RAG PERFORMANCE DASHBOARD':^100}")
    print(f"{'Last Updated: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^100}")
    print("=" * 100)
    print()


def print_main_metrics(results: Dict[str, Dict]):
    """Print main benchmark comparison table."""
    print("MAIN BENCHMARKS (200 queries)")
    print("-" * 100)
    print(f"{'Backend':<20} {'P@1':>8} {'P@5':>8} {'P@10':>8} {'MRR':>8} {'nDCG':>8} {'Latency':>12} {'Status':>8}")
    print("-" * 100)

    # Define display order
    backends = ['vecRAG', 'TMD_rerank', 'BM25', 'Lexical', 'GraphRAG']

    for backend in backends:
        if backend not in results:
            print(f"{backend:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>12} {'MISSING':>8}")
            continue

        data = results[backend]
        metrics = data.get('metrics', {})

        p1 = format_metric(metrics.get('p_at_1', 0))
        p5 = format_metric(metrics.get('p_at_5', 0))
        p10 = format_metric(metrics.get('p_at_10', 0))
        mrr = f"{metrics.get('mrr', 0):.4f}"
        ndcg = f"{metrics.get('ndcg', 0):.4f}"

        # Calculate latency
        latencies = data.get('latencies', [])
        avg_latency = f"{sum(latencies)/len(latencies):.2f}ms" if latencies else "N/A"

        # Status indicator
        if backend == 'GraphRAG' and metrics.get('p_at_1', 0) < 0.2:
            status = "🔴 BROKEN"
        elif backend == 'TMD_rerank':
            status = "✅ BEST"
        else:
            status = "✅ OK"

        print(f"{backend:<20} {p1:>8} {p5:>8} {p10:>8} {mrr:>8} {ndcg:>8} {avg_latency:>12} {status:>8}")

    print("-" * 100)
    print()


def print_alpha_tuning_status(alpha_results: List[Tuple[float, Dict]]):
    """Print alpha tuning progress and results."""
    print("TMD ALPHA PARAMETER TUNING")
    print("-" * 100)

    if not alpha_results:
        print("⏳ No alpha tuning results yet. Run: bash tune_alpha.sh")
        print()
        return

    print(f"{'Alpha':<8} {'TMD%':>6} {'Vec%':>6} {'P@1':>8} {'P@5':>8} {'P@10':>8} {'MRR':>8} {'Status':>10}")
    print("-" * 100)

    best_p5 = max(alpha_results, key=lambda x: x[1]['metrics'].get('p_at_5', 0))

    for alpha, data in sorted(alpha_results):
        metrics = data.get('metrics', {})

        p1 = format_metric(metrics.get('p_at_1', 0))
        p5 = format_metric(metrics.get('p_at_5', 0))
        p10 = format_metric(metrics.get('p_at_10', 0))
        mrr = f"{metrics.get('mrr', 0):.4f}"

        tmd_pct = f"{alpha*100:.0f}%"
        vec_pct = f"{(1-alpha)*100:.0f}%"

        status = "🏆 OPTIMAL" if alpha == best_p5[0] else "✅"

        print(f"{alpha:<8.1f} {tmd_pct:>6} {vec_pct:>6} {p1:>8} {p5:>8} {p10:>8} {mrr:>8} {status:>10}")

    print("-" * 100)
    print(f"Best configuration: alpha={best_p5[0]:.1f} (P@5={best_p5[1]['metrics']['p_at_5']*100:.1f}%)")
    print()


def print_improvement_summary(results: Dict[str, Dict]):
    """Print improvement summary comparing TMD rerank to baseline."""
    if 'vecRAG' not in results or 'TMD_rerank' not in results:
        print("⚠️  Cannot compute improvements - missing baseline or TMD results")
        print()
        return

    print("TMD RE-RANKING IMPROVEMENTS")
    print("-" * 100)

    baseline = results['vecRAG']['metrics']
    tmd = results['TMD_rerank']['metrics']

    metrics_to_compare = [
        ('P@1', 'p_at_1'),
        ('P@5', 'p_at_5'),
        ('P@10', 'p_at_10'),
        ('MRR', 'mrr'),
        ('nDCG', 'ndcg'),
    ]

    print(f"{'Metric':<10} {'Baseline':>12} {'TMD Rerank':>12} {'Δ Absolute':>12} {'Δ Relative':>12}")
    print("-" * 100)

    for metric_name, metric_key in metrics_to_compare:
        base_val = baseline.get(metric_key, 0)
        tmd_val = tmd.get(metric_key, 0)
        abs_delta = tmd_val - base_val
        rel_delta = (abs_delta / base_val * 100) if base_val > 0 else 0

        if metric_name.startswith('P@'):
            base_str = f"{base_val*100:.1f}%"
            tmd_str = f"{tmd_val*100:.1f}%"
            abs_str = f"+{abs_delta*100:.1f}pp" if abs_delta > 0 else f"{abs_delta*100:.1f}pp"
        else:
            base_str = f"{base_val:.4f}"
            tmd_str = f"{tmd_val:.4f}"
            abs_str = f"+{abs_delta:.4f}" if abs_delta > 0 else f"{abs_delta:.4f}"

        rel_str = f"+{rel_delta:.1f}%" if rel_delta > 0 else f"{rel_delta:.1f}%"

        print(f"{metric_name:<10} {base_str:>12} {tmd_str:>12} {abs_str:>12} {rel_str:>12}")

    print("-" * 100)
    print()


def print_recommendations(results: Dict[str, Dict], alpha_results: List[Tuple[float, Dict]]):
    """Print actionable recommendations based on current metrics."""
    print("RECOMMENDATIONS")
    print("-" * 100)

    recommendations = []

    # Check if GraphRAG is broken
    if 'GraphRAG' in results:
        graphrag_p1 = results['GraphRAG']['metrics'].get('p_at_1', 0)
        if graphrag_p1 < 0.2:
            recommendations.append("🔴 CRITICAL: GraphRAG performance degraded (P@1 < 20%). Review Neo4j edge expansion.")

    # Check if alpha tuning is needed
    if not alpha_results:
        recommendations.append("⚡ Run alpha parameter tuning: bash tune_alpha.sh (~25 min)")
    elif len(alpha_results) < 5:
        recommendations.append(f"⏳ Alpha tuning in progress ({len(alpha_results)}/5 complete)")
    else:
        best_alpha = max(alpha_results, key=lambda x: x[1]['metrics'].get('p_at_5', 0))
        if best_alpha[0] != 0.3:
            recommendations.append(f"✨ Update default alpha to {best_alpha[0]:.1f} for optimal P@5")

    # Check if TMD rerank shows significant improvement
    if 'vecRAG' in results and 'TMD_rerank' in results:
        base_p5 = results['vecRAG']['metrics'].get('p_at_5', 0)
        tmd_p5 = results['TMD_rerank']['metrics'].get('p_at_5', 0)
        improvement = (tmd_p5 - base_p5) * 100

        if improvement > 5.0:
            recommendations.append(f"🚀 TMD improvement significant ({improvement:.1f}pp). Consider corpus re-ingestion with LLM-based TMD.")

    if not recommendations:
        recommendations.append("✅ All systems performing well. No immediate actions needed.")

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    print("-" * 100)
    print()


def print_quick_commands():
    """Print quick reference commands."""
    print("QUICK COMMANDS")
    print("-" * 100)
    print("  make rag-status              # This dashboard")
    print("  bash tune_alpha.sh           # Run alpha parameter tuning")
    print("  make lnsp-status             # Check LNSP API health")
    print("  make graph-smoke             # Test GraphRAG endpoints")
    print("  make slo-snapshot            # Save current SLO metrics")
    print("-" * 100)
    print()


def main():
    parser = argparse.ArgumentParser(description="RAG Performance Dashboard")
    parser.add_argument("--watch", action="store_true", help="Continuously update metrics")
    parser.add_argument("--compare", action="store_true", help="Show historical comparison")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                os.system('clear' if os.name != 'nt' else 'cls')
                print_dashboard_header()

                results = load_latest_results()
                alpha_results = load_alpha_tuning_results()

                print_main_metrics(results)
                print_alpha_tuning_status(alpha_results)
                print_improvement_summary(results)
                print_recommendations(results, alpha_results)
                print_quick_commands()

                print(f"Refreshing in 5s... (Ctrl+C to exit)")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")
            sys.exit(0)
    else:
        print_dashboard_header()

        results = load_latest_results()
        alpha_results = load_alpha_tuning_results()

        print_main_metrics(results)
        print_alpha_tuning_status(alpha_results)
        print_improvement_summary(results)
        print_recommendations(results, alpha_results)
        print_quick_commands()


if __name__ == "__main__":
    main()
