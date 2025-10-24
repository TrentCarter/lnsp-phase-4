#!/usr/bin/env python3
"""
Performance Comparison Table Generator for LVM Approaches

Creates detailed comparison tables showing deltas between:
- Direct Prediction (baseline)
- Tiny Recursion (TR)
- TwoTower (TT)

Includes statistical analysis, inference times, and recommendations.
"""

import json
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict
import time

def load_comprehensive_results():
    """Load the latest comprehensive evaluation results."""
    results_dir = Path("artifacts/lvm/comprehensive_evaluation")
    results_files = list(results_dir.glob("comprehensive_results_*.json"))
    results_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    if not results_files:
        print("❌ No comprehensive results found!")
        return None

    latest = results_files[0]
    print(f"📊 Loading: {latest.name}")

    with open(latest, 'r') as f:
        return json.load(f)

def calculate_performance_deltas(data):
    """Calculate performance deltas between approaches."""

    results = {
        'overall_comparison': {},
        'model_comparison': {},
        'context_comparison': {},
        'inference_time_comparison': {},
        'statistical_analysis': {}
    }

    # Overall performance comparison
    overall_scores = defaultdict(list)

    for model_name, model_data in data['models'].items():
        for dataset_name, dataset_data in model_data.items():
            if not dataset_data:
                continue

            for context_size, context_data in dataset_data.items():
                if not context_data:
                    continue

                for approach, approach_data in context_data.items():
                    if not approach_data:
                        continue

                    # Calculate average performance for this configuration
                    cosines = [sample['evaluation']['cosine_similarity'] for sample in approach_data]
                    times = [sample['prediction'].get('inference_time', 0) for sample in approach_data]

                    avg_cosine = np.mean(cosines)
                    avg_time = np.mean(times) if times else 0

                    key = f"{model_name}_{dataset_name}_{context_size}"
                    overall_scores[approach].append({
                        'key': key,
                        'avg_cosine': avg_cosine,
                        'avg_time': avg_time,
                        'num_samples': len(cosines)
                    })

    # Calculate overall averages
    for approach, scores in overall_scores.items():
        cosines = [s['avg_cosine'] for s in scores]
        times = [s['avg_time'] for s in scores]

        results['overall_comparison'][approach] = {
            'avg_cosine': np.mean(cosines),
            'std_cosine': np.std(cosines),
            'avg_time': np.mean(times),
            'total_measurements': len(cosines),
            'measurements': scores
        }

    # Calculate deltas from direct prediction baseline
    if 'direct_prediction' in results['overall_comparison']:
        baseline = results['overall_comparison']['direct_prediction']

        for approach in ['tiny_recursion', 'twotower']:
            if approach in results['overall_comparison']:
                enhanced = results['overall_comparison'][approach]

                cosine_delta = enhanced['avg_cosine'] - baseline['avg_cosine']
                time_delta = enhanced['avg_time'] - baseline['avg_time']
                relative_cosine = cosine_delta / baseline['avg_cosine'] if baseline['avg_cosine'] > 0 else 0

                results['overall_comparison'][approach].update({
                    'cosine_delta': cosine_delta,
                    'time_delta': time_delta,
                    'relative_cosine_delta': relative_cosine,
                    'cosine_delta_pct': relative_cosine * 100
                })

    # Model-by-model comparison
    for model_name, model_data in data['models'].items():
        model_scores = defaultdict(list)

        for dataset_name, dataset_data in model_data.items():
            for context_size, context_data in dataset_data.items():
                for approach, approach_data in context_data.items():
                    cosines = [sample['evaluation']['cosine_similarity'] for sample in approach_data]
                    times = [sample['prediction'].get('inference_time', 0) for sample in approach_data]

                    avg_cosine = np.mean(cosines)
                    avg_time = np.mean(times)

                    model_scores[approach].append({
                        'dataset': dataset_name,
                        'context': context_size,
                        'avg_cosine': avg_cosine,
                        'avg_time': avg_time
                    })

        # Calculate model averages and deltas
        results['model_comparison'][model_name] = {}
        for approach, scores in model_scores.items():
            cosines = [s['avg_cosine'] for s in scores]
            times = [s['avg_time'] for s in scores]

            results['model_comparison'][model_name][approach] = {
                'avg_cosine': np.mean(cosines),
                'std_cosine': np.std(cosines),
                'avg_time': np.mean(times),
                'measurements': len(cosines)
            }

        # Add deltas for this model
        if 'direct_prediction' in results['model_comparison'][model_name]:
            baseline = results['model_comparison'][model_name]['direct_prediction']

            for approach in ['tiny_recursion', 'twotower']:
                if approach in results['model_comparison'][model_name]:
                    enhanced = results['model_comparison'][model_name][approach]

                    cosine_delta = enhanced['avg_cosine'] - baseline['avg_cosine']
                    relative_cosine = cosine_delta / baseline['avg_cosine'] if baseline['avg_cosine'] > 0 else 0

                    results['model_comparison'][model_name][approach].update({
                        'cosine_delta': cosine_delta,
                        'relative_cosine_delta': relative_cosine,
                        'cosine_delta_pct': relative_cosine * 100
                    })

    return results

def generate_performance_table(results):
    """Generate formatted performance comparison table."""

    table = []
    table.append("🏆 LVM APPROACH PERFORMANCE COMPARISON")
    table.append("=" * 80)
    table.append("")

    # Overall Performance Summary
    table.append("📊 OVERALL PERFORMANCE SUMMARY")
    table.append("-" * 80)
    table.append(f"{'Approach'"<20"} {'Cosine Score'"<15"} {'±Std'"<10"} {'Time (ms)'"<12"} {'Rating'"<10"}")
    table.append("-" * 80)

    baseline_cosine = results['overall_comparison']['direct_prediction']['avg_cosine']
    baseline_time = results['overall_comparison']['direct_prediction']['avg_time']

    approaches = ['direct_prediction', 'tiny_recursion', 'twotower']
    for approach in approaches:
        if approach in results['overall_comparison']:
            data = results['overall_comparison'][approach]
            cosine = data['avg_cosine']
            std = data['std_cosine']
            time_ms = data['avg_time'] * 1000

            # Calculate rating
            if approach == 'direct_prediction':
                rating = "⚡ BASELINE"
            else:
                delta_pct = data.get('cosine_delta_pct', 0)
                if abs(delta_pct) < 1:  # Within 1%
                    rating = "🟡 EQUIVALENT"
                elif delta_pct > 0:
                    rating = "🟢 BETTER"
                else:
                    rating = "🔴 WORSE"

            row = "{:<20} {:<15.4f} {:<10.4f} {:<12.2f} {:<10}".format(
                approach.upper(), cosine, std, time_ms, rating)
            table.append(row)

    table.append("")

    # Performance Deltas
    table.append("📈 PERFORMANCE DELTAS FROM BASELINE")
    table.append("-" * 80)
    table.append(f"{'Approach'<20} {'Cosine Δ'<15} {'Cosine %'<12} {'Time Δ (ms)'<15} {'Trade-off'<15}")
    table.append("-" * 80)

    for approach in ['tiny_recursion', 'twotower']:
        if approach in results['overall_comparison']:
            data = results['overall_comparison'][approach]
            cosine_delta = data.get('cosine_delta', 0)
            cosine_delta_pct = data.get('cosine_delta_pct', 0)
            time_delta_ms = data.get('time_delta', 0) * 1000

            # Trade-off assessment
            if approach == 'tiny_recursion':
                if abs(cosine_delta_pct) < 2 and time_delta_ms < 0:
                    trade_off = "🟢 WIN-WIN"
                elif abs(cosine_delta_pct) < 1:
                    trade_off = "🟡 NEUTRAL"
                else:
                    trade_off = "🟠 TRADE-OFF"
            else:  # twotower
                if cosine_delta_pct > -2 and time_delta_ms < 10:
                    trade_off = "🟡 ACCEPTABLE"
                else:
                    trade_off = "🔴 COSTLY"

            row = "{:<20} {:<15.4f} {:<12.2f} {:<15.2f} {:<15}".format(
                approach.upper(), cosine_delta, cosine_delta_pct, time_delta_ms, trade_off)
            table.append(row)

    table.append("")

    # Model-by-Model Breakdown
    table.append("🔍 MODEL-BY-MODEL PERFORMANCE")
    table.append("-" * 80)

    for model_name, model_data in results['model_comparison'].items():
        table.append(f"\n{model_name.upper()}:")
        table.append(f"{'Approach'"<20"} {'Cosine Score'"<15"} {'Δ from Direct'"<15"} {'Time (ms)'"<12"} {'Best?'"<10"}")
        table.append("-" * 60)

        baseline_cosine = model_data['direct_prediction']['avg_cosine']
        baseline_time = model_data['direct_prediction']['avg_time']

        for approach in approaches:
            if approach in model_data:
                data = model_data[approach]
                cosine = data['avg_cosine']
                time_ms = data['avg_time'] * 1000
                delta = data.get('cosine_delta', 0)

                # Mark best approach for this model
                is_best = ""
                if approach != 'direct_prediction':
                    if data['avg_cosine'] > baseline_cosine:
                        is_best = "🟢"
                    elif abs(data['avg_cosine'] - baseline_cosine) < 0.01:
                        is_best = "🟡"
                    else:
                        is_best = "🔴"

                table.append(f"{approach.upper()"<20"} {cosine"<15.4f"} {delta"<15.4f"} {time_ms"<12.2f"} {is_best"<10"}")

    table.append("")

    # Recommendations
    table.append("🎯 RECOMMENDATIONS")
    table.append("-" * 80)

    # Tiny Recursion recommendation
    tr_data = results['overall_comparison'].get('tiny_recursion', {})
    if tr_data:
        delta_pct = tr_data.get('cosine_delta_pct', 0)
        time_delta = tr_data.get('time_delta', 0)

        if abs(delta_pct) < 2 and time_delta < 0:
            tr_rec = "🟢 HIGHLY RECOMMENDED - Better performance, faster inference"
        elif abs(delta_pct) < 1:
            tr_rec = "🟡 RECOMMENDED - Equivalent performance, potential benefits"
        else:
            tr_rec = "🟠 CONSIDER - Performance trade-off exists"

        table.append(f"Tiny Recursion: {tr_rec}")

    # TwoTower recommendation
    tt_data = results['overall_comparison'].get('twotower', {})
    if tt_data:
        delta_pct = tt_data.get('cosine_delta_pct', 0)
        time_delta = tt_data.get('time_delta', 0)

        if delta_pct > -3 and time_delta < 0.01:  # Within 3% and similar time
            tt_rec = "🟡 VIABLE OPTION - For retrieval-enhanced scenarios"
        else:
            tt_rec = "🔴 NOT RECOMMENDED - Performance penalty too high"

        table.append(f"TwoTower: {tt_rec}")

    return "\n".join(table)

def generate_detailed_analysis(results):
    """Generate detailed statistical analysis."""

    analysis = []

    # Statistical significance analysis
    analysis.append("📊 STATISTICAL ANALYSIS")
    analysis.append("=" * 80)
    analysis.append("")

    # Compare approaches statistically
    approaches = ['direct_prediction', 'tiny_recursion', 'twotower']

    for model_name, model_data in results['model_comparison'].items():
        analysis.append(f"Model: {model_name.upper()}")

        # Collect scores for each approach
        scores_by_approach = {}
        for approach in approaches:
            if approach in model_data:
                scores_by_approach[approach] = [
                    m['avg_cosine'] for m in results['overall_comparison'][approach]['measurements']
                    if f"{model_name}_" in m['key']
                ]

        # Perform pairwise comparisons
        for i, approach1 in enumerate(approaches[:-1]):
            for approach2 in approaches[i+1:]:
                if approach1 in scores_by_approach and approach2 in scores_by_approach:
                    scores1 = scores_by_approach[approach1]
                    scores2 = scores_by_approach[approach2]

                    if len(scores1) > 1 and len(scores2) > 1:
                        # Simple t-test-like comparison
                        mean1, mean2 = np.mean(scores1), np.mean(scores2)
                        std1, std2 = np.std(scores1), np.std(scores2)
                        n1, n2 = len(scores1), len(scores2)

                        # Calculate standard error
                        se1, se2 = std1/np.sqrt(n1), std2/np.sqrt(n2)
                        pooled_se = np.sqrt(se1**2 + se2**2)

                        if pooled_se > 0:
                            t_stat = abs(mean1 - mean2) / pooled_se
                            significant = "🟢" if t_stat > 2.0 else "🟡"

                            analysis.append(f"  {approach1.upper()} vs {approach2.upper()}: {significant} Δ={mean1-mean2:.4f} (t={t_stat:.2f})")

        analysis.append("")

    return "\n".join(analysis)

def main():
    """Main analysis function."""

    print("🔍 Analyzing LVM Approach Performance Deltas...")
    print("=" * 80)

    # Load data
    data = load_comprehensive_results()
    if not data:
        return

    # Calculate deltas
    results = calculate_performance_deltas(data)

    # Generate performance table
    table = generate_performance_table(results)
    print(table)

    # Generate detailed analysis
    detailed = generate_detailed_analysis(results)
    print(detailed)

    # Save results
    output_file = Path("artifacts/lvm/comprehensive_evaluation") / f"performance_comparison_{int(time.time())}.txt"
    with open(output_file, 'w') as f:
        f.write(table + "\n\n" + detailed)

    print(f"\n💾 Performance analysis saved to: {output_file}")

    # Summary insights
    print("\n" + "=" * 80)
    print("💡 KEY INSIGHTS:")
    print("=" * 80)

    tr_data = results['overall_comparison'].get('tiny_recursion', {})
    tt_data = results['overall_comparison'].get('twotower', {})

    if tr_data:
        delta_pct = tr_data.get('cosine_delta_pct', 0)
        time_delta = tr_data.get('time_delta', 0)
        print(f"• Tiny Recursion: {delta_pct:+.2f}% cosine delta, {time_delta*1000:+.1f}ms time delta")

    if tt_data:
        delta_pct = tt_data.get('cosine_delta_pct', 0)
        time_delta = tt_data.get('time_delta', 0)
        print(f"• TwoTower: {delta_pct:+.2f}% cosine delta, {time_delta*1000:+.1f}ms time delta")

    # Best performing model
    best_model = max(results['model_comparison'].items(),
                    key=lambda x: results['model_comparison'][x[0]]['direct_prediction']['avg_cosine'])
    print(f"• Best Model: {best_model[0].upper()} ({best_model[1]['direct_prediction']['avg_cosine']:.4f} cosine)")

if __name__ == "__main__":
    main()
