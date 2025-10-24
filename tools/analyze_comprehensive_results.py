#!/usr/bin/env python3
"""
Extract and analyze comprehensive LVM evaluation results.

Analyzes the 68MB results file to show actual performance of:
- Direct Prediction
- Tiny Recursion (TR)
- TwoTower approach
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import sys
import time

def load_evaluation_results(results_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    print(f"Loading results from {results_path}...")

    with open(results_path, 'r') as f:
        data = json.load(f)

    print(f"✅ Loaded {len(json.dumps(data)) / 1024 / 1024:.1f}MB of results")
    print(f"Models tested: {list(data['models'].keys())}")

    return data

def extract_performance_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract performance metrics for each model/approach/context combination."""

    results = {
        'by_model': {},
        'by_approach': {},
        'by_context_size': {},
        'overall_comparison': {}
    }

    # Process each model
    for model_name, model_data in data['models'].items():
        print(f"\n🔍 Analyzing {model_name.upper()}...")

        model_results = {
            'approaches': {},
            'contexts': {},
            'overall': {'total_samples': 0, 'total_cosine': 0}
        }

        # Process each dataset
        for dataset_name, dataset_data in model_data.items():
            if not dataset_data:  # Skip empty datasets
                continue

            print(f"  Dataset: {dataset_name}")

            # Process each context size
            for context_size, context_data in dataset_data.items():
                if not context_data:  # Skip empty contexts
                    continue

                # Process each approach
                for approach, approach_data in context_data.items():
                    if not approach_data:  # Skip empty approaches
                        continue

                    approach_key = f"{approach}_ctx{context_size}"

                    # Extract metrics from samples
                    cosines = []
                    inference_times = []
                    tr_metadata_list = []

                    for sample in approach_data:
                        eval_data = sample['evaluation']
                        cosines.append(eval_data['cosine_similarity'])
                        inference_times.append(sample['prediction'].get('inference_time', 0))

                        # Collect TR metadata if available
                        if 'tr_metadata' in sample['prediction'] and sample['prediction']['tr_metadata']:
                            tr_metadata_list.append(sample['prediction']['tr_metadata'])

                    # Calculate statistics
                    if cosines:
                        avg_cosine = np.mean(cosines)
                        std_cosine = np.std(cosines)
                        min_cosine = np.min(cosines)
                        max_cosine = np.max(cosines)
                        avg_time = np.mean(inference_times) if inference_times else 0

                        # Store results
                        if approach not in model_results['approaches']:
                            model_results['approaches'][approach] = []
                        if context_size not in model_results['contexts']:
                            model_results['contexts'][context_size] = []

                        model_results['approaches'][approach].append({
                            'context_size': context_size,
                            'dataset': dataset_name,
                            'avg_cosine': avg_cosine,
                            'std_cosine': std_cosine,
                            'min_cosine': min_cosine,
                            'max_cosine': max_cosine,
                            'avg_time': avg_time,
                            'num_samples': len(cosines)
                        })

                        model_results['contexts'][context_size].append({
                            'approach': approach,
                            'dataset': dataset_name,
                            'avg_cosine': avg_cosine,
                            'std_cosine': std_cosine,
                            'min_cosine': min_cosine,
                            'max_cosine': max_cosine,
                            'avg_time': avg_time,
                            'num_samples': len(cosines)
                        })

                        model_results['overall']['total_samples'] += len(cosines)
                        model_results['overall']['total_cosine'] += sum(cosines)

                        print(f"    Context {context_size} | {approach}: {avg_cosine:.4f} ± {std_cosine:.4f} (n={len(cosines)})")

                        # Show TR-specific metrics
                        if approach == 'tiny_recursion' and tr_metadata_list:
                            avg_attempts = np.mean([m.get('attempts', 1) for m in tr_metadata_list])
                            convergence_rate = np.mean([1 if m.get('converged', False) else 0 for m in tr_metadata_list])
                            avg_confidence = np.mean([m.get('confidence', 1.0) for m in tr_metadata_list])
                            print(f"      TR: {convergence_rate:.1%} convergence, {avg_attempts:.1f} attempts, {avg_confidence:.3f} confidence")

        results['by_model'][model_name] = model_results

    return results

def compare_approaches(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare performance across different approaches."""

    comparison = {
        'direct_vs_tr': {},
        'direct_vs_twotower': {},
        'tr_vs_twotower': {},
        'best_overall': {}
    }

    # Compare approaches for each model and context size
    for model_name, model_data in results['by_model'].items():
        for approach1, approach2 in [('direct_prediction', 'tiny_recursion'),
                                   ('direct_prediction', 'twotower'),
                                   ('tiny_recursion', 'twotower')]:
            comp_key = f"{approach1}_vs_{approach2}"

            if approach1 in model_data['approaches'] and approach2 in model_data['approaches']:
                scores1 = [item['avg_cosine'] for item in model_data['approaches'][approach1]]
                scores2 = [item['avg_cosine'] for item in model_data['approaches'][approach2]]

                if scores1 and scores2:
                    avg1 = np.mean(scores1)
                    avg2 = np.mean(scores2)
                    improvement = avg2 - avg1

                    if comp_key not in comparison:
                        comparison[comp_key] = {}
                    comparison[comp_key][model_name] = {
                        'baseline_score': avg1,
                        'enhanced_score': avg2,
                        'improvement': improvement,
                        'relative_improvement': improvement / avg1 if avg1 > 0 else 0
                    }

    # Find best overall approach
    approach_scores = defaultdict(list)
    for model_data in results['by_model'].values():
        for approach, approach_results in model_data['approaches'].items():
            for result in approach_results:
                approach_scores[approach].append(result['avg_cosine'])

    for approach, scores in approach_scores.items():
        comparison['best_overall'][approach] = {
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'num_measurements': len(scores)
        }

    return comparison

def generate_summary_report(results: Dict[str, Any], comparison: Dict[str, Any]) -> str:
    """Generate a comprehensive summary report."""

    report = []
    report.append("🏆 COMPREHENSIVE LVM EVALUATION RESULTS")
    report.append("=" * 60)

    # Overall best approaches
    report.append("\n📈 OVERALL APPROACH PERFORMANCE:")
    for approach, metrics in comparison['best_overall'].items():
        report.append(f"  {approach.upper()}: {metrics['avg_score']:.4f} ± {metrics['std_score']:.4f} (n={metrics['num_measurements']})")

    # Model-by-model breakdown
    report.append("\n🔍 MODEL-BY-MODEL ANALYSIS:")
    for model_name, model_data in results['by_model'].items():
        report.append(f"\n{model_name.upper()}:")

        # Best approach for this model
        best_approach = max(model_data['approaches'].keys(),
                          key=lambda a: np.mean([r['avg_cosine'] for r in model_data['approaches'][a]]))

        best_score = np.mean([r['avg_cosine'] for r in model_data['approaches'][best_approach]])
        report.append(f"  Best approach: {best_approach} ({best_score:.4f})")

        # Context size analysis
        report.append("  Context size performance:")
        for context_size in sorted(model_data['contexts'].keys()):
            context_approaches = model_data['contexts'][context_size]
            best_ctx_approach = max(context_approaches,
                                  key=lambda r: r['avg_cosine'])
            report.append(f"    Context {context_size}: {best_ctx_approach['approach']} ({best_ctx_approach['avg_cosine']:.4f})")

    # Enhancement effectiveness
    report.append("\n🚀 ENHANCEMENT EFFECTIVENESS:")

    for comp_key, model_results in comparison['direct_vs_tr'].items():
        if model_results:
            best_model = max(model_results.items(), key=lambda x: x[1]['improvement'])
            report.append(f"  Tiny Recursion best improvement: {best_model[0]} (+{best_model[1]['improvement']:.4f})")

    for comp_key, model_results in comparison['direct_vs_twotower'].items():
        if model_results:
            best_model = max(model_results.items(), key=lambda x: x[1]['improvement'])
            report.append(f"  TwoTower best improvement: {best_model[0]} (+{best_model[1]['improvement']:.4f})")

    return "\n".join(report)

def main():
    """Main analysis function."""

    # Load the most recent comprehensive results
    results_dir = Path("artifacts/lvm/comprehensive_evaluation")
    results_files = list(results_dir.glob("comprehensive_results_*.json"))
    results_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    if not results_files:
        print("❌ No comprehensive results files found!")
        return

    latest_results = results_files[0]
    print(f"📊 Analyzing latest results: {latest_results.name}")

    # Load and analyze
    data = load_evaluation_results(str(latest_results))
    performance_results = extract_performance_metrics(data)
    comparison = compare_approaches(performance_results)

    # Generate and display report
    report = generate_summary_report(performance_results, comparison)
    print(report)

    # Save detailed analysis
    analysis_file = results_dir / f"detailed_analysis_{int(time.time())}.json"
    analysis_data = {
        'performance_results': performance_results,
        'comparison': comparison,
        'report': report
    }

    with open(analysis_file, 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)

    print(f"\n💾 Detailed analysis saved to: {analysis_file}")

if __name__ == "__main__":
    main()
