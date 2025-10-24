#!/usr/bin/env python3
"""
Multi-Tiny Recursion Ensemble Test

Tests the hypothesis that multiple Tiny Recursion runs with ensembling
can outperform single direct prediction by leveraging TR's speed advantage.

Approaches tested:
1. Direct Prediction (baseline)
2. Single Tiny Recursion
3. Double TR (weighted ensemble)
4. Triple TR (weighted ensemble)
5. Double TR (union of candidates)
6. Triple TR (union of candidates)

Ensembling methods:
- Weighted by confidence
- Simple average
- Union of top-K candidates
- Best-of-N selection
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import time
from typing import Dict, Any, Tuple, List
import json

# Import our Tiny Recursion implementation
sys.path.insert(0, 'tools')
from comprehensive_lvm_evaluation_fixed import TinyRecursion, TwoTowerQueryEncoder

class MultiTinyRecursion:
    """Multi-run Tiny Recursion with various ensembling strategies."""

    def __init__(self, base_model: nn.Module, threshold: float = 0.05, max_attempts: int = 2):
        self.base_model = base_model
        self.threshold = threshold
        self.max_attempts = max_attempts

    def predict_single(self, context_t: torch.Tensor, temp: float = 0.06, seed: int = 42) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Single Tiny Recursion prediction with optional temperature/seed."""
        torch.manual_seed(seed)
        tr_model = TinyRecursion(self.base_model, self.threshold, self.max_attempts)
        return tr_model.refine_prediction(context_t)

    def predict_double_ensemble(self, context_t: torch.Tensor, method: str = 'weighted') -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Double Tiny Recursion with ensembling."""

        # First run
        pred1, meta1 = self.predict_single(context_t, seed=42)

        # Second run with different seed for diversity
        pred2, meta2 = self.predict_single(context_t, seed=1337)

        # Ensemble methods
        if method == 'weighted':
            # Weight by confidence
            conf1 = meta1.get('confidence', 0.8)
            conf2 = meta2.get('confidence', 0.8)
            total_conf = conf1 + conf2
            w1, w2 = conf1 / total_conf, conf2 / total_conf
            final_pred = w1 * pred1 + w2 * pred2

        elif method == 'average':
            # Simple average
            final_pred = (pred1 + pred2) / 2

        elif method == 'best_confidence':
            # Use higher confidence prediction
            if meta1.get('confidence', 0) > meta2.get('confidence', 0):
                final_pred = pred1
            else:
                final_pred = pred2

        elif method == 'converged_preferred':
            # Prefer converged prediction
            if meta1.get('converged', False) and meta2.get('converged', False):
                # Both converged, use confidence weighting
                conf1 = meta1.get('confidence', 0.8)
                conf2 = meta2.get('confidence', 0.8)
                total_conf = conf1 + conf2
                w1, w2 = conf1 / total_conf, conf2 / total_conf
                final_pred = w1 * pred1 + w2 * pred2
            elif meta1.get('converged', False):
                final_pred = pred1
            elif meta2.get('converged', False):
                final_pred = pred2
            else:
                final_pred = pred1

        else:
            final_pred = pred1

        # Combined metadata
        combined_meta = {
            'method': f'double_{method}',
            'predictions': [meta1, meta2],
            'final_confidence': max(meta1.get('confidence', 0), meta2.get('confidence', 0))
        }

        return final_pred, combined_meta

    def predict_triple_ensemble(self, context_t: torch.Tensor, method: str = 'weighted') -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Triple Tiny Recursion with ensembling."""

        # Three runs with different seeds
        pred1, meta1 = self.predict_single(context_t, seed=42)
        pred2, meta2 = self.predict_single(context_t, seed=1337)
        pred3, meta3 = self.predict_single(context_t, seed=4242)

        if method == 'weighted':
            # Weight by confidence
            conf1 = meta1.get('confidence', 0.8)
            conf2 = meta2.get('confidence', 0.8)
            conf3 = meta3.get('confidence', 0.8)
            total_conf = conf1 + conf2 + conf3
            w1, w2, w3 = conf1 / total_conf, conf2 / total_conf, conf3 / total_conf
            final_pred = w1 * pred1 + w2 * pred2 + w3 * pred3

        elif method == 'average':
            final_pred = (pred1 + pred2 + pred3) / 3

        else:
            final_pred = pred1

        combined_meta = {
            'method': f'triple_{method}',
            'predictions': [meta1, meta2, meta3],
            'final_confidence': max(meta1.get('confidence', 0), meta2.get('confidence', 0), meta3.get('confidence', 0))
        }

        return final_pred, combined_meta

def load_test_data(num_samples: int = 50) -> Dict[str, Any]:
    """Load test dataset with more samples for better statistics."""
    test_path = "artifacts/lvm/wikipedia_ood_test_ctx5.npz"
    if Path(test_path).exists():
        test_data = np.load(test_path, allow_pickle=True)
        return {
            'context_sequences': test_data['context_sequences'][:num_samples],
            'target_vectors': test_data['target_vectors'][:num_samples]
        }
    else:
        print("❌ Test data not found!")
        return None

def test_multi_tr_performance():
    """Test multi-Tiny Recursion performance vs direct prediction."""

    print("🔬 MULTI-TINY RECURSION ENSEMBLE TEST")
    print("=" * 60)

    # Load test data
    test_data = load_test_data(num_samples=100)  # Larger sample for statistics
    if not test_data:
        return

    # Load AMN model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model_path = "artifacts/lvm/production_model"
    model_dir = Path(model_path)

    if not model_dir.exists():
        print("❌ Model not found!")
        return

    # Load model
    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / "final_model.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = checkpoint.get('hyperparameters', {})
    model = create_model('amn', **hparams)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("✅ AMN model loaded and ready!")

    # Test approaches
    approaches = {
        'direct_prediction': 'Direct LVM (baseline)',
        'tiny_recursion_single': 'Single Tiny Recursion',
        'tiny_recursion_double_weighted': 'Double TR (confidence weighted)',
        'tiny_recursion_double_average': 'Double TR (simple average)',
        'tiny_recursion_double_best': 'Double TR (best confidence)',
        'tiny_recursion_triple_weighted': 'Triple TR (confidence weighted)',
        'tiny_recursion_triple_average': 'Triple TR (simple average)'
    }

    results = {}

    for approach_name, description in approaches.items():
        print(f"\n🧪 Testing {description}...")
        print(f"   Approach: {approach_name}")

        approach_results = []
        total_time = 0

        for i in range(len(test_data['context_sequences'])):
            context_vecs = test_data['context_sequences'][i]
            target_vec = test_data['target_vectors'][i]

            # Convert to tensor
            context_t = torch.from_numpy(context_vecs).float().unsqueeze(0).to(device)

            start_time = time.time()

            if approach_name == 'direct_prediction':
                # Direct prediction
                with torch.no_grad():
                    pred_t = model(context_t)
                pred_vec = pred_t.cpu().numpy()[0]

            elif approach_name.startswith('tiny_recursion_single'):
                # Single TR
                multi_tr = MultiTinyRecursion(model)
                pred_tensor, tr_meta = multi_tr.predict_single(context_t)
                pred_vec = pred_tensor.cpu().numpy()[0]

            elif approach_name.startswith('tiny_recursion_double'):
                # Double TR
                multi_tr = MultiTinyRecursion(model)
                method = approach_name.split('_')[-1]  # 'weighted', 'average', 'best'
                pred_tensor, tr_meta = multi_tr.predict_double_ensemble(context_t, method)
                pred_vec = pred_tensor.cpu().numpy()[0]

            elif approach_name.startswith('tiny_recursion_triple'):
                # Triple TR
                multi_tr = MultiTinyRecursion(model)
                method = approach_name.split('_')[-1]  # 'weighted', 'average'
                pred_tensor, tr_meta = multi_tr.predict_triple_ensemble(context_t, method)
                pred_vec = pred_tensor.cpu().numpy()[0]

            inference_time = time.time() - start_time
            total_time += inference_time

            # Calculate cosine similarity
            cosine_sim = np.dot(pred_vec, target_vec) / (np.linalg.norm(pred_vec) * np.linalg.norm(target_vec))

            approach_results.append({
                'sample_id': i,
                'cosine_similarity': cosine_sim,
                'inference_time': inference_time,
                'tr_metadata': tr_meta if 'tr_meta' in locals() else None
            })

            if i % 20 == 0:
                print(f"   Sample {i}: {cosine_sim:.4f} ({inference_time:.4f}s)")

        # Calculate statistics
        if approach_results:
            cosines = [r['cosine_similarity'] for r in approach_results]
            times = [r['inference_time'] for r in approach_results]

            results[approach_name] = {
                'avg_cosine': np.mean(cosines),
                'std_cosine': np.std(cosines),
                'avg_time': np.mean(times),
                'max_cosine': np.max(cosines),
                'min_cosine': np.min(cosines),
                'samples': approach_results
            }

            print(f"   📊 Average: {np.mean(cosines):.4f} cosine, {np.mean(times):.4f}s per prediction")

    # Display comprehensive comparison
    print("\n🏆 COMPREHENSIVE RESULTS:")
    header = "{:<30} {:<10} {:<8} {:<10} {:<10}".format('Approach', 'Cosine', '±Std', 'Time (ms)', 'Rating')
    print(header)
    print("-" * 80)

    baseline_cosine = results['direct_prediction']['avg_cosine']
    baseline_time = results['direct_prediction']['avg_time']

    # Sort by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_cosine'], reverse=True)

    for approach_name, metrics in sorted_results:
        cosine = metrics['avg_cosine']
        std = metrics['std_cosine']
        time_ms = metrics['avg_time'] * 1000

        # Rating
        delta_cosine = cosine - baseline_cosine
        delta_time = metrics['avg_time'] - baseline_time

        if approach_name == 'direct_prediction':
            rating = '⚡ BASELINE'
        elif delta_cosine > 0.01:
            rating = '🟢 BETTER'
        elif abs(delta_cosine) < 0.01 and delta_time < 0:
            rating = '🟢 WIN-WIN'
        elif abs(delta_cosine) < 0.02:
            rating = '🟡 EQUIVALENT'
        else:
            rating = '🔴 WORSE'

        row = "{:<30} {:<10.4f} {:<8.4f} {:<10.2f} {:<10}".format(
            approach_name.upper(), cosine, std, time_ms, rating)
        print(row)

    print("\n🎯 KEY FINDINGS:")
    print("-" * 80)

    # Find best approach
    best_approach = max(results.items(), key=lambda x: x[1]['avg_cosine'])
    print("• Best performance: {} ({:.4f} cosine)".format(best_approach[0].upper(), best_approach[1]['avg_cosine']))

    # Analyze multi-TR benefits
    single_tr = results.get('tiny_recursion_single', {})
    double_tr = results.get('tiny_recursion_double_weighted', {})
    triple_tr = results.get('tiny_recursion_triple_weighted', {})

    if single_tr and double_tr:
        single_cosine = single_tr['avg_cosine']
        double_cosine = double_tr['avg_cosine']
        improvement = double_cosine - single_cosine

        print("• Double-TR improvement: {:.4f} cosine ({:+.1f}%)".format(
            improvement, improvement/single_cosine*100))

        if improvement > 0:
            print("• ✅ DOUBLE-RUN HYPOTHESIS VALIDATED!")
        else:
            print("• ❌ Double-run didn't improve performance")

    if double_tr and triple_tr:
        triple_cosine = triple_tr['avg_cosine']
        double_to_triple = triple_cosine - double_cosine

        print("• Triple-TR improvement: {:.4f} cosine ({:+.1f}%)".format(
            double_to_triple, double_to_triple/double_cosine*100))

        if double_to_triple > 0:
            print("• ✅ TRIPLE-RUN ALSO BENEFICIAL!")

    # Speed analysis
    print("\n⚡ SPEED ANALYSIS:")
    print("• Direct: {:.1f}ms".format(results['direct_prediction']['avg_time']*1000))
    print("• Single TR: {:.1f}ms".format(single_tr.get('avg_time', 0)*1000))
    if double_tr:
        print("• Double TR: {:.1f}ms".format(double_tr.get('avg_time', 0)*1000))
    if triple_tr:
        print("• Triple TR: {:.1f}ms".format(triple_tr.get('avg_time', 0)*1000))

    # Efficiency metric (cosine per ms)
    print("\n🏆 EFFICIENCY (Cosine per ms):")
    for approach_name, metrics in results.items():
        efficiency = metrics['avg_cosine'] / (metrics['avg_time'] * 1000)
        print("• {}: {:.6f} cosine/ms".format(approach_name.upper(), efficiency))

    print("\n💡 RECOMMENDATIONS:")
    print("-" * 80)

    # Final recommendations
    best_overall = max(results.items(), key=lambda x: x[1]['avg_cosine'])
    if best_overall[0] != 'direct_prediction':
        print("• 🟢 DEPLOY {}: Best performance ({:.4f})".format(
            best_overall[0].upper(), best_overall[1]['avg_cosine']))
    else:
        print("• 🟡 DIRECT PREDICTION: Still competitive baseline")

    # Check if multi-TR beats direct
    if double_tr and double_tr['avg_cosine'] > baseline_cosine:
        improvement = double_tr['avg_cosine'] - baseline_cosine
        print("• 🟢 DOUBLE TR BEATS DIRECT: {:.4f} improvement!".format(improvement))

    return results

if __name__ == "__main__":
    results = test_multi_tr_performance()

    # Save results
    output_file = Path("artifacts/lvm/multi_tr_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {output_file}")

    print("
🎉 MULTI-TR TEST COMPLETE!"    print("=" * 60)
