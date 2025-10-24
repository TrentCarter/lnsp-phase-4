#!/usr/bin/env python3
"""
Tiny Recursion Double-Run Enhancement

Tests the hypothesis: Run TR twice and ensemble the results for better performance.
Since TR is faster and converges well, double execution could outperform single direct prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import time
from typing import Dict, Any, Tuple

# Import our Tiny Recursion implementation
sys.path.insert(0, 'tools')
from comprehensive_lvm_evaluation_fixed import TinyRecursion, TwoTowerQueryEncoder

class TinyRecursionEnsemble:
    """Enhanced Tiny Recursion with ensemble methods."""

    def __init__(self, base_model: nn.Module, threshold: float = 0.05, max_attempts: int = 2):
        self.base_model = base_model
        self.threshold = threshold
        self.max_attempts = max_attempts
        self.tr_model = TinyRecursion(base_model, threshold, max_attempts)

    def predict_single(self, context_t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Single Tiny Recursion prediction."""
        return self.tr_model.refine_prediction(context_t)

    def predict_double_ensemble(self, context_t: torch.Tensor, ensemble_method: str = 'weighted') -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Double Tiny Recursion run with ensembling."""

        # First run
        start_time = time.time()
        pred1, meta1 = self.predict_single(context_t)
        time1 = time.time() - start_time

        # Second run with enhanced context using first prediction
        batch_size, seq_len, dim = context_t.shape
        enhanced_context = context_t.clone()

        if seq_len > 1:
            # Use first prediction to enhance context
            confidence_weight = min(0.5, meta1.get('confidence', 0.8))
            enhanced_context[:, -1, :] = (1 - confidence_weight) * context_t[:, -1, :] + confidence_weight * pred1.squeeze(1)

        # Second run
        pred2, meta2 = self.predict_single(enhanced_context)
        time2 = time.time() - start_time - time1

        total_time = time.time() - start_time

        # Ensemble methods
        if ensemble_method == 'weighted':
            # Weight by confidence and convergence
            conf1 = meta1.get('confidence', 0.8)
            conf2 = meta2.get('confidence', 0.8)

            # Normalize weights
            total_conf = conf1 + conf2
            w1, w2 = conf1 / total_conf, conf2 / total_conf

            final_pred = w1 * pred1 + w2 * pred2

        elif ensemble_method == 'converged_preferred':
            # Prefer the more converged prediction
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
                # Neither converged, use first prediction
                final_pred = pred1

        elif ensemble_method == 'average':
            # Simple average
            final_pred = (pred1 + pred2) / 2

        else:
            # Default to first prediction
            final_pred = pred1

        # Combined metadata
        combined_meta = {
            'ensemble_method': ensemble_method,
            'total_time': total_time,
            'first_run_time': time1,
            'second_run_time': time2,
            'first_meta': meta1,
            'second_meta': meta2,
            'final_confidence': max(meta1.get('confidence', 0.8), meta2.get('confidence', 0.8))
        }

        return final_pred, combined_meta

    def predict_triple_ensemble(self, context_t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Triple Tiny Recursion run for maximum refinement."""

        # First run
        pred1, meta1 = self.predict_single(context_t)

        # Second run
        batch_size, seq_len, dim = context_t.shape
        enhanced_context = context_t.clone()
        if seq_len > 1:
            confidence_weight = min(0.5, meta1.get('confidence', 0.8))
            enhanced_context[:, -1, :] = (1 - confidence_weight) * context_t[:, -1, :] + confidence_weight * pred1.squeeze(1)

        pred2, meta2 = self.predict_single(enhanced_context)

        # Third run
        enhanced_context2 = context_t.clone()
        if seq_len > 1:
            # Use both previous predictions
            avg_pred = (pred1 + pred2) / 2
            confidence_weight = min(0.5, max(meta1.get('confidence', 0.8), meta2.get('confidence', 0.8)))
            enhanced_context2[:, -1, :] = (1 - confidence_weight) * context_t[:, -1, :] + confidence_weight * avg_pred.squeeze(1)

        pred3, meta3 = self.predict_single(enhanced_context2)

        # Ensemble all three
        conf1 = meta1.get('confidence', 0.8)
        conf2 = meta2.get('confidence', 0.8)
        conf3 = meta3.get('confidence', 0.8)

        total_conf = conf1 + conf2 + conf3
        w1, w2, w3 = conf1 / total_conf, conf2 / total_conf, conf3 / total_conf

        final_pred = w1 * pred1 + w2 * pred2 + w3 * pred3

        combined_meta = {
            'ensemble_method': 'triple_weighted',
            'total_time': time.time() - time.time() + (meta1.get('total_time', 0) + meta2.get('total_time', 0) + meta3.get('total_time', 0)),
            'predictions': [meta1, meta2, meta3],
            'weights': [w1, w2, w3],
            'final_confidence': max(conf1, conf2, conf3)
        }

        return final_pred, combined_meta

def test_double_run_hypothesis():
    """Test the hypothesis that double TR runs can outperform single direct prediction."""

    print("🔬 TESTING DOUBLE-RUN TINY RECURSION HYPOTHESIS")
    print("=" * 60)

    # Load test data
    test_data = load_test_data(num_samples=20)  # Larger sample for better statistics
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
        'tiny_recursion_double_weighted': 'Double TR (weighted)',
        'tiny_recursion_double_converged': 'Double TR (converged pref)',
        'tiny_recursion_triple': 'Triple TR (maximum refinement)'
    }

    results = {}

    for approach_name, description in approaches.items():
        print(f"\n🧪 Testing {description}...")
        print(f"   Approach: {approach_name}")

        approach_results = []

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

            elif approach_name == 'tiny_recursion_single':
                # Single TR
                tr_ensemble = TinyRecursionEnsemble(model)
                pred_tensor, tr_meta = tr_ensemble.predict_single(context_t)
                pred_vec = pred_tensor.cpu().numpy()[0]

            elif approach_name == 'tiny_recursion_double_weighted':
                # Double TR weighted
                tr_ensemble = TinyRecursionEnsemble(model)
                pred_tensor, tr_meta = tr_ensemble.predict_double_ensemble(context_t, 'weighted')
                pred_vec = pred_tensor.cpu().numpy()[0]

            elif approach_name == 'tiny_recursion_double_converged':
                # Double TR converged preferred
                tr_ensemble = TinyRecursionEnsemble(model)
                pred_tensor, tr_meta = tr_ensemble.predict_double_ensemble(context_t, 'converged_preferred')
                pred_vec = pred_tensor.cpu().numpy()[0]

            elif approach_name == 'tiny_recursion_triple':
                # Triple TR
                tr_ensemble = TinyRecursionEnsemble(model)
                pred_tensor, tr_meta = tr_ensemble.predict_triple_ensemble(context_t)
                pred_vec = pred_tensor.cpu().numpy()[0]

            inference_time = time.time() - start_time

            # Calculate cosine similarity
            cosine_sim = np.dot(pred_vec, target_vec) / (np.linalg.norm(pred_vec) * np.linalg.norm(target_vec))

            approach_results.append({
                'sample_id': i,
                'cosine_similarity': cosine_sim,
                'inference_time': inference_time,
                'tr_metadata': tr_meta if 'tr_meta' in locals() else None
            })

            if i % 5 == 0:
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

    # Display comparison
    print(f"\n🏆 APPROACH COMPARISON:")
    header = "{:<25} {:<10} {:<8} {:<10} {:<10}".format('Approach', 'Cosine', '±Std', 'Time (ms)', 'Rating')
    print(header)
    print("-" * 70)

    baseline_cosine = results['direct_prediction']['avg_cosine']
    baseline_time = results['direct_prediction']['avg_time']

    for approach_name, metrics in results.items():
        cosine = metrics['avg_cosine']
        std = metrics['std_cosine']
        time_ms = metrics['avg_time'] * 1000

        # Rating
        delta_cosine = cosine - baseline_cosine
        delta_time = metrics['avg_time'] - baseline_time

        if approach_name == 'direct_prediction':
            rating = '⚡ BASELINE'
        elif abs(delta_cosine) < 0.01 and delta_time < 0:
            rating = '🟢 WIN-WIN'
        elif delta_cosine > 0:
            rating = '🟢 BETTER'
        elif abs(delta_cosine) < 0.02:
            rating = '🟡 EQUIVALENT'
        else:
            rating = '🔴 WORSE'

        row = "{:<25} {:<10.4f} {:<8.4f} {:<10.2f} {:<10}".format(
            approach_name.upper(), cosine, std, time_ms, rating)
        print(row)

    print("\n💡 CONCLUSION:")
    print("-" * 70)

    # Find best approach
    best_approach = max(results.items(), key=lambda x: x[1]['avg_cosine'])
    print("• Best performance: {} ({:.4f} cosine)".format(best_approach[0].upper(), best_approach[1]['avg_cosine']))

    # Check double-run hypothesis
    single_tr = results.get('tiny_recursion_single', {})
    double_weighted = results.get('tiny_recursion_double_weighted', {})

    if single_tr and double_weighted:
        single_cosine = single_tr['avg_cosine']
        double_cosine = double_weighted['avg_cosine']
        improvement = double_cosine - single_cosine

        print("• Double-run improvement: {:.4f} cosine ({:+.1f}%)".format(
            improvement, improvement/single_cosine*100))
        print("• Double-run vs baseline: {:.4f} cosine".format(double_cosine - baseline_cosine))

        if improvement > 0:
            print("• ✅ DOUBLE-RUN HYPOTHESIS VALIDATED!")
        else:
            print("• ❌ Double-run didn't improve performance")

    return results

def load_test_data(num_samples=10):
    """Load test dataset."""
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

if __name__ == "__main__":
    test_double_run_hypothesis()
