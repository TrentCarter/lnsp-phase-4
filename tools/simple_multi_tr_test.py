#!/usr/bin/env python3
"""
Multi-Tiny Recursion Test - Simplified Version

Tests multiple TR runs and validates the ensemble hypothesis.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import time
from typing import Dict, Any

# Import LVM components
sys.path.insert(0, 'app/lvm')
from models import create_model

# Import Tiny Recursion
sys.path.insert(0, 'tools')
from comprehensive_lvm_evaluation_fixed import TinyRecursion

def load_test_data(num_samples: int = 50) -> Dict[str, Any]:
    """Load test dataset."""
    test_path = "artifacts/lvm/wikipedia_ood_test_ctx5.npz"
    if Path(test_path).exists():
        test_data = np.load(test_path, allow_pickle=True)
        return {
            'context_sequences': test_data['context_sequences'][:num_samples],
            'target_vectors': test_data['target_vectors'][:num_samples]
        }
    return None

def test_multi_tr():
    """Test multi-Tiny Recursion approaches."""

    print("🔬 MULTI-TINY RECURSION TEST")
    print("=" * 50)

    # Load data
    test_data = load_test_data(50)
    if not test_data:
        print("❌ No test data found")
        return

    # Load AMN model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model_path = "artifacts/lvm/production_model"
    if not Path(model_path).exists():
        print("❌ Model not found")
        return

    # Load model
    model_dir = Path(model_path)
    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / "final_model.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = checkpoint.get('hyperparameters', {})
    model = create_model('amn', **hparams)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("✅ Model loaded successfully")

    # Test approaches
    approaches = {
        'direct': 'Direct Prediction',
        'single_tr': 'Single Tiny Recursion',
        'double_tr': 'Double TR Ensemble',
        'triple_tr': 'Triple TR Ensemble'
    }

    results = {}

    for approach in approaches:
        print(f"\n🧪 Testing {approaches[approach]}...")

        cosines = []
        times = []

        for i in range(len(test_data['context_sequences'])):
            context_vecs = test_data['context_sequences'][i]
            target_vec = test_data['target_vectors'][i]

            # Convert to tensor
            context_t = torch.from_numpy(context_vecs).float().unsqueeze(0).to(device)

            start_time = time.time()

            if approach == 'direct':
                # Direct prediction
                with torch.no_grad():
                    pred_t = model(context_t)
                pred_vec = pred_t.cpu().numpy()[0]

            elif approach == 'single_tr':
                # Single TR
                tr_model = TinyRecursion(model)
                pred_tensor, _ = tr_model.refine_prediction(context_t)
                pred_vec = pred_tensor.cpu().numpy()[0]

            elif approach == 'double_tr':
                # Double TR with ensembling
                tr_model = TinyRecursion(model)

                # First run
                pred1, meta1 = tr_model.refine_prediction(context_t)

                # Enhanced context for second run
                batch_size, seq_len, dim = context_t.shape
                enhanced_context = context_t.clone()
                if seq_len > 1:
                    confidence_weight = min(0.5, meta1.get('confidence', 0.8))
                    enhanced_context[:, -1, :] = (1 - confidence_weight) * context_t[:, -1, :] + confidence_weight * pred1.squeeze(1)

                # Second run
                pred2, meta2 = tr_model.refine_prediction(enhanced_context)

                # Ensemble by confidence
                conf1 = meta1.get('confidence', 0.8)
                conf2 = meta2.get('confidence', 0.8)
                total_conf = conf1 + conf2
                w1, w2 = conf1 / total_conf, conf2 / total_conf

                pred_vec = (w1 * pred1 + w2 * pred2).cpu().numpy()[0]

            elif approach == 'triple_tr':
                # Triple TR
                tr_model = TinyRecursion(model)

                # First run
                pred1, meta1 = tr_model.refine_prediction(context_t)

                # Second run
                batch_size, seq_len, dim = context_t.shape
                enhanced_context = context_t.clone()
                if seq_len > 1:
                    confidence_weight = min(0.5, meta1.get('confidence', 0.8))
                    enhanced_context[:, -1, :] = (1 - confidence_weight) * context_t[:, -1, :] + confidence_weight * pred1.squeeze(1)

                pred2, meta2 = tr_model.refine_prediction(enhanced_context)

                # Third run
                enhanced_context2 = context_t.clone()
                if seq_len > 1:
                    avg_pred = (pred1 + pred2) / 2
                    confidence_weight = min(0.5, max(meta1.get('confidence', 0.8), meta2.get('confidence', 0.8)))
                    enhanced_context2[:, -1, :] = (1 - confidence_weight) * context_t[:, -1, :] + confidence_weight * avg_pred.squeeze(1)

                pred3, meta3 = tr_model.refine_prediction(enhanced_context2)

                # Ensemble all three
                conf1 = meta1.get('confidence', 0.8)
                conf2 = meta2.get('confidence', 0.8)
                conf3 = meta3.get('confidence', 0.8)
                total_conf = conf1 + conf2 + conf3
                w1, w2, w3 = conf1 / total_conf, conf2 / total_conf, conf3 / total_conf

                pred_vec = (w1 * pred1 + w2 * pred2 + w3 * pred3).cpu().numpy()[0]

            inference_time = time.time() - start_time

            # Calculate cosine similarity
            cosine_sim = np.dot(pred_vec, target_vec) / (np.linalg.norm(pred_vec) * np.linalg.norm(target_vec))

            cosines.append(cosine_sim)
            times.append(inference_time)

            if i % 10 == 0:
                print(f"  Sample {i}: {cosine_sim:.4f} ({inference_time:.4f}s)")

        # Store results
        results[approach] = {
            'cosine': np.mean(cosines),
            'cosine_std': np.std(cosines),
            'time': np.mean(times),
            'max_cosine': np.max(cosines),
            'min_cosine': np.min(cosines)
        }

        print(f"  Average: {np.mean(cosines):.4f} cosine, {np.mean(times):.4f}s")

    # Display results
    print("\n🏆 RESULTS SUMMARY:")
    print("-" * 50)

    baseline_cosine = results['direct']['cosine']
    baseline_time = results['direct']['time']

    print(f"{'Approach':<20} {'Cosine':<10} {'Delta':<10} {'Time (ms)':<10} {'Rating':<10}")
    print("-" * 70)

    for approach in ['direct', 'single_tr', 'double_tr', 'triple_tr']:
        if approach in results:
            data = results[approach]
            cosine = data['cosine']
            delta = cosine - baseline_cosine
            time_ms = data['time'] * 1000

            # Rating
            if approach == 'direct':
                rating = 'Baseline'
            elif delta > 0.01:
                rating = 'Better'
            elif abs(delta) < 0.01:
                rating = 'Equivalent'
            else:
                rating = 'Worse'

            print(f"{approaches[approach]:<20} {cosine:<10.4f} {delta:<10.4f} {time_ms:<10.2f} {rating:<10}")

    print("\n💡 ANALYSIS:")
    print("-" * 50)

    # Compare approaches
    if 'double_tr' in results and 'single_tr' in results:
        single_to_double = results['double_tr']['cosine'] - results['single_tr']['cosine']
        double_to_baseline = results['double_tr']['cosine'] - baseline_cosine

        print(f"Single TR vs Direct: {results['single_tr']['cosine'] - baseline_cosine:+.4f}")
        print(f"Double TR vs Direct: {double_to_baseline:+.4f}")
        print(f"Double TR vs Single: {single_to_double:+.4f}")

        if double_to_baseline > 0:
            print("✅ DOUBLE-RUN BEATS DIRECT PREDICTION!")
        if single_to_double > 0:
            print("✅ DOUBLE-RUN BEATS SINGLE TR!")

    print("\n🎯 CONCLUSION:")
    print("-" * 50)

    # Find best approach
    best_approach = max(results.items(), key=lambda x: x[1]['cosine'])
    print(f"Best performance: {best_approach[0].upper()} ({best_approach[1]['cosine']:.4f} cosine)")

    # Efficiency analysis
    print("\nEfficiency (cosine/ms):")
    for approach, data in results.items():
        efficiency = data['cosine'] / (data['time'] * 1000)
        print(f"  {approach.upper()}: {efficiency:.6f}")

    return results

if __name__ == "__main__":
    results = test_multi_tr()

    # Save results
    import json
    output_file = Path("artifacts/lvm/multi_tr_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {output_file}")
    print("\n🎉 Multi-TR test complete!")
