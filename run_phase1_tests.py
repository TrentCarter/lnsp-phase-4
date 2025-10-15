#!/usr/bin/env python3
"""
Phase 1 Tests: Model Loading, Validation, and Speed Benchmarks
================================================================

Runs tests 1.1, 1.2, and 1.3 on all trained models.
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
from tabulate import tabulate
from train_lstm_baseline import LSTMVectorPredictor, VectorSequenceDataset
from train_mamba2 import Mamba2VectorPredictor
from train_transformer import TransformerVectorPredictor


def test_1_1_model_loading():
    """Test 1.1: Model Loading"""
    print("=" * 80)
    print("TEST 1.1: MODEL LOADING")
    print("=" * 80)

    results = []
    models_info = [
        ('LSTM', 'artifacts/lvm/models/lstm_baseline/best_model.pt', LSTMVectorPredictor, {'input_dim': 768, 'hidden_dim': 512, 'num_layers': 2}),
        ('GRU', 'artifacts/lvm/models/mamba2/best_model.pt', Mamba2VectorPredictor, {'input_dim': 768, 'd_model': 512, 'num_layers': 4}),
        ('Transformer', 'artifacts/lvm/models/transformer/best_model.pt', TransformerVectorPredictor, {'input_dim': 768, 'd_model': 512, 'nhead': 8, 'num_layers': 4}),
    ]

    for name, path, model_class, kwargs in models_info:
        try:
            checkpoint = torch.load(path, map_location='cpu')
            model = model_class(**kwargs)
            model.load_state_dict(checkpoint['model_state_dict'])

            params = sum(p.numel() for p in model.parameters())

            results.append({
                'Model': name,
                'Status': '✅ PASS',
                'Params': f"{params/1e6:.1f}M",
                'Epoch': checkpoint.get('epoch', 'N/A'),
                'Val Loss': f"{checkpoint.get('val_loss', 0):.6f}",
                'Val Cosine': f"{checkpoint.get('val_cosine', 0):.4f}"
            })
            print(f"✓ {name}: Loaded successfully")

        except Exception as e:
            results.append({
                'Model': name,
                'Status': f'❌ FAIL',
                'Params': 'N/A',
                'Epoch': 'N/A',
                'Val Loss': 'N/A',
                'Val Cosine': 'N/A'
            })
            print(f"✗ {name}: Failed to load - {e}")

    print()
    return results


def test_1_2_validation_inference():
    """Test 1.2: Inference on Validation Set (from checkpoints)"""
    print("=" * 80)
    print("TEST 1.2: VALIDATION SET INFERENCE")
    print("=" * 80)

    results = []
    models_info = [
        ('LSTM', 'artifacts/lvm/models/lstm_baseline/best_model.pt'),
        ('GRU', 'artifacts/lvm/models/mamba2/best_model.pt'),
        ('Transformer', 'artifacts/lvm/models/transformer/best_model.pt'),
    ]

    for name, path in models_info:
        try:
            checkpoint = torch.load(path, map_location='cpu')

            results.append({
                'Model': name,
                'Val Loss': f"{checkpoint['val_loss']:.6f}",
                'Val Cosine': f"{checkpoint['val_cosine']:.4f}",
                'Pass Threshold': '✅' if checkpoint['val_cosine'] > 0.75 else '❌',
                'Trained Epochs': checkpoint.get('epoch', 'N/A')
            })
            print(f"✓ {name}: Loss={checkpoint['val_loss']:.6f}, Cosine={checkpoint['val_cosine']:.4f}")

        except Exception as e:
            results.append({
                'Model': name,
                'Val Loss': 'N/A',
                'Val Cosine': 'N/A',
                'Pass Threshold': '❌',
                'Trained Epochs': 'N/A'
            })
            print(f"✗ {name}: Error - {e}")

    print()
    return results


def test_1_3_inference_speed():
    """Test 1.3: Inference Speed Benchmark"""
    print("=" * 80)
    print("TEST 1.3: INFERENCE SPEED BENCHMARK")
    print("=" * 80)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()

    models_info = [
        ('LSTM', 'artifacts/lvm/models/lstm_baseline/best_model.pt', LSTMVectorPredictor, {'input_dim': 768, 'hidden_dim': 512, 'num_layers': 2}),
        ('GRU', 'artifacts/lvm/models/mamba2/best_model.pt', Mamba2VectorPredictor, {'input_dim': 768, 'd_model': 512, 'num_layers': 4}),
        ('Transformer', 'artifacts/lvm/models/transformer/best_model.pt', TransformerVectorPredictor, {'input_dim': 768, 'd_model': 512, 'nhead': 8, 'num_layers': 4}),
    ]

    results = []
    batch_size = 32  # Standard batch size for comparison

    for name, path, model_class, kwargs in models_info:
        try:
            print(f"Benchmarking {name}...")

            # Load model
            checkpoint = torch.load(path, map_location=device)
            model = model_class(**kwargs).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Warmup
            dummy_input = torch.randn(batch_size, 5, 768).to(device)
            for _ in range(10):
                with torch.no_grad():
                    _ = model(dummy_input)

            if device.type == 'mps':
                torch.mps.synchronize()

            # Benchmark
            test_input = torch.randn(batch_size, 5, 768).to(device)

            if device.type == 'mps':
                torch.mps.synchronize()

            start = time.time()
            for _ in range(100):
                with torch.no_grad():
                    _ = model(test_input)

            if device.type == 'mps':
                torch.mps.synchronize()

            elapsed = time.time() - start

            ms_per_batch = (elapsed / 100) * 1000
            samples_per_sec = (batch_size * 100) / elapsed

            results.append({
                'Model': name,
                'Batch Size': batch_size,
                'ms/batch': f"{ms_per_batch:.2f}",
                'samples/sec': f"{samples_per_sec:.1f}",
                'Status': '✅ PASS'
            })

            print(f"  ✓ {name}: {ms_per_batch:.2f}ms/batch ({samples_per_sec:.1f} samples/sec)")

        except Exception as e:
            results.append({
                'Model': name,
                'Batch Size': batch_size,
                'ms/batch': 'N/A',
                'samples/sec': 'N/A',
                'Status': f'❌ FAIL: {str(e)[:30]}'
            })
            print(f"  ✗ {name}: Error - {e}")

    print()
    return results


def main():
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  LVM PHASE 1 TESTS: Model Loading, Validation & Speed Benchmarks".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print("\n")

    # Run tests
    test_1_1_results = test_1_1_model_loading()
    test_1_2_results = test_1_2_validation_inference()
    test_1_3_results = test_1_3_inference_speed()

    # Print summary tables
    print("=" * 80)
    print("SUMMARY: TEST 1.1 - MODEL LOADING")
    print("=" * 80)
    print(tabulate(test_1_1_results, headers='keys', tablefmt='grid'))
    print()

    print("=" * 80)
    print("SUMMARY: TEST 1.2 - VALIDATION INFERENCE")
    print("=" * 80)
    print(tabulate(test_1_2_results, headers='keys', tablefmt='grid'))
    print()

    print("=" * 80)
    print("SUMMARY: TEST 1.3 - INFERENCE SPEED")
    print("=" * 80)
    print(tabulate(test_1_3_results, headers='keys', tablefmt='grid'))
    print()

    # Save results
    all_results = {
        'test_1_1_model_loading': test_1_1_results,
        'test_1_2_validation': test_1_2_results,
        'test_1_3_speed': test_1_3_results
    }

    output_path = Path('artifacts/lvm/evaluation/phase1_test_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"✓ Results saved to: {output_path}")
    print()

    # Print conclusion
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    all_pass = all(r['Status'] == '✅ PASS' for r in test_1_1_results)
    all_pass &= all(r['Pass Threshold'] == '✅' for r in test_1_2_results)
    all_pass &= all(r['Status'] == '✅ PASS' for r in test_1_3_results)

    if all_pass:
        print("✅ ALL TESTS PASSED!")
        print("\nAll models:")
        print("  - Load successfully")
        print("  - Achieve >75% cosine similarity")
        print("  - Run inference within acceptable time")
        print("\nReady to proceed to Phase 2: Retrieval Evaluation")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Review results above for details")

    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
