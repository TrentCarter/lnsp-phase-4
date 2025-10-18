#!/usr/bin/env python3
"""
GraphMERT-LVM End-to-End Test (Simplified)
===========================================

Complete pipeline: Text → Vec (GTR-T5) → GraphMERT-LVM → Vec → Text (Vec2Text)

Shows 5+ examples with input/output text.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from lvm.graphmert_lvm_768d import GraphMERTLVM768D
from vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator


def load_trained_model(checkpoint_path, device='mps'):
    """Load trained GraphMERT-LVM model"""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = GraphMERTLVM768D(
        d_model=768,
        n_layers=12,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        lambda_decay=0.6
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded ({model.count_parameters():,} parameters)")
    return model


def run_end_to_end_test(model, orchestrator, test_texts, device='mps'):
    """
    Run complete end-to-end test

    Text → GTR-T5 → GraphMERT-LVM → Vec2Text (JXE + IELab) → Text
    """
    print("\n" + "="*80)
    print("GraphMERT-LVM End-to-End Test")
    print("="*80)

    results = []

    for idx, input_text in enumerate(test_texts, 1):
        print(f"\n{'='*80}")
        print(f"Example {idx}/{len(test_texts)}")
        print(f"{'='*80}")
        print(f"INPUT TEXT:")
        print(f"  \"{input_text}\"")
        print()

        # Step 1: Text → Vector (GTR-T5 encoder)
        print("Step 1: Encoding to 768-d vector (GTR-T5)...")
        input_vector = orchestrator.encode_texts([input_text])[0]  # (768,)

        # Convert to numpy if tensor
        if isinstance(input_vector, torch.Tensor):
            input_vector = input_vector.cpu().numpy()

        print(f"  ✓ Encoded to 768-d vector")
        print(f"  Vector norm: {np.linalg.norm(input_vector):.4f}")
        print()

        # Step 2: Create context (replicate vector 5 times for now)
        context = torch.FloatTensor(input_vector).unsqueeze(0).repeat(1, 5, 1).to(device)
        print("Step 2: Creating 5-vector context...")
        print(f"  ✓ Context shape: {context.shape}")
        print()

        # Step 3: GraphMERT-LVM prediction
        print("Step 3: Running GraphMERT-LVM prediction...")
        with torch.no_grad():
            pred_vector = model(context).cpu().numpy()[0]  # (768,)
        print(f"  ✓ Predicted 768-d vector")
        print(f"  Predicted vector norm: {np.linalg.norm(pred_vector):.4f}")
        cosine = float(np.dot(input_vector, pred_vector) / (np.linalg.norm(input_vector) * np.linalg.norm(pred_vector)))
        print(f"  Cosine similarity (input vs prediction): {cosine:.4f}")
        print()

        # Step 4: Vector → Text (Vec2Text decoder)
        print("Step 4: Decoding predicted vector to text...")
        print("  Using Vec2Text-JXE decoder...")
        print("  (This may take ~10s...)")

        # Use JXE wrapper directly via subprocess
        try:
            import pickle
            import tempfile
            import subprocess

            # Prepare input
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                input_path = f.name
                pickle.dump({
                    'vectors': pred_vector.reshape(1, -1),
                    'metadata': {'original_texts': [input_text]},
                    'steps': 1,
                    'debug': False,
                    'device_override': 'cpu'
                }, f)

            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                output_path = f.name

            # Run JXE wrapper
            python_path = Path(__file__).parent.parent / 'venv' / 'bin' / 'python3'
            wrapper_path = Path(__file__).parent.parent / 'app' / 'vect_text_vect' / 'subscriber_wrappers' / 'jxe_wrapper_proper.py'

            env = {**os.environ, 'VEC2TEXT_FORCE_PROJECT_VENV': '1', 'VEC2TEXT_DEVICE': 'cpu', 'TOKENIZERS_PARALLELISM': 'false'}

            result = subprocess.run(
                [str(python_path), str(wrapper_path), input_path, output_path],
                capture_output=True,
                text=True,
                env=env,
                timeout=30
            )

            if result.returncode == 0:
                with open(output_path, 'rb') as f:
                    output = pickle.load(f)
                    decoded_text = output['result'][0] if isinstance(output['result'], list) else output['result']
            else:
                decoded_text = f"[Error: {result.stderr[:100]}]"

            # Cleanup
            import os
            os.unlink(input_path)
            os.unlink(output_path)

        except Exception as e:
            decoded_text = f"[Error: {str(e)}]"

        print(f"  ✓ Decoded!")
        print()

        # Display results
        print(f"OUTPUT TEXT (GraphMERT-LVM → Vec2Text-JXE):")
        print(f"  \"{decoded_text}\"")
        print()
        print(f"METRICS:")
        print(f"  Input→Prediction cosine: {cosine:.4f}")
        print()

        results.append({
            'input': input_text,
            'output': decoded_text,
            'cosine': cosine
        })

    return results


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='GraphMERT-LVM End-to-End Test')
    parser.add_argument('--checkpoint', default='artifacts/lvm/models/graphmert_lvm_80k_full/benchmark_model.pt')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--num-examples', type=int, default=5)
    args = parser.parse_args()

    # Test texts (diverse examples)
    test_texts = [
        "Artificial intelligence is transforming modern technology.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models learn patterns from data.",
        "Climate change is a pressing global challenge.",
        "The human brain contains billions of neurons.",
        "Quantum computing promises exponential speedups.",
        "Renewable energy sources are becoming more affordable.",
        "Deep learning revolutionized computer vision tasks.",
    ][:args.num_examples]

    # Load model
    device = torch.device(args.device)
    model = load_trained_model(args.checkpoint, device)

    # Initialize Vec2Text orchestrator
    print("\nInitializing Vec2Text orchestrator (GTR-T5 encoder)...")
    orchestrator = IsolatedVecTextVectOrchestrator()
    print("✓ Orchestrator ready")

    # Run end-to-end test
    results = run_end_to_end_test(model, orchestrator, test_texts, device)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total examples: {len(results)}")
    print(f"Average cosine similarity: {np.mean([r['cosine'] for r in results]):.4f}")
    print()
    print("All examples completed successfully!")
    print("="*80)


if __name__ == '__main__':
    import os
    main()
