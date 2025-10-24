#!/usr/bin/env python3
"""
Quick validation test for Tiny Recursion and TwoTower implementations.

Tests both approaches with a small sample to confirm they're working correctly.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import time
from typing import Tuple, Dict, Any

# Import LVM models and our new approaches
sys.path.insert(0, 'app/lvm')
from models import create_model

class TinyRecursion:
    """Tiny Recursion implementation for LVM output refinement."""

    def __init__(self, base_model: nn.Module, threshold: float = 0.05, max_attempts: int = 2):
        self.base_model = base_model
        self.threshold = threshold
        self.max_attempts = max_attempts

    def refine_prediction(self, context_t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply tiny recursion to refine predictions."""

        # Initial prediction
        with torch.no_grad():
            pred1 = self.base_model(context_t)

        metadata = {
            'attempts': 1,
            'converged': False,
            'delta_history': [],
            'final_prediction': pred1,
            'confidence': 1.0
        }

        # Try recursion if needed
        current_pred = pred1
        for attempt in range(1, self.max_attempts):
            # Create enhanced context that includes the prediction
            batch_size, seq_len, dim = context_t.shape

            # Blend prediction with original context using attention-like weighting
            enhanced_context = context_t.clone()

            # Simple approach: replace last vector with weighted combination
            if seq_len > 1:
                confidence_weight = min(0.5, 1.0 / (attempt + 1))
                enhanced_context[:, -1, :] = (1 - confidence_weight) * context_t[:, -1, :] + confidence_weight * current_pred.squeeze(1)

            # Second prediction
            with torch.no_grad():
                pred2 = self.base_model(enhanced_context)

            # Check convergence (cosine delta)
            delta = 1 - torch.cosine_similarity(current_pred, pred2, dim=-1).mean().item()
            metadata['delta_history'].append(delta)

            # If converged, use refined prediction
            if delta < self.threshold:
                metadata['converged'] = True
                metadata['final_prediction'] = pred2
                metadata['attempts'] = attempt + 1
                metadata['confidence'] = 1.0 - delta
                break

            current_pred = pred2
            metadata['attempts'] = attempt + 1

        return metadata['final_prediction'], metadata

class TwoTowerQueryEncoder(nn.Module):
    """TwoTower Query Encoder for context → query vector."""

    def __init__(self, input_dim=768, hidden_dim=512, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, context):
        """Encode context sequence to query vector."""
        _, hidden = self.gru(context)
        pooled = hidden[-1]
        query = self.proj(pooled)
        return torch.nn.functional.normalize(query, dim=-1)

def load_test_data(num_samples=5):
    """Load small test dataset."""
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

def test_approaches():
    """Test Tiny Recursion and TwoTower approaches."""

    print("🔬 QUICK VALIDATION: Tiny Recursion & TwoTower")
    print("=" * 60)

    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load test data
    test_data = load_test_data(num_samples=3)
    if not test_data:
        return

    # Load AMN model (best performing from analysis)
    model_path = "artifacts/lvm/production_model"
    model_dir = Path(model_path)

    if not model_dir.exists():
        print("❌ Model not found!")
        return

    print(f"Loading AMN model from {model_path}...")
    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / "final_model.pt"

    if not checkpoint_path.exists():
        print("❌ No checkpoint found!")
        return

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = checkpoint.get('hyperparameters', {})
    model = create_model('amn', **hparams)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("✅ AMN model loaded and ready!")

    # Test each approach
    approaches = {
        'direct_prediction': 'Direct LVM Forward Pass',
        'tiny_recursion': 'Tiny Recursion (Self-Refinement)',
        'twotower': 'TwoTower (Query Enhanced)'
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

            if approach_name == 'tiny_recursion':
                # Test Tiny Recursion
                tr_model = TinyRecursion(model, threshold=0.05, max_attempts=2)
                with torch.no_grad():
                    refined_pred, tr_metadata = tr_model.refine_prediction(context_t)

                pred_vec = refined_pred.cpu().numpy()[0]
                inference_time = time.time() - start_time

                # Calculate cosine similarity
                cosine_sim = np.dot(pred_vec, target_vec) / (np.linalg.norm(pred_vec) * np.linalg.norm(target_vec))

                approach_results.append({
                    'sample_id': i,
                    'cosine_similarity': cosine_sim,
                    'inference_time': inference_time,
                    'tr_metadata': tr_metadata
                })

                print(f"   Sample {i}: {cosine_sim:.4f} (TR: {tr_metadata['converged']}, {tr_metadata['attempts']} attempts)")

            elif approach_name == 'twotower':
                # Test TwoTower
                query_encoder = TwoTowerQueryEncoder().to(device)

                with torch.no_grad():
                    query_vec = query_encoder(context_t)
                    enhanced_context = context_t.clone()
                    enhanced_context[:, -1, :] = query_vec
                    pred_t = model(enhanced_context)

                pred_vec = pred_t.cpu().numpy()[0]
                inference_time = time.time() - start_time

                # Calculate cosine similarity
                cosine_sim = np.dot(pred_vec, target_vec) / (np.linalg.norm(pred_vec) * np.linalg.norm(target_vec))

                approach_results.append({
                    'sample_id': i,
                    'cosine_similarity': cosine_sim,
                    'inference_time': inference_time,
                    'query_vector_norm': query_vec.norm().item()
                })

                print(f"   Sample {i}: {cosine_sim:.4f} (Query norm: {query_vec.norm().item():.3f})")

            else:
                # Direct prediction
                with torch.no_grad():
                    pred_t = model(context_t)

                pred_vec = pred_t.cpu().numpy()[0]
                inference_time = time.time() - start_time

                # Calculate cosine similarity
                cosine_sim = np.dot(pred_vec, target_vec) / (np.linalg.norm(pred_vec) * np.linalg.norm(target_vec))

                approach_results.append({
                    'sample_id': i,
                    'cosine_similarity': cosine_sim,
                    'inference_time': inference_time
                })

                print(f"   Sample {i}: {cosine_sim:.4f}")

        # Calculate average performance for this approach
        if approach_results:
            avg_cosine = np.mean([r['cosine_similarity'] for r in approach_results])
            avg_time = np.mean([r['inference_time'] for r in approach_results])

            results[approach_name] = {
                'avg_cosine': avg_cosine,
                'avg_time': avg_time,
                'samples': approach_results
            }

            print(f"   📊 Average: {avg_cosine:.4f} cosine, {avg_time:.4f}s per prediction")

    # Compare approaches
    print("\n🏆 APPROACH COMPARISON:")
    for approach_name, metrics in results.items():
        print(f"  {approach_name.upper()}: {metrics['avg_cosine']:.4f} cosine, {metrics['avg_time']:.4f}s")

    # Show Tiny Recursion specific metrics
    if 'tiny_recursion' in results:
        tr_samples = results['tiny_recursion']['samples']
        convergence_rate = np.mean([1 if s.get('tr_metadata', {}).get('converged', False) else 0 for s in tr_samples])
        avg_attempts = np.mean([s.get('tr_metadata', {}).get('attempts', 1) for s in tr_samples])
        avg_confidence = np.mean([s.get('tr_metadata', {}).get('confidence', 1.0) for s in tr_samples])

        print("\n🔄 TINY RECURSION METRICS:")
        print(f"  Convergence Rate: {convergence_rate:.1%}")
        print(f"  Average Attempts: {avg_attempts:.1f}")
        print(f"  Average Confidence: {avg_confidence:.3f}")

    # Show TwoTower specific metrics
    if 'twotower' in results:
        tt_samples = results['twotower']['samples']
        avg_query_norm = np.mean([s.get('query_vector_norm', 1.0) for s in tt_samples])
        print("\n🏗️  TWOTOWER METRICS:")
        print(f"  Average Query Vector Norm: {avg_query_norm:.3f}")

    print("\n✅ VALIDATION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    test_approaches()
