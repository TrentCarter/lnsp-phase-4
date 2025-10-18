#!/usr/bin/env python3
"""
Comprehensive LVM-T Inference Testing and Evaluation

Tests:
1. Single Prediction with Vec2Text Decoding
2. Batch Inference Performance
3. Held-Out Set Evaluation
4. Baseline Comparisons
5. Qualitative Analysis
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment for vec2text
os.environ['VEC2TEXT_FORCE_PROJECT_VENV'] = '1'
os.environ['VEC2TEXT_DEVICE'] = 'cpu'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

# ============================================================================
# Model Architecture (copied from training script)
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerVectorPredictor(nn.Module):
    """Transformer decoder for next-vector prediction"""
    def __init__(self, input_dim=768, d_model=512, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, input_dim),
        )

    def forward(self, x, return_raw: bool = False):
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        x = self.transformer_decoder(tgt=x, memory=x, tgt_mask=causal_mask, memory_mask=causal_mask)
        last_hidden = x[:, -1, :]
        raw = self.head(last_hidden)
        cos = nn.functional.normalize(raw, p=2, dim=-1)

        if return_raw:
            return raw, cos
        return cos


# ============================================================================
# Helper Functions
# ============================================================================

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict:
    """Compute evaluation metrics"""
    cosine_sims = []
    l2_distances = []

    for pred, target in zip(predictions, targets):
        cos_sim = cosine_similarity(pred, target)
        l2_dist = np.linalg.norm(pred - target)
        cosine_sims.append(cos_sim)
        l2_distances.append(l2_dist)

    return {
        'cosine_mean': float(np.mean(cosine_sims)),
        'cosine_std': float(np.std(cosine_sims)),
        'cosine_min': float(np.min(cosine_sims)),
        'cosine_max': float(np.max(cosine_sims)),
        'cosine_median': float(np.median(cosine_sims)),
        'l2_mean': float(np.mean(l2_distances)),
        'l2_std': float(np.std(l2_distances)),
        'top1_accuracy': float(np.mean([1.0 if c > 0.9 else 0.0 for c in cosine_sims])),
        'top5_accuracy': float(np.mean([1.0 if c > 0.5 else 0.0 for c in cosine_sims])),
    }


# ============================================================================
# Test 1: Load Model and Basic Inference
# ============================================================================

def test_model_loading(model_path: Path, device: str) -> TransformerVectorPredictor:
    """Test loading the trained model"""
    print("=" * 80)
    print("TEST 1: Model Loading")
    print("=" * 80)
    print()

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    print(f"Val Cosine: {checkpoint.get('val_cosine', 'N/A'):.4f}")
    print()

    # Create model
    model = TransformerVectorPredictor(
        input_dim=768,
        d_model=512,
        nhead=8,
        num_layers=4
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {num_params:,}")
    print(f"  Device: {device}")
    print()

    return model


# ============================================================================
# Test 2: Single Prediction with Vec2Text Decoding
# ============================================================================

def test_single_prediction(
    model: TransformerVectorPredictor,
    test_data: np.ndarray,
    test_texts: np.ndarray,
    orchestrator: IsolatedVecTextVectOrchestrator,
    device: str,
    num_samples: int = 10
):
    """Test single predictions and decode with vec2text"""
    print("=" * 80)
    print("TEST 2: Single Prediction with Vec2Text Decoding")
    print("=" * 80)
    print()

    results = []

    for i in range(num_samples):
        idx = np.random.randint(0, len(test_data) - 5)
        context_vecs = test_data[idx:idx+5]
        target_vec = test_data[idx+5]
        target_text = test_texts[idx+5]

        # Predict
        with torch.no_grad():
            context_tensor = torch.FloatTensor(context_vecs).unsqueeze(0).to(device)
            pred_vec = model(context_tensor).cpu().numpy()[0]

        # Compute similarity
        cos_sim = cosine_similarity(pred_vec, target_vec)

        # Decode both vectors with vec2text
        try:
            # Decode predicted vector
            pred_decoded = orchestrator.decode_vector(pred_vec, steps=1, subscribers=['jxe'])
            pred_text = pred_decoded.get('jxe', {}).get('text', '[DECODE FAILED]')

            # Decode target vector for comparison
            target_decoded = orchestrator.decode_vector(target_vec, steps=1, subscribers=['jxe'])
            target_text_decoded = target_decoded.get('jxe', {}).get('text', '[DECODE FAILED]')
        except Exception as e:
            pred_text = f"[ERROR: {e}]"
            target_text_decoded = target_text

        result = {
            'index': idx,
            'cosine_similarity': float(cos_sim),
            'target_text_original': target_text,
            'target_text_decoded': target_text_decoded,
            'predicted_text': pred_text,
            'context_texts': [test_texts[idx+j] for j in range(5)]
        }
        results.append(result)

        print(f"Sample {i+1}/{num_samples}:")
        print(f"  Cosine Similarity: {cos_sim:.4f}")
        print(f"  Target (original):  {target_text[:80]}...")
        print(f"  Target (decoded):   {target_text_decoded[:80]}...")
        print(f"  Predicted:          {pred_text[:80]}...")
        print()

    return results


# ============================================================================
# Test 3: Held-Out Set Evaluation
# ============================================================================

def test_heldout_evaluation(
    model: TransformerVectorPredictor,
    test_data: np.ndarray,
    device: str,
    context_length: int = 5,
    batch_size: int = 32
):
    """Evaluate on held-out test set"""
    print("=" * 80)
    print("TEST 3: Held-Out Set Evaluation")
    print("=" * 80)
    print()

    # Create test sequences
    num_test_sequences = min(1000, len(test_data) - context_length)
    print(f"Creating {num_test_sequences:,} test sequences...")

    all_predictions = []
    all_targets = []

    start_time = time.time()

    for i in range(0, num_test_sequences, batch_size):
        batch_end = min(i + batch_size, num_test_sequences)
        batch_contexts = []
        batch_targets = []

        for j in range(i, batch_end):
            context = test_data[j:j+context_length]
            target = test_data[j+context_length]
            batch_contexts.append(context)
            batch_targets.append(target)

        # Batch inference
        with torch.no_grad():
            context_tensor = torch.FloatTensor(np.array(batch_contexts)).to(device)
            predictions = model(context_tensor).cpu().numpy()

        all_predictions.extend(predictions)
        all_targets.extend(batch_targets)

        if (i // batch_size + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + batch_size) / elapsed
            print(f"  Processed {i + batch_size}/{num_test_sequences} "
                  f"({rate:.1f} samples/sec)")

    elapsed_time = time.time() - start_time

    # Compute metrics
    predictions_np = np.array(all_predictions)
    targets_np = np.array(all_targets)

    metrics = compute_metrics(predictions_np, targets_np)

    print()
    print("Results:")
    print(f"  Total samples:      {len(all_predictions):,}")
    print(f"  Total time:         {elapsed_time:.2f}s")
    print(f"  Throughput:         {len(all_predictions) / elapsed_time:.1f} samples/sec")
    print()
    print("Metrics:")
    print(f"  Cosine Similarity:  {metrics['cosine_mean']:.4f} ± {metrics['cosine_std']:.4f}")
    print(f"  Cosine Min/Max:     {metrics['cosine_min']:.4f} / {metrics['cosine_max']:.4f}")
    print(f"  Cosine Median:      {metrics['cosine_median']:.4f}")
    print(f"  L2 Distance:        {metrics['l2_mean']:.4f} ± {metrics['l2_std']:.4f}")
    print(f"  Top-1 Accuracy:     {metrics['top1_accuracy']:.2%} (cosine > 0.9)")
    print(f"  Top-5 Accuracy:     {metrics['top5_accuracy']:.2%} (cosine > 0.5)")
    print()

    return {
        'metrics': metrics,
        'predictions': predictions_np,
        'targets': targets_np,
        'num_samples': len(all_predictions),
        'elapsed_time': elapsed_time
    }


# ============================================================================
# Test 4: Baseline Comparisons
# ============================================================================

def test_baseline_comparisons(
    test_data: np.ndarray,
    lvm_predictions: np.ndarray,
    targets: np.ndarray,
    context_length: int = 5
):
    """Compare LVM-T against baseline models"""
    print("=" * 80)
    print("TEST 4: Baseline Comparisons")
    print("=" * 80)
    print()

    num_samples = len(targets)

    # Baseline 1: Random Unit Vectors
    print("Baseline 1: Random Unit Vectors")
    random_preds = np.random.randn(num_samples, 768)
    random_preds = random_preds / np.linalg.norm(random_preds, axis=1, keepdims=True)
    random_metrics = compute_metrics(random_preds, targets)
    print(f"  Cosine Similarity: {random_metrics['cosine_mean']:.4f} ± {random_metrics['cosine_std']:.4f}")
    print()

    # Baseline 2: Mean Vector
    print("Baseline 2: Mean Vector (repeat average)")
    mean_vec = np.mean(test_data, axis=0)
    mean_vec = mean_vec / np.linalg.norm(mean_vec)
    mean_preds = np.tile(mean_vec, (num_samples, 1))
    mean_metrics = compute_metrics(mean_preds, targets)
    print(f"  Cosine Similarity: {mean_metrics['cosine_mean']:.4f} ± {mean_metrics['cosine_std']:.4f}")
    print()

    # Baseline 3: Last Context Vector (simple persistence)
    print("Baseline 3: Last Context Vector (persistence)")
    persistence_preds = []
    for i in range(num_samples):
        last_context = test_data[i + context_length - 1]
        persistence_preds.append(last_context)
    persistence_preds = np.array(persistence_preds)
    persistence_metrics = compute_metrics(persistence_preds, targets)
    print(f"  Cosine Similarity: {persistence_metrics['cosine_mean']:.4f} ± {persistence_metrics['cosine_std']:.4f}")
    print()

    # Baseline 4: Linear Average of Context
    print("Baseline 4: Linear Average of Context")
    linear_preds = []
    for i in range(num_samples):
        context = test_data[i:i+context_length]
        avg = np.mean(context, axis=0)
        avg = avg / np.linalg.norm(avg)
        linear_preds.append(avg)
    linear_preds = np.array(linear_preds)
    linear_metrics = compute_metrics(linear_preds, targets)
    print(f"  Cosine Similarity: {linear_metrics['cosine_mean']:.4f} ± {linear_metrics['cosine_std']:.4f}")
    print()

    # LVM-T Performance
    print("LVM-T Transformer (Trained Model)")
    lvm_metrics = compute_metrics(lvm_predictions, targets)
    print(f"  Cosine Similarity: {lvm_metrics['cosine_mean']:.4f} ± {lvm_metrics['cosine_std']:.4f}")
    print()

    # Comparison table
    print("=" * 80)
    print("BASELINE COMPARISON TABLE")
    print("=" * 80)
    print()
    print(f"{'Model':<30} {'Cosine Mean':<15} {'Cosine Std':<15} {'Top-5 Acc':<15}")
    print("-" * 80)
    print(f"{'Random Vectors':<30} {random_metrics['cosine_mean']:<15.4f} {random_metrics['cosine_std']:<15.4f} {random_metrics['top5_accuracy']:<15.2%}")
    print(f"{'Mean Vector':<30} {mean_metrics['cosine_mean']:<15.4f} {mean_metrics['cosine_std']:<15.4f} {mean_metrics['top5_accuracy']:<15.2%}")
    print(f"{'Persistence (Last Context)':<30} {persistence_metrics['cosine_mean']:<15.4f} {persistence_metrics['cosine_std']:<15.4f} {persistence_metrics['top5_accuracy']:<15.2%}")
    print(f"{'Linear Average':<30} {linear_metrics['cosine_mean']:<15.4f} {linear_metrics['cosine_std']:<15.4f} {linear_metrics['top5_accuracy']:<15.2%}")
    print("-" * 80)
    print(f"{'LVM-T Transformer':<30} {lvm_metrics['cosine_mean']:<15.4f} {lvm_metrics['cosine_std']:<15.4f} {lvm_metrics['top5_accuracy']:<15.2%}")
    print("=" * 80)
    print()

    # Improvement calculations
    improvements = {}
    for name, baseline in [
        ('Random', random_metrics),
        ('Mean', mean_metrics),
        ('Persistence', persistence_metrics),
        ('Linear Avg', linear_metrics)
    ]:
        improvement = ((lvm_metrics['cosine_mean'] - baseline['cosine_mean']) /
                      baseline['cosine_mean'] * 100)
        improvements[name] = improvement

    print("LVM-T Improvements over Baselines:")
    for name, improvement in improvements.items():
        print(f"  vs {name:<15}: {improvement:+.1f}%")
    print()

    return {
        'random': random_metrics,
        'mean': mean_metrics,
        'persistence': persistence_metrics,
        'linear': linear_metrics,
        'lvm_t': lvm_metrics,
        'improvements': improvements
    }


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive LVM-T Testing")
    parser.add_argument('--model-path', default='artifacts/lvm/models/transformer_corrected_80k/best_model.pt')
    parser.add_argument('--data-path', default='artifacts/wikipedia_500k_corrected_vectors.npz')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--test-samples', type=int, default=1000)
    parser.add_argument('--decode-samples', type=int, default=10)
    parser.add_argument('--output-dir', default='artifacts/lvm/test_results')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LVM-T COMPREHENSIVE TESTING SUITE")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Model:        {args.model_path}")
    print(f"  Data:         {args.data_path}")
    print(f"  Device:       {args.device}")
    print(f"  Test samples: {args.test_samples}")
    print()

    # Load data
    print("Loading test data...")
    data = np.load(args.data_path, allow_pickle=True)
    vectors = data['vectors']
    texts = data['concept_texts']

    # Use last 20% as held-out test set
    train_size = int(0.8 * len(vectors))
    test_vectors = vectors[train_size:]
    test_texts = texts[train_size:]

    print(f"✓ Loaded {len(vectors):,} vectors")
    print(f"  Training set: {train_size:,}")
    print(f"  Test set:     {len(test_vectors):,}")
    print()

    # Initialize orchestrator for decoding
    print("Initializing Vec2Text orchestrator...")
    orchestrator = IsolatedVecTextVectOrchestrator()
    print("✓ Orchestrator loaded")
    print()

    # Test 1: Load Model
    model = test_model_loading(Path(args.model_path), args.device)

    # Test 2: Single Predictions with Decoding
    single_results = test_single_prediction(
        model, test_vectors, test_texts, orchestrator,
        args.device, num_samples=args.decode_samples
    )

    # Test 3: Held-Out Evaluation
    heldout_results = test_heldout_evaluation(
        model, test_vectors, args.device,
        context_length=5, batch_size=32
    )

    # Test 4: Baseline Comparisons
    baseline_results = test_baseline_comparisons(
        test_vectors,
        heldout_results['predictions'][:args.test_samples],
        heldout_results['targets'][:args.test_samples],
        context_length=5
    )

    # Save results
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    print()

    results = {
        'config': vars(args),
        'single_predictions': single_results,
        'heldout_metrics': heldout_results['metrics'],
        'baseline_comparisons': {
            'metrics': {
                'random': baseline_results['random'],
                'mean': baseline_results['mean'],
                'persistence': baseline_results['persistence'],
                'linear': baseline_results['linear'],
                'lvm_t': baseline_results['lvm_t']
            },
            'improvements': baseline_results['improvements']
        },
        'test_size': len(test_vectors),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    results_path = output_dir / 'lvm_t_test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {results_path}")
    print()

    print("=" * 80)
    print("TESTING COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
