#!/usr/bin/env python3
"""
Test 80k vs 232k models on HELD-OUT data to check generalization.

The validation sets used during training may be too similar to training data.
This tests on completely unseen Wikipedia articles NOT in either dataset.
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, 'app/lvm')
from models import create_model

# Import vec2text orchestrator for encoding
sys.path.insert(0, 'app/vect_text_vect')
from vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

def load_model(checkpoint_path, model_type='standard'):
    """Load a trained model"""
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    if model_type == 'graphmert':
        from graphmert_lvm_768d import GraphMERTLVM768D
        model = GraphMERTLVM768D(d_model=768, n_layers=12, n_heads=8,
                                  d_ff=2048, dropout=0.1, lambda_decay=0.6)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model = create_model(ckpt['model_type'], **ckpt.get('model_config', {}))
        model.load_state_dict(ckpt['model_state_dict'])

    model.eval()
    return model

def create_held_out_test_set(orchestrator, n_samples=100):
    """
    Create a held-out test set from Wikipedia articles NOT in training.

    We'll use articles 5000-5100 (definitely not in first 3,431 articles)
    """
    print("Creating held-out test set from Wikipedia articles 5000-5100...")

    import json
    test_sequences = []

    with open('data/datasets/wikipedia/wikipedia_500k.jsonl', 'r') as f:
        for idx, line in enumerate(f):
            if idx < 5000:
                continue
            if idx >= 5000 + n_samples:
                break

            article = json.loads(line)
            text = article.get('text', '')

            if len(text) > 100:  # Skip very short articles
                # Take first ~500 chars as context
                context_text = text[:500]
                test_sequences.append(context_text)

    print(f"Loaded {len(test_sequences)} held-out articles")

    # Encode to vectors
    print("Encoding test articles to vectors...")
    vectors = orchestrator.encode_texts(test_sequences)

    # Create sequences (5 consecutive 100-char chunks → predict 6th)
    sequences = []
    for text in test_sequences:
        chunks = [text[i:i+100] for i in range(0, min(len(text), 600), 100)]
        if len(chunks) >= 6:
            # Encode each chunk
            chunk_vectors = orchestrator.encode_texts(chunks[:6])
            context = chunk_vectors[:5]  # First 5 as context
            target = chunk_vectors[5]    # 6th as target

            sequences.append({
                'context': context,
                'target': target,
                'text': text[:600]
            })

    print(f"Created {len(sequences)} test sequences")
    return sequences

def test_model_on_held_out(model, test_sequences, device='cpu'):
    """Test model on held-out data"""
    cosine_similarities = []

    for seq in test_sequences:
        context = torch.FloatTensor(seq['context']).unsqueeze(0).to(device)  # (1, 5, 768)
        target = torch.FloatTensor(seq['target']).to(device)

        with torch.no_grad():
            pred = model(context).cpu().numpy()[0]  # (768,)

        target_np = seq['target']

        # Cosine similarity
        cosine = float(
            np.dot(target_np, pred) /
            (np.linalg.norm(target_np) * np.linalg.norm(pred))
        )

        cosine_similarities.append(cosine)

    return {
        'mean_cosine': np.mean(cosine_similarities),
        'std_cosine': np.std(cosine_similarities),
        'median_cosine': np.median(cosine_similarities),
        'min_cosine': np.min(cosine_similarities),
        'max_cosine': np.max(cosine_similarities)
    }

def main():
    print("=" * 80)
    print("Generalization Test: 80k vs 232k Models on Held-Out Data")
    print("=" * 80)
    print()
    print("Testing on Wikipedia articles 5000-5100 (NOT in either training set)")
    print()

    # Initialize encoder
    print("Loading encoder...")
    orchestrator = IsolatedVecTextVectOrchestrator()

    # Create held-out test set
    test_sequences = create_held_out_test_set(orchestrator, n_samples=50)

    if not test_sequences:
        print("❌ Failed to create test sequences")
        return

    models = [
        ('AMN', 'artifacts/lvm/models/amn_20251016_133427/best_model.pt',
                'artifacts/lvm/models/amn_232k_20251017_090129/best_model.pt', 'standard'),
        ('LSTM', 'artifacts/lvm/models/lstm_20251016_133934/best_model.pt',
                 'artifacts/lvm/models/lstm_232k_20251017_090129/best_model.pt', 'standard'),
        ('GRU', 'artifacts/lvm/models/gru_20251016_134451/best_model.pt',
                'artifacts/lvm/models/gru_232k_20251017_090129/best_model.pt', 'standard'),
        ('Transformer', 'artifacts/lvm/models/transformer_20251016_135606/best_model.pt',
                       'artifacts/lvm/models/transformer_232k_20251017_090129/best_model.pt', 'standard'),
    ]

    results = []

    for name, path_80k, path_232k, model_type in models:
        print()
        print("=" * 80)
        print(f"Testing: {name}")
        print("=" * 80)

        # Load 80k model
        print(f"  Loading 80k model...")
        model_80k = load_model(path_80k, model_type)

        # Load 232k model
        print(f"  Loading 232k model...")
        model_232k = load_model(path_232k, model_type)

        # Test both
        print(f"  Testing 80k model on {len(test_sequences)} held-out sequences...")
        metrics_80k = test_model_on_held_out(model_80k, test_sequences)

        print(f"  Testing 232k model on {len(test_sequences)} held-out sequences...")
        metrics_232k = test_model_on_held_out(model_232k, test_sequences)

        # Compare
        improvement = ((metrics_232k['mean_cosine'] - metrics_80k['mean_cosine']) /
                      metrics_80k['mean_cosine'] * 100)

        print()
        print(f"  80k Held-Out Performance:")
        print(f"    Mean cosine: {metrics_80k['mean_cosine']:.4f} ± {metrics_80k['std_cosine']:.4f}")
        print(f"    Median: {metrics_80k['median_cosine']:.4f}")
        print(f"    Range: [{metrics_80k['min_cosine']:.4f}, {metrics_80k['max_cosine']:.4f}]")

        print()
        print(f"  232k Held-Out Performance:")
        print(f"    Mean cosine: {metrics_232k['mean_cosine']:.4f} ± {metrics_232k['std_cosine']:.4f}")
        print(f"    Median: {metrics_232k['median_cosine']:.4f}")
        print(f"    Range: [{metrics_232k['min_cosine']:.4f}, {metrics_232k['max_cosine']:.4f}]")

        print()
        print(f"  Generalization Improvement: {improvement:+.2f}%")

        if improvement > 2:
            print(f"  ✅ 232k generalizes BETTER (+{improvement:.2f}%)")
        elif improvement < -2:
            print(f"  ❌ 80k generalizes BETTER ({improvement:.2f}%)")
        else:
            print(f"  ≈ Similar generalization")

        results.append({
            'name': name,
            'held_out_80k': metrics_80k['mean_cosine'],
            'held_out_232k': metrics_232k['mean_cosine'],
            'improvement': improvement
        })

    # Summary
    print()
    print()
    print("=" * 80)
    print("GENERALIZATION SUMMARY")
    print("=" * 80)
    print()

    print(f"{'Model':<15} {'80k Held-Out':<15} {'232k Held-Out':<15} {'Improvement':<15} {'Verdict'}")
    print("-" * 80)

    for r in results:
        verdict = "✅ 232k better" if r['improvement'] > 2 else "❌ 80k better" if r['improvement'] < -2 else "≈ Similar"
        print(f"{r['name']:<15} {r['held_out_80k']:<15.4f} {r['held_out_232k']:<15.4f} {r['improvement']:>+14.2f}%  {verdict}")

    print()
    avg_improvement = np.mean([r['improvement'] for r in results])
    print(f"Average Generalization Improvement: {avg_improvement:+.2f}%")
    print()

    if avg_improvement > 2:
        print("🎯 CONCLUSION: 232k models generalize BETTER to unseen data!")
        print("   Even though validation scores were lower, they learned more general patterns.")
    elif avg_improvement < -2:
        print("⚠️  CONCLUSION: 80k models still generalize better")
    else:
        print("≈ CONCLUSION: Similar generalization, dataset size doesn't matter much")

if __name__ == '__main__':
    main()
