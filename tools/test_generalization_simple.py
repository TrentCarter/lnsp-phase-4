#!/usr/bin/env python3
"""
Test 80k vs 232k models on truly HELD-OUT data.

Use concepts from the database that were NOT in the training sequences.
"""

import torch
import numpy as np
import sys
import psycopg2

sys.path.insert(0, 'app/lvm')
from models import create_model

def load_model(checkpoint_path):
    """Load a trained model"""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model = create_model(ckpt['model_type'], **ckpt.get('model_config', {}))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

def get_held_out_test_sequences(n_samples=500):
    """
    Get test sequences from database concepts NOT in either training set.

    Strategy: Use concepts from articles 4000-5000 (definitely not in first 3,431)
    """
    print(f"Loading {n_samples} held-out test sequences from database...")

    conn = psycopg2.connect("dbname=lnsp user=trentcarter")
    cur = conn.cursor()

    # Get concepts from higher article IDs
    cur.execute("""
        SELECT concept_vec
        FROM cpe_vectors
        WHERE article_id BETWEEN 4000 AND 5000
        ORDER BY article_id
        LIMIT %s
    """, (n_samples,))

    vectors = []
    for row in cur.fetchall():
        vec_str = row[0]
        # Parse array string to floats
        vec = np.array([float(x) for x in vec_str.strip('[]{}').split(',')], dtype=np.float32)
        if len(vec) == 768:
            vectors.append(vec)

    cur.close()
    conn.close()

    print(f"Loaded {len(vectors)} held-out vectors")

    # Create sequences (5 consecutive → predict 6th)
    sequences = []
    for i in range(0, len(vectors) - 5, 1):  # Sliding window
        context = np.array(vectors[i:i+5])
        target = vectors[i+5]
        sequences.append({'context': context, 'target': target})

    print(f"Created {len(sequences)} test sequences")
    return sequences

def test_model(model, test_sequences, device='cpu'):
    """Test model on held-out sequences"""
    cosines = []

    for seq in test_sequences:
        context = torch.FloatTensor(seq['context']).unsqueeze(0).to(device)  # (1, 5, 768)
        target = seq['target']

        with torch.no_grad():
            pred = model(context).cpu().numpy()[0]

        cosine = float(np.dot(target, pred) / (np.linalg.norm(target) * np.linalg.norm(pred)))
        cosines.append(cosine)

    return {
        'mean': np.mean(cosines),
        'std': np.std(cosines),
        'median': np.median(cosines),
        'min': np.min(cosines),
        'max': np.max(cosines)
    }

def main():
    print("=" * 80)
    print("Generalization Test: 80k vs 232k on Held-Out Data")
    print("=" * 80)
    print()

    # Get held-out test data
    test_sequences = get_held_out_test_sequences(n_samples=1000)

    if len(test_sequences) < 100:
        print("❌ Not enough test sequences")
        return

    models = [
        ('AMN',
         'artifacts/lvm/models/amn_20251016_133427/best_model.pt',
         'artifacts/lvm/models/amn_232k_20251017_090129/best_model.pt'),
        ('LSTM',
         'artifacts/lvm/models/lstm_20251016_133934/best_model.pt',
         'artifacts/lvm/models/lstm_232k_20251017_090129/best_model.pt'),
        ('GRU',
         'artifacts/lvm/models/gru_20251016_134451/best_model.pt',
         'artifacts/lvm/models/gru_232k_20251017_090129/best_model.pt'),
        ('Transformer',
         'artifacts/lvm/models/transformer_20251016_135606/best_model.pt',
         'artifacts/lvm/models/transformer_232k_20251017_090129/best_model.pt'),
    ]

    results = []

    for name, path_80k, path_232k in models:
        print(f"\nTesting {name}...")

        # Load models
        model_80k = load_model(path_80k)
        model_232k = load_model(path_232k)

        # Test
        print(f"  80k model...")
        metrics_80k = test_model(model_80k, test_sequences)

        print(f"  232k model...")
        metrics_232k = test_model(model_232k, test_sequences)

        # Get training validation scores
        ckpt_80k = torch.load(path_80k, map_location='cpu')
        ckpt_232k = torch.load(path_232k, map_location='cpu')

        train_val_80k = ckpt_80k.get('val_cosine', 0)
        train_val_232k = ckpt_232k.get('val_cosine', 0)

        improvement = ((metrics_232k['mean'] - metrics_80k['mean']) / metrics_80k['mean'] * 100)

        print()
        print(f"  Results for {name}:")
        print(f"    Training Validation:  80k={train_val_80k:.4f}, 232k={train_val_232k:.4f}")
        print(f"    Held-Out Test:        80k={metrics_80k['mean']:.4f}, 232k={metrics_232k['mean']:.4f}")
        print(f"    Generalization Improvement: {improvement:+.2f}%")

        results.append({
            'name': name,
            'train_val_80k': train_val_80k,
            'train_val_232k': train_val_232k,
            'held_out_80k': metrics_80k['mean'],
            'held_out_232k': metrics_232k['mean'],
            'improvement': improvement
        })

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY: Training Val vs Held-Out Generalization")
    print("=" * 80)
    print()

    print(f"{'Model':<12} {'Train Val':<20} {'Held-Out Test':<20} {'Generalization'}")
    print(f"{'':12} {'80k':>9} {'232k':>9} {'80k':>9} {'232k':>9} {'Improvement'}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<12} {r['train_val_80k']:>9.4f} {r['train_val_232k']:>9.4f} "
              f"{r['held_out_80k']:>9.4f} {r['held_out_232k']:>9.4f} {r['improvement']:>+11.2f}%")

    print()
    print("=" * 80)
    print("CRITICAL QUESTION: Does 232k generalize better despite lower validation scores?")
    print("=" * 80)
    print()

    avg_improvement = np.mean([r['improvement'] for r in results])

    if avg_improvement > 3:
        print(f"✅ YES! 232k generalizes {avg_improvement:+.2f}% better on average")
        print("   Lower validation scores were misleading - 232k learned better patterns!")
    elif avg_improvement < -3:
        print(f"❌ NO. 80k still generalizes better ({avg_improvement:.2f}%)")
        print("   Validation scores were correct - 80k is genuinely better")
    else:
        print(f"≈ MIXED RESULTS (avg {avg_improvement:+.2f}%)")
        print("   Dataset size doesn't significantly affect generalization")

if __name__ == '__main__':
    main()
