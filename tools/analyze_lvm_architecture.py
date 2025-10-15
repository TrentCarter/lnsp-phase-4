#!/usr/bin/env python3
"""
Comprehensive LVM Architecture Analysis
========================================

Analyzes all 3 trained models to understand why vec2text fails:
1. Architecture comparison
2. Vector output properties (norm, mean, std)
3. Comparison with GTR-T5 vectors
4. Normalization analysis
"""

import sys
sys.path.insert(0, 'app/lvm')

import torch
import numpy as np
from pathlib import Path

# Import all 3 models
from train_lstm_baseline import LSTMVectorPredictor
from train_mamba2 import Mamba2VectorPredictor
from train_transformer import TransformerVectorPredictor

print("\n" + "="*80)
print("LVM ARCHITECTURE & VECTOR ANALYSIS")
print("="*80 + "\n")

# Load data
train_data = np.load('artifacts/lvm/training_sequences_ctx5.npz', allow_pickle=True)
context_sequences = train_data['context_sequences']
target_vectors = train_data['target_vectors']

# Use validation split
split_idx = int(0.9 * len(context_sequences))
val_contexts = context_sequences[split_idx:split_idx+100]
val_targets = target_vectors[split_idx:split_idx+100]

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

models = [
    ('LSTM', 'artifacts/lvm/models/lstm_baseline/best_model.pt', LSTMVectorPredictor, {'input_dim': 768, 'hidden_dim': 512, 'num_layers': 2}),
    ('GRU', 'artifacts/lvm/models/mamba2/best_model.pt', Mamba2VectorPredictor, {'input_dim': 768, 'd_model': 512, 'num_layers': 4}),
    ('Transformer', 'artifacts/lvm/models/transformer/best_model.pt', TransformerVectorPredictor, {'input_dim': 768, 'd_model': 512, 'nhead': 8, 'num_layers': 4}),
]

print("## ARCHITECTURE COMPARISON\n")
print("| Model | Parameters | Architecture | Normalization Layer |")
print("|-------|------------|--------------|---------------------|")

for name, path, model_class, kwargs in models:
    checkpoint = torch.load(path, map_location=device)
    model = model_class(**kwargs).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    params = sum(p.numel() for p in model.parameters())

    # Check architecture
    arch_type = model.__class__.__name__

    # Check normalization
    has_norm = any('norm' in n.lower() for n, _ in model.named_modules())
    norm_layers = [n for n, m in model.named_modules() if 'norm' in n.lower()]

    print(f"| {name} | {params:,} | {arch_type} | {', '.join(norm_layers) if norm_layers else 'None'} |")

print("\n" + "="*80)
print("## VECTOR OUTPUT ANALYSIS\n")
print("Analyzing 100 validation samples...\n")

# Analyze GTR-T5 target vectors first
target_norms = np.linalg.norm(val_targets, axis=1)
target_means = np.mean(val_targets, axis=1)
target_stds = np.std(val_targets, axis=1)

print("### GTR-T5 Target Vectors (Ground Truth)\n")
print(f"  L2 Norm:  mean={target_norms.mean():.4f}, std={target_norms.std():.4f}, min={target_norms.min():.4f}, max={target_norms.max():.4f}")
print(f"  Mean:     mean={target_means.mean():.6f}, std={target_means.std():.6f}")
print(f"  Std Dev:  mean={target_stds.mean():.4f}, std={target_stds.std():.4f}")
print()

# Analyze each model's outputs
for name, path, model_class, kwargs in models:
    print(f"### {name} Model Predictions\n")

    checkpoint = torch.load(path, map_location=device)
    model = model_class(**kwargs).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get predictions
    predictions = []
    cosines = []

    with torch.no_grad():
        for i in range(len(val_contexts)):
            context = torch.FloatTensor(val_contexts[i:i+1]).to(device)
            target = val_targets[i]

            pred = model(context).cpu().numpy()[0]
            predictions.append(pred)

            # Compute cosine similarity
            pred_norm = pred / (np.linalg.norm(pred) + 1e-8)
            target_norm = target / (np.linalg.norm(target) + 1e-8)
            cosine = np.dot(pred_norm, target_norm)
            cosines.append(cosine)

    predictions = np.array(predictions)
    pred_norms = np.linalg.norm(predictions, axis=1)
    pred_means = np.mean(predictions, axis=1)
    pred_stds = np.std(predictions, axis=1)

    print(f"  L2 Norm:  mean={pred_norms.mean():.4f}, std={pred_norms.std():.4f}, min={pred_norms.min():.4f}, max={pred_norms.max():.4f}")
    print(f"  Mean:     mean={pred_means.mean():.6f}, std={pred_means.std():.6f}")
    print(f"  Std Dev:  mean={pred_stds.mean():.4f}, std={pred_stds.std():.4f}")
    print(f"  Cosine:   mean={np.mean(cosines):.4f}, std={np.std(cosines):.4f}")
    print()

    # Norm ratio analysis
    norm_ratio = pred_norms.mean() / target_norms.mean()
    print(f"  **Norm Ratio** (pred/target): {norm_ratio:.4f}")

    if norm_ratio < 0.9 or norm_ratio > 1.1:
        print(f"  ⚠️  WARNING: Predicted vectors have {'smaller' if norm_ratio < 1 else 'larger'} norms than targets!")
        print(f"  This will break vec2text decoding, which expects specific norm ranges.")

    print()

print("="*80)
print("## KEY FINDINGS\n")
print("Vec2text expects vectors with:")
print("  • L2 Norm: ~10-12 (typical for GTR-T5 embeddings)")
print("  • Specific distributional properties matching GTR-T5 training data")
print("\nIf LVM predictions have significantly different norms/distributions,")
print("vec2text will fail even with high cosine similarity.")
print("\n" + "="*80 + "\n")
