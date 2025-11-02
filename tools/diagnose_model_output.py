#!/usr/bin/env python3
"""
Diagnose Model Output Scale Issue
==================================

Check what the model is actually outputting and why cosine similarities are so small.
"""
import numpy as np
import torch
import sys

def l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, keepdims=True)
    return x / (n + eps)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))

print("=" * 80)
print("MODEL OUTPUT DIAGNOSTIC")
print("=" * 80)
print()

# Load model
model_path = "artifacts/lvm/models/amn_584k_clean/best_model.pt"
print(f"Loading model: {model_path}")
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

# Check if it's a checkpoint dict or direct model
if isinstance(checkpoint, dict):
    print(f"✅ Checkpoint dict with keys: {list(checkpoint.keys())}")

    # Get metadata
    if 'model_type' in checkpoint:
        print(f"   Model type: {checkpoint['model_type']}")
    if 'val_cosine' in checkpoint:
        print(f"   Checkpoint val_cosine: {checkpoint['val_cosine']:.4f}")
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")

    # Load model architecture
    model_type = checkpoint.get('model_type', 'amn').lower()
    sys.path.insert(0, 'app/lvm')
    from model import AMNModel, GRUModel, LSTMModel, TransformerModel

    if model_type == 'amn':
        model = AMNModel(input_dim=768, d_model=256, hidden_dim=512)
    elif model_type == 'gru':
        model = GRUModel(input_dim=768, d_model=256, hidden_dim=512)
    elif model_type == 'lstm':
        model = LSTMModel(input_dim=768, d_model=256, hidden_dim=512)
    elif model_type == 'transformer':
        model = TransformerModel(input_dim=768, d_model=256, hidden_dim=512)
    else:
        print(f"❌ Unknown model type: {model_type}")
        sys.exit(1)

    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print(f"✅ Direct model object")
    model = checkpoint

model.eval()
print()

# Load validation data
val_path = "artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz"
print(f"Loading validation data: {val_path}")
data = np.load(val_path, allow_pickle=True)
contexts = data['contexts']
targets = data['targets']
print(f"   Contexts shape: {contexts.shape}")
print(f"   Targets shape: {targets.shape}")
print()

# Take first 5 samples
print("Testing first 5 samples:")
print("-" * 80)
for i in range(5):
    ctx = contexts[i]  # (5, 768)
    tgt = targets[i]   # (768,)

    # Model prediction
    with torch.no_grad():
        ctx_t = torch.from_numpy(ctx).unsqueeze(0).float()  # (1, 5, 768)
        pred_t = model(ctx_t)  # (1, 768)
        pred = pred_t.squeeze(0).numpy()  # (768,)

    # Check raw output statistics
    pred_norm = np.linalg.norm(pred)
    pred_mean = np.mean(pred)
    pred_std = np.std(pred)
    pred_min = np.min(pred)
    pred_max = np.max(pred)

    # Normalize
    pred_normalized = l2_normalize(pred)
    tgt_normalized = l2_normalize(tgt)

    # Cosine similarity
    cos_sim = cosine(pred, tgt)

    print(f"\nSample {i+1}:")
    print(f"  Raw prediction:")
    print(f"    Norm: {pred_norm:.6f}")
    print(f"    Mean: {pred_mean:.6f}")
    print(f"    Std:  {pred_std:.6f}")
    print(f"    Range: [{pred_min:.6f}, {pred_max:.6f}]")
    print(f"  Normalized prediction:")
    print(f"    Norm: {np.linalg.norm(pred_normalized):.6f} (should be 1.0)")
    print(f"  Cosine similarity to target: {cos_sim:.6f}")

print()
print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)

# Compute statistics over larger sample
n_samples = min(1000, len(contexts))
cosines = []
raw_norms = []

for i in range(n_samples):
    ctx = contexts[i]
    tgt = targets[i]

    with torch.no_grad():
        ctx_t = torch.from_numpy(ctx).unsqueeze(0).float()
        pred_t = model(ctx_t)
        pred = pred_t.squeeze(0).numpy()

    raw_norms.append(np.linalg.norm(pred))
    cosines.append(cosine(pred, tgt))

print(f"\nStatistics over {n_samples} samples:")
print(f"  Mean cosine similarity: {np.mean(cosines):.6f}")
print(f"  Std cosine similarity:  {np.std(cosines):.6f}")
print(f"  Min cosine similarity:  {np.min(cosines):.6f}")
print(f"  Max cosine similarity:  {np.max(cosines):.6f}")
print(f"  Mean raw norm:          {np.mean(raw_norms):.6f}")
print(f"  Std raw norm:           {np.std(raw_norms):.6f}")
print()

# Expected based on training
expected_val_cosine = checkpoint.get('val_cosine', 0.53) if isinstance(checkpoint, dict) else 0.53
print(f"Expected val_cosine (from training): {expected_val_cosine:.4f}")
print(f"Actual mean cosine (this test):      {np.mean(cosines):.4f}")
print()

if abs(np.mean(cosines) - expected_val_cosine) > 0.1:
    print("🚨 PROBLEM DETECTED!")
    print(f"   Cosine similarity is {abs(np.mean(cosines) - expected_val_cosine):.4f} off from expected!")
    print(f"   This suggests the 5CAT test may be using wrong data or wrong computation.")
else:
    print("✅ Cosine similarity matches expected value!")
    print("   The model output is correct. 5CAT test should work.")
