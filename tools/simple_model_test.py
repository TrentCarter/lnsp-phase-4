#!/usr/bin/env python3
"""
Simple Model Sanity Check

Tests if Phase-3 model makes reasonable predictions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from app.lvm.models import MemoryAugmentedGRU

# Load validation data
val_data = np.load('artifacts/lvm/data_phase3/validation_phase3_exact.npz', allow_pickle=True)
contexts = torch.from_numpy(val_data['context_sequences']).float()[:10]  # First 10
targets = torch.from_numpy(val_data['target_vectors']).float()[:10]

print(f"Loaded {len(contexts)} validation samples")

# Load model
checkpoint = torch.load('artifacts/lvm/models_phase3/run_1000ctx_pilot/best_val_hit5.pt', map_location='cpu')
config = checkpoint.get('config', {})

# Try to get model config from checkpoint
if 'config' in checkpoint:
    d_model = config.get('input_dim', 768)
    hidden_dim = config.get('hidden_dim', 512)
    num_layers = config.get('num_layers', 4)
    memory_slots = config.get('memory_slots', 2048)
else:
    # Use args instead
    args = checkpoint.get('args', {})
    d_model = 768  # Default
    hidden_dim = 512  # From training history
    num_layers = 4  # From training history
    memory_slots = 2048  # From training history

model = MemoryAugmentedGRU(
    d_model=d_model,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    memory_slots=memory_slots
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model: {d_model}D → {hidden_dim}D hidden, {num_layers} layers, {memory_slots} slots")

# Make predictions
with torch.no_grad():
    predictions = model(contexts)

# Calculate cosine similarities
pred_norm = F.normalize(predictions, dim=1)
target_norm = F.normalize(targets, dim=1)

cosine_sims = (pred_norm * target_norm).sum(dim=1).numpy()

print(f"\nCosine similarities (prediction vs target):")
print(f"  Min:  {cosine_sims.min():.4f}")
print(f"  Mean: {cosine_sims.mean():.4f}")
print(f"  Max:  {cosine_sims.max():.4f}")
print(f"\nExpected during training: ~0.48 val_cosine (from checkpoint)")
print(f"Actual: {cosine_sims.mean():.4f}")

if cosine_sims.mean() < 0.3:
    print("\n⚠️ MODEL NOT WORKING! Predictions are random/broken!")
elif cosine_sims.mean() < 0.45:
    print("\n⚠️ Model working but poorly (lower than training)")
else:
    print("\n✅ Model predictions look reasonable")
