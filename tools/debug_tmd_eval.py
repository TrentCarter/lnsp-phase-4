#!/usr/bin/env python3
"""
Debug TMD Evaluation Issues

Investigates why TMD re-ranking shows 0% Hit@5 instead of expected 75.65%.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Load validation data
val_data = np.load('artifacts/lvm/data_phase3/validation_sequences_ctx100.npz', allow_pickle=True)
val_contexts = torch.from_numpy(val_data['context_sequences']).float()
val_targets = torch.from_numpy(val_data['target_vectors']).float()

print(f"Validation data loaded:")
print(f"  Contexts: {val_contexts.shape}")
print(f"  Targets: {val_targets.shape}")
print()

# Load vector bank
bank_data = np.load('artifacts/wikipedia_637k_phase3_vectors.npz', allow_pickle=True)
vector_bank = torch.from_numpy(bank_data['vectors']).float()

print(f"Vector bank loaded:")
print(f"  Bank size: {vector_bank.shape}")
print()

# Load model
checkpoint = torch.load('artifacts/lvm/models_phase3/run_1000ctx_pilot/best_val_hit5.pt', map_location='cpu')
print(f"Model checkpoint loaded:")
print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"  Best Hit@5: {checkpoint.get('best_val_hit5', checkpoint.get('val_results', {}).get('hit@5', 'N/A'))}")
print()

# Compute target_indices on-the-fly (same method as eval script)
print("Computing target_indices by matching targets to bank...")
target_norm = F.normalize(val_targets, dim=1)
bank_norm = F.normalize(vector_bank, dim=1)

similarities = torch.mm(target_norm, bank_norm.t())  # [127, 637997]
computed_indices = similarities.argmax(dim=1).numpy()
max_similarities = similarities.max(dim=1).values.numpy()

print(f"Target indices computed: {len(computed_indices)}")
print(f"Max similarity stats:")
print(f"  Min: {max_similarities.min():.4f}")
print(f"  Mean: {max_similarities.mean():.4f}")
print(f"  Max: {max_similarities.max():.4f}")
print()

# Check if targets are exact matches in bank (similarity = 1.0)
exact_matches = (max_similarities > 0.999).sum()
print(f"Exact matches (similarity > 0.999): {exact_matches}/{len(computed_indices)}")
print()

# Load model and make predictions on first 5 samples
from app.lvm.models import MemoryAugmentedGRU

config = checkpoint.get('config', {})
model = MemoryAugmentedGRU(
    d_model=config.get('input_dim', 768),
    hidden_dim=config.get('hidden_dim', 512),
    num_layers=config.get('num_layers', 4),
    memory_slots=config.get('memory_slots', 2048)
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Testing first 5 samples:")
print("=" * 80)

with torch.no_grad():
    for i in range(min(5, len(val_contexts))):
        context = val_contexts[i:i+1]  # [1, 1000, 768]
        target = val_targets[i:i+1]  # [1, 768]
        target_idx = computed_indices[i]

        # Model prediction
        pred = model(context)  # [1, 768]
        pred_norm = F.normalize(pred, dim=1)

        # Find top-10 candidates
        sims = torch.mm(pred_norm, bank_norm.t())  # [1, 637997]
        topk_scores, topk_indices = sims.topk(10, dim=1)
        topk_indices = topk_indices[0].numpy()
        topk_scores = topk_scores[0].numpy()

        print(f"\nSample {i}:")
        print(f"  Target index: {target_idx}")
        print(f"  Target similarity to bank: {max_similarities[i]:.4f}")
        print(f"  Top-10 candidates: {topk_indices.tolist()}")
        print(f"  Top-10 scores: {[f'{s:.4f}' for s in topk_scores]}")
        print(f"  Target in top-10? {target_idx in topk_indices}")

        # Check if target_idx is anywhere in top-100
        top100_indices = sims.topk(100, dim=1).indices[0].numpy()
        if target_idx in top100_indices:
            rank = np.where(top100_indices == target_idx)[0][0] + 1
            print(f"  Target rank: {rank} (in top-100)")
        else:
            print(f"  Target rank: >100 (NOT in top-100)")

print("\n" + "=" * 80)
print("\nDiagnostic Summary:")
print(f"  Validation samples: {len(val_contexts)}")
print(f"  Bank size: {len(vector_bank)}")
print(f"  Exact target matches in bank: {exact_matches}/{len(computed_indices)}")
print(f"  If exact matches = 0, targets may not be from this bank!")
