#!/bin/bash
#
# Train AMN with CLEAN Representative Splits
# ===========================================
#
# Data splits:
# - Training:   Articles 1-1499, 2000-3999, 4500-7671 (438k sequences, coherence 0.4683)
# - Validation: Articles 4000-4499 (18k sequences, coherence 0.4704)
# - OOD Test:   Articles 1500-1999 (10k sequences, coherence 0.4637)
#
# All regions have REPRESENTATIVE coherence (~0.47)!
# High-coherence tail (7672-8470) REMOVED from dataset.
#
# Expected results:
# - Val cosine: ~0.50-0.54 (REAL generalization!)
# - OOD cosine: ~0.48-0.52 (should match val!)
# - Delta < 0.10 (proves true generalization!)

set -e

export KMP_DUPLICATE_LIB_OK=TRUE  # macOS OpenMP fix

# Train using the clean splits
# (Script will load separate train/val files)
PYTHONPATH=. ./.venv/bin/python -c "
import sys
import numpy as np
import torch
from pathlib import Path

# Add project to path
sys.path.insert(0, 'app/lvm')
from models import create_model

def cosine_similarity(pred, target):
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    return (pred_norm * target_norm).sum(dim=1).mean()

print('=' * 80)
print('Training AMN with CLEAN Representative Splits')
print('=' * 80)
print()

# Load training data
print('📥 Loading training data...')
train_data = np.load('artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz', allow_pickle=True)
train_contexts = torch.FloatTensor(train_data['context_sequences'])
train_targets = torch.FloatTensor(train_data['target_vectors'])
print(f'   Train: {len(train_contexts):,} sequences')
print()

# Load validation data
print('📥 Loading validation data...')
val_data = np.load('artifacts/lvm/validation_sequences_ctx5_articles4000-4499.npz', allow_pickle=True)
val_contexts = torch.FloatTensor(val_data['context_sequences'])
val_targets = torch.FloatTensor(val_data['target_vectors'])
print(f'   Val: {len(val_contexts):,} sequences')
print()

# Create data loaders
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(train_contexts, train_targets)
val_dataset = TensorDataset(val_contexts, val_targets)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

# Create model
print('🏗️  Creating AMN model...')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = create_model('amn', input_dim=768, d_model=256, hidden_dim=512).to(device)
params = sum(p.numel() for p in model.parameters())
print(f'   Parameters: {params:,}')
print(f'   Device: {device}')
print()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Training loop
print('🏃 Starting training...')
print()

best_val_loss = float('inf')
output_dir = Path('artifacts/lvm/models')

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_dir = output_dir / f'amn_clean_splits_{timestamp}'
model_dir.mkdir(parents=True, exist_ok=True)

for epoch in range(1, 21):
    print(f'Epoch {epoch}/20')

    # Train
    model.train()
    total_loss = 0.0
    total_cosine = 0.0
    n_batches = 0

    for batch_idx, (ctx, tgt) in enumerate(train_loader):
        ctx, tgt = ctx.to(device), tgt.to(device)

        optimizer.zero_grad()
        pred = model(ctx)

        loss = torch.nn.functional.mse_loss(pred, tgt)
        cos = cosine_similarity(pred, tgt)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cosine += cos.item()
        n_batches += 1

        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f} | Cosine: {cos.item():.4f}')

    train_loss = total_loss / n_batches
    train_cosine = total_cosine / n_batches

    # Validate
    model.eval()
    val_loss = 0.0
    val_cosine = 0.0
    n_val = 0

    with torch.no_grad():
        for ctx, tgt in val_loader:
            ctx, tgt = ctx.to(device), tgt.to(device)
            pred = model(ctx)

            loss = torch.nn.functional.mse_loss(pred, tgt)
            cos = cosine_similarity(pred, tgt)

            val_loss += loss.item()
            val_cosine += cos.item()
            n_val += 1

    val_loss /= n_val
    val_cosine /= n_val

    print(f'  Train Loss: {train_loss:.6f} | Train Cosine: {train_cosine:.4f}')
    print(f'  Val Loss: {val_loss:.6f} | Val Cosine: {val_cosine:.4f}')

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'amn',
            'val_cosine': val_cosine,
            'val_loss': val_loss,
            'epoch': epoch,
        }, model_dir / 'best_model.pt')
        print(f'  ✓ Saved best model (val_loss: {val_loss:.6f})')

    print()

    # Learning rate schedule
    scheduler.step(val_loss)

print('=' * 80)
print('✅ Training Complete!')
print('=' * 80)
print(f'Best val loss: {best_val_loss:.6f}')
print(f'Model saved to: {model_dir}')
print()
print('Next: Evaluate on OOD test set')
print(f'./.venv/bin/python tools/eval_model_ood.py \\\\')
print(f'  --model {model_dir}/best_model.pt \\\\')
print(f'  --ood-data artifacts/lvm/wikipedia_ood_test_ctx5_TRULY_FIXED.npz \\\\')
print(f'  --device mps')
print()
print('Expected: OOD cosine ~0.48-0.52 (should match val!)')
"

echo ""
echo "Training complete!"
