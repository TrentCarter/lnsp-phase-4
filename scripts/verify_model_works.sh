#!/bin/bash
#
# Quick Model Verification Script
# ================================
#
# Checks if a model produces reasonable predictions (not random noise)
#
# Usage:
#   ./scripts/verify_model_works.sh <model_path>
#
# Example:
#   ./scripts/verify_model_works.sh artifacts/lvm/models/transformer_584k_stable/best_model.pt

set -e

MODEL_PATH="${1:-artifacts/lvm/models/transformer_584k_stable/best_model.pt}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    exit 1
fi

echo "============================================"
echo "Model Verification Test"
echo "============================================"
echo "Model: $MODEL_PATH"
echo ""

./.venv/bin/python3 << EOF
import torch
import numpy as np
import sys

sys.path.insert(0, 'app/lvm')
from model import TransformerModel, AMNModel, GRUModel, LSTMModel

# Load checkpoint
checkpoint = torch.load('$MODEL_PATH', weights_only=False)
model_type = checkpoint.get('model_type', 'transformer')
config = checkpoint.get('model_config', {})

print(f"Model type: {model_type}")
print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Checkpoint val_cosine: {checkpoint.get('val_cosine', 'N/A'):.4f}")
print()

# Load model
if model_type == 'transformer':
    model = TransformerModel(**config)
elif model_type == 'amn':
    model = AMNModel(**config)
elif model_type == 'gru':
    model = GRUModel(**config)
elif model_type == 'lstm':
    model = LSTMModel(**config)
else:
    print(f"❌ Unknown model type: {model_type}")
    sys.exit(1)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load validation data
data = np.load('artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz')
contexts = torch.FloatTensor(data['contexts'][:500])
targets = torch.FloatTensor(data['targets'][:500])

# Test predictions
print("🧪 Testing on 500 validation samples...")
with torch.no_grad():
    predictions = model(contexts)

    # Compute cosines
    pred_norm = predictions / (predictions.norm(dim=1, keepdim=True) + 1e-8)
    tgt_norm = targets / (targets.norm(dim=1, keepdim=True) + 1e-8)
    cosines = (pred_norm * tgt_norm).sum(dim=1).cpu().numpy()

mean_cos = np.mean(cosines)
std_cos = np.std(cosines)
min_cos = np.min(cosines)
max_cos = np.max(cosines)

print()
print("=" * 50)
print("RESULTS")
print("=" * 50)
print(f"Mean cosine:  {mean_cos:.4f}")
print(f"Std cosine:   {std_cos:.4f}")
print(f"Range:        [{min_cos:.4f}, {max_cos:.4f}]")
print(f"Checkpoint:   {checkpoint.get('val_cosine', 0):.4f}")
print()

# Verdict
if mean_cos < 0.01 and abs(mean_cos) < 0.05:
    print("❌ FAIL: Model produces RANDOM NOISE")
    print("   Mean cosine near zero - model is broken!")
    print("   Do NOT deploy this model.")
    sys.exit(2)
elif mean_cos < 0.40:
    print("⚠️  WARNING: Low prediction quality")
    print(f"   Mean cosine {mean_cos:.4f} is below expected 0.50+")
    print("   Model may have issues.")
    sys.exit(1)
elif abs(mean_cos - checkpoint.get('val_cosine', 0)) > 0.10:
    print("⚠️  WARNING: Checkpoint metrics don't match reality")
    print(f"   Actual: {mean_cos:.4f}, Checkpoint: {checkpoint.get('val_cosine', 0):.4f}")
    print(f"   Difference: {abs(mean_cos - checkpoint.get('val_cosine', 0)):.4f}")
    print("   Checkpoint may be corrupted, but model works.")
    sys.exit(1)
else:
    print("✅ PASS: Model produces reasonable predictions!")
    print(f"   Mean cosine {mean_cos:.4f} is in expected range.")
    print("   Model is ready for deployment.")
    sys.exit(0)
EOF

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Model verification PASSED"
elif [ $EXIT_CODE -eq 1 ]; then
    echo "⚠️  Model verification WARNING"
elif [ $EXIT_CODE -eq 2 ]; then
    echo "❌ Model verification FAILED"
fi

exit $EXIT_CODE
