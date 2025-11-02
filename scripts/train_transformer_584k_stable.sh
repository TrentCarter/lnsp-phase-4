#!/bin/bash
#
# Train Transformer_584K using EXACT settings that worked for 340k
# Based on working production model parameters
#

set -e

# Use exact same parameters as working 340k Transformer
./.venv/bin/python app/lvm/train_unified.py \
  --model-type transformer \
  --data artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.0005 \
  --device mps \
  --output-dir artifacts/lvm/models/transformer_584k_stable \
  --lambda-mse 1.0 \
  --lambda-info 0.0 \
  --lambda-moment 0.0 \
  --lambda-variance 0.0 \
  --tau 0.07 \
  --lambda-mmd 0.0 \
  --mmd-anchors 0 \
  --lambda-stat 0.0 \
  --cycle-pct 0.0 \
  --cycle-lambda 0.0 \
  --cycle-steps 1 \
  --cycle-timeout 30.0

echo ""
echo "=========================================="
echo "✅ Training Complete!"
echo "=========================================="
echo "Output: artifacts/lvm/models/transformer_584k_stable/"
echo ""
echo "Next: Verify model with diagnostic:"
echo "  python tools/diagnose_model_output.py"
