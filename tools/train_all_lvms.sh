#!/bin/bash
# Train All 4 LVM Architectures
# This script trains all models with consistent hyperparameters for fair comparison

set -e

echo "=================================="
echo "Training All LVM Architectures"
echo "=================================="
echo ""
echo "Models to train:"
echo "  1. AMN (Attention Mixture Network) - RECOMMENDED for LNSP"
echo "  2. LSTM Baseline"
echo "  3. GRU Stack"
echo "  4. Transformer"
echo ""
echo "Training configuration:"
echo "  Loss: MSE (weight=1.0)"
echo "  Epochs: 20"
echo "  Batch size: 32"
echo "  Learning rate: 0.0005"
echo "  Device: MPS (Apple Silicon)"
echo ""

# 1. Train AMN (fastest, most efficient for LNSP)
echo "=================================="
echo "1/4: Training AMN (~15 min)"
echo "=================================="
./.venv/bin/python app/lvm/train_unified.py \
  --model-type amn \
  --epochs 20 \
  --batch-size 32 \
  --lr 0.0005 \
  --lambda-mse 1.0 \
  --device mps

echo ""
echo "✓ AMN training complete!"
echo ""

# 2. Train LSTM
echo "=================================="
echo "2/4: Training LSTM (~15 min)"
echo "=================================="
./.venv/bin/python app/lvm/train_unified.py \
  --model-type lstm \
  --epochs 20 \
  --batch-size 32 \
  --lr 0.0005 \
  --lambda-mse 1.0 \
  --device mps

echo ""
echo "✓ LSTM training complete!"
echo ""

# 3. Train GRU
echo "=================================="
echo "3/4: Training GRU (~15 min)"
echo "=================================="
./.venv/bin/python app/lvm/train_unified.py \
  --model-type gru \
  --epochs 20 \
  --batch-size 32 \
  --lr 0.0005 \
  --lambda-mse 1.0 \
  --device mps

echo ""
echo "✓ GRU training complete!"
echo ""

# 4. Train Transformer
echo "=================================="
echo "4/4: Training Transformer (~20 min)"
echo "=================================="
./.venv/bin/python app/lvm/train_unified.py \
  --model-type transformer \
  --epochs 20 \
  --batch-size 32 \
  --lr 0.0005 \
  --lambda-mse 1.0 \
  --device mps

echo ""
echo "✓ Transformer training complete!"
echo ""

echo "=================================="
echo "All Models Trained Successfully!"
echo "=================================="
echo ""
echo "Models saved to:"
echo "  artifacts/lvm/models/amn_*/"
echo "  artifacts/lvm/models/lstm_*/"
echo "  artifacts/lvm/models/gru_*/"
echo "  artifacts/lvm/models/transformer_*/"
echo ""
echo "Next step: Compare results with tools/compare_lvm_models.py"
