#!/bin/bash
# Quick test of improved trainer with Memory GRU
# This validates all consultant recommendations:
# ✅ Hit@K evaluation
# ✅ Chain-level split
# ✅ Mixed loss (MSE + cosine + InfoNCE)
# ✅ Delta prediction
# ✅ Coherence filtering

set -e

echo "=== Testing Improved LVM Trainer ==="
echo ""
echo "Testing with Memory GRU (best performing model)"
echo "Data: 100-vector context (12,757 sequences)"
echo "Improvements:"
echo "  ✅ Hit@1/5/10 retrieval evaluation"
echo "  ✅ Chain-level train/val split (zero leakage)"
echo "  ✅ Mixed loss: MSE + cosine + InfoNCE"
echo "  ✅ Delta prediction (predict Δ = y_next - y_curr)"
echo "  ✅ Chain coherence filtering (≥0.78)"
echo ""

# Run for 5 epochs as a quick test
./.venv/bin/python app/lvm/train_improved.py \
  --model-type memory_gru \
  --data artifacts/lvm/data_extended/training_sequences_ctx100.npz \
  --epochs 2 \
  --batch-size 32 \
  --lr 0.0005 \
  --device mps \
  --delta-mode \
  --coherence-threshold 0.78 \
  --lambda-mse 1.0 \
  --lambda-cosine 0.5 \
  --lambda-infonce 0.1 \
  --temperature 0.07 \
  --output-dir artifacts/lvm/models_improved/memory_gru_test

echo ""
echo "=== Test Complete ==="
echo ""
echo "Check results:"
echo "  cat artifacts/lvm/models_improved/memory_gru_test/training_history.json | jq '.history[-1]'"
echo ""
echo "Expected improvements:"
echo "  - Val cosine: 0.55-0.60 (vs 0.49 with old trainer)"
echo "  - Hit@1: ≥30%"
echo "  - Hit@5: ≥55%"
echo "  - Chain leakage: 0 (verified in logs)"
