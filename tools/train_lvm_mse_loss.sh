#!/bin/bash
# Train LVM-T with MSE Loss (FIXED)
# This script uses the corrected loss function where MSE is PRIMARY, not InfoNCE

set -e

echo "========================================"
echo "Training LVM-T with MSE Loss (CORRECTED)"
echo "========================================"
echo ""
echo "Key Changes from Previous Training:"
echo "  - MSE loss is PRIMARY (weight=1.0)"
echo "  - InfoNCE loss is DISABLED (weight=0.0)"
echo "  - Expected improvement: 0.35 → 0.45+ cosine"
echo ""

OUTPUT_DIR="artifacts/lvm/models/transformer_mse_corrected"
LOG_FILE="${OUTPUT_DIR}/training.log"

mkdir -p "${OUTPUT_DIR}"

./.venv/bin/python app/lvm/train_transformer.py \
  --data artifacts/lvm/training_sequences_ctx5.npz \
  --epochs 20 \
  --batch-size 32 \
  --lr 0.0005 \
  --d-model 512 \
  --nhead 8 \
  --num-layers 4 \
  --device mps \
  --lambda-mse 1.0 \
  --lambda-moment 0.0 \
  --lambda-variance 0.0 \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Model saved to: ${OUTPUT_DIR}"
echo "Training log: ${LOG_FILE}"
echo ""
echo "Next step: Run evaluation to compare with baseline"
