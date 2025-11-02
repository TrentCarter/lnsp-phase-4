#!/bin/bash
# P1: Clean MSE-only baseline (3-5 epochs) to verify pipeline
# NO positional encoding, NO directional losses, NO context drop
# Just pure MSE to confirm we can still learn

set -e

# Critical for macOS training
export KMP_DUPLICATE_LIB_OK=TRUE

# Training config
MODEL_TYPE="transformer"
EPOCHS=5
BATCH_SIZE=64
LR=0.0005
DEVICE="mps"

# Data paths (train_unified.py does internal 90/10 split)
DATA_NPZ="artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz"

# Output directory
OUTPUT_DIR="artifacts/lvm/models/transformer_baseline_p1"
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "P1: Clean MSE-Only Baseline"
echo "========================================="
echo "Model: $MODEL_TYPE"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Device: $DEVICE"
echo "Data: $DATA_NPZ"
echo ""
echo "Expectations by epoch 3:"
echo "  - val_cos ≥ 0.50"
echo "  - train_cos ≥ 0.48"
echo "  - No collapse"
echo "========================================="
echo ""

./.venv/bin/python app/lvm/train_unified.py \
  --model-type "$MODEL_TYPE" \
  --data "$DATA_NPZ" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --device "$DEVICE" \
  --output-dir "$OUTPUT_DIR" \
  --lambda-dir 0.0 \
  --lambda-ac 0.0 \
  --lambda-fut 0.0 \
  --context-drop-p 0.0 \
  2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "========================================="
echo "P1 Baseline Complete"
echo "========================================="
echo "Check training history:"
echo "  cat $OUTPUT_DIR/training_history.json | jq '.history[] | {epoch, val_cosine, train_cosine}'"
echo ""
echo "Expected: val_cos ≥ 0.50 by epoch 3"
echo "If this passes, proceed to P2 (residual architecture)"
echo "========================================="
