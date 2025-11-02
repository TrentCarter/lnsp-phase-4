#!/bin/bash
# P3: P1 Baseline Architecture + Tiny Late Guards
# NO residual wrapper - just direct prediction with minimal guard nudges

set -e

# Critical for macOS training
export KMP_DUPLICATE_LIB_OK=TRUE

# Training config
MODEL_TYPE="transformer"
EPOCHS=20
BATCH_SIZE=64
LR=0.0005
DEVICE="mps"

# Data paths
DATA_NPZ="artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz"

# Output directory
OUTPUT_DIR="artifacts/lvm/models/transformer_p3_tiny_guards"
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "P3: P1 Architecture + Tiny Late Guards"
echo "========================================="
echo "Model: $MODEL_TYPE (original architecture, NO residual)"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Device: $DEVICE"
echo ""
echo "Warm-up (epochs 1-5): Pure MSE"
echo ""
echo "Guards (start epoch 6):"
echo "  λ_dir = 0.002 (directional: next > prev)"
echo "  λ_fut = 0.002 (future: next > +2/+3)"
echo "  λ_ac  = 0.0   (anti-copy: disabled)"
echo "  context_drop = 0.05 (from epoch 6)"
echo ""
echo "Success Criteria:"
echo "  Epoch 3: val_cos ≥ 0.50 (same as P1)"
echo "  Epoch 10: margin > 0 (positive)"
echo "  Final: val_cos ≥ 0.54, margin ≥ +0.08"
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
  --guards-start-epoch 6 \
  --lambda-dir 0.002 \
  --lambda-fut 0.002 \
  --lambda-ac 0.0 \
  --margin-dir 0.01 \
  --margin-fut 0.008 \
  --context-drop-p 0.05 \
  2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "========================================="
echo "P3 Training Complete"
echo "========================================="
echo "Check training history:"
echo "  cat $OUTPUT_DIR/training_history.json | jq '.history[] | {epoch, val_cosine, train_cosine}'"
echo ""
echo "Next: Run 5CAT validation"
echo "  ./.venv/bin/python tools/tests/test_5to1_alignment.py \\"
echo "    --model $OUTPUT_DIR/best_model.pt \\"
echo "    --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \\"
echo "    --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \\"
echo "    --articles-npz artifacts/wikipedia_584k_fresh.npz \\"
echo "    --device mps --max-samples 5000"
echo "========================================="
