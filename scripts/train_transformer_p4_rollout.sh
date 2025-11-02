#!/bin/bash
# P4: Rollout Loss + Adaptive Guards
# Changes the learning signal - makes copying fail over 2-3 steps

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
OUTPUT_DIR="artifacts/lvm/models/transformer_p4_rollout"
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "P4: Rollout Loss + Adaptive Guards"
echo "========================================="
echo "Model: $MODEL_TYPE (original architecture)"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Device: $DEVICE"
echo ""
echo "Training Curriculum:"
echo "  Epochs 1-3: Pure MSE warm-up"
echo "  Epochs 4-6: + Rollout loss (H=3, λ=0.05)"
echo "  Epochs 7+:  + Rollout loss (λ=0.10)"
echo "  Epochs 6+:  + Adaptive guards (λ_dir=0.002)"
echo "  Epochs 10+: + Future ranking (λ_fut=0.002)"
echo ""
echo "Key Innovation: Rollout Loss"
echo "  Predict 3 steps ahead autoregressively"
echo "  Penalizes flat/copying trajectories"
echo "  Makes copy-last fail over multiple steps"
echo ""
echo "Success Criteria:"
echo "  Epoch 3: val_cos ≥ 0.50 (warm-up)"
echo "  Epoch 8: margin ≥ 0.0 (crossing zero)"
echo "  Epoch 20: val_cos ≥ 0.54, margin ≥ +0.10"
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
  --rollout-h 3 \
  --lambda-roll 0.05 \
  --rollout-start-epoch 4 \
  --guards-start-epoch 6 \
  --lambda-dir 0.002 \
  --lambda-fut 0.002 \
  --lambda-ac 0.0 \
  --margin-dir 0.01 \
  --margin-fut 0.008 \
  --context-drop-p 0.05 \
  --adaptive-dir \
  2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "========================================="
echo "P4 Training Complete"
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
