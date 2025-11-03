#!/usr/bin/env bash
set -euo pipefail

# P6: NEXT Token Architecture
# Strategy: Remove identity path by predicting target_next instead of target
#
# Key Difference from P1-P5.1:
# - P1-P5.1: ctx[0..4] → target (can copy ctx[4])
# - P6:      ctx[0..4] → target_next (CANNOT copy ctx[4]!)
#
# This script uses P6 data format with optional P5.1 enhancements.
# The main benefit comes from the architectural change, but P5.1 can still help.
#
# Usage:
#   ./scripts/train_transformer_p6_next_token.sh [DEVICE] [USE_P5.1_ENHANCEMENTS]
#
# Examples:
#   ./scripts/train_transformer_p6_next_token.sh mps false   # Pure P6, no enhancements
#   ./scripts/train_transformer_p6_next_token.sh mps true    # P6 + P5.1 enhancements

PY=${PY:-./.venv/bin/python}
TR=app/lvm/train_unified.py
TEST=tools/tests/test_5to1_alignment.py

# P6 data files
TRAIN_NPZ=artifacts/lvm/training_sequences_ctx5_p6_next_token.npz
VAL_NPZ=artifacts/lvm/validation_sequences_ctx5_p6_next_token.npz
OOD_NPZ=artifacts/lvm/ood_sequences_ctx5_p6_next_token.npz
ART_NPZ=artifacts/wikipedia_584k_fresh.npz

DEVICE=${1:-mps}
USE_ENHANCEMENTS=${2:-false}  # Set to true to enable P5.1 enhancements

echo "============================================="
echo "P6: NEXT Token Architecture"
echo "============================================="
echo "Training Data: $TRAIN_NPZ"
echo "Val Data: $VAL_NPZ"
echo "OOD Data: $OOD_NPZ"
echo "Device: $DEVICE"
echo "P5.1 Enhancements: $USE_ENHANCEMENTS"
echo ""

BASE_DIR="artifacts/lvm/models/transformer_p6_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_DIR"

# Base training arguments (always used)
TRAIN_ARGS=(
  --model-type transformer
  --data "$TRAIN_NPZ"
  --epochs 10
  --batch-size 32
  --device "$DEVICE"
  --fivecat-every-epoch 1
  --fivecat-max-samples 2000
  --output-dir "$BASE_DIR"
)

# P5.1 enhancements (optional, add if requested)
if [ "$USE_ENHANCEMENTS" = "true" ]; then
    echo "✨ Enabling P5.1 enhancements (positional ramp, attention bias, noise)"
    TRAIN_ARGS+=(
        --positional-scalar 0.10
        --positional-ramp-epochs 3
        --attn-last-bias-max 0.6
        --attn-last-bias-warmup-epochs 4
        --last-slot-noise-p 0.15
        --last-slot-noise-sigma 0.03
        --last-slot-swap-p 0.05
        --lambda-dir 0.001
        --dir-gamma 5.0
        --dir-margin 0.02
        --dir-warmup-epochs 6
    )
fi

echo "============================================="
echo "P6 Training (10 epochs)"
echo "============================================="
echo ""

$PY $TR "${TRAIN_ARGS[@]}"

TRAIN_STATUS=$?

echo ""
echo "[P6] Full 5CAT evaluation..."
$PY $TEST --model "$BASE_DIR/best_model.pt" \
  --val-npz "$VAL_NPZ" --ood-npz "$OOD_NPZ" --articles-npz "$ART_NPZ" \
  --device "$DEVICE" --max-samples 5000 | tee "$BASE_DIR/5cat_results.json"

# Check training status
if [ $TRAIN_STATUS -ne 0 ]; then
    echo ""
    echo "❌ P6 Training FAILED"
    echo "   Check logs above for details"
    exit 1
fi

echo ""
echo "✅ P6 Training Complete!"
echo "   Model: $BASE_DIR/best_model.pt"
echo ""
echo "📊 Expected Results:"
echo "   - Margin should be POSITIVE (≥ +0.05)"
echo "   - No backward bias (identity path removed by design)"
echo "   - R@5 should be high (≥ 70%)"
echo ""
echo "📋 Next Steps:"
echo "   1. Check 5CAT results above"
echo "   2. If margin is positive → SUCCESS! P6 broke the copy-last curse"
echo "   3. If margin is still negative → Check data quality or model architecture"
echo ""

echo "[P6] DONE → $BASE_DIR"
