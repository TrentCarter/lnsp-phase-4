#!/bin/bash
#
# P2 SMOKE TEST: AMN on Filtered 277k Dataset
# ===========================================
#
# Strategy: Train from scratch on filtered data, quick 5-epoch probe
# Goal: Verify quality lift translates to improved OOD performance
# Note: train_unified.py doesn't support warm-start, but filtered
#       dataset quality should be sufficient
#
# Gates:
#   Ep1: val_cosine >= 0.50
#   Ep3: OOD_smoke >= 0.50
#   Ep5: OOD_smoke >= 0.58
#

set -e

# Configuration
DATA_FILE="artifacts/lvm/training_sequences_ctx5_filtered.npz"
MODEL_DIR="artifacts/lvm/models/amn_filtered_smoke_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${MODEL_DIR}/training.log"

# Hyperparameters (per user spec)
EPOCHS=5
BATCH_SIZE=64
LEARNING_RATE=0.0005
DEVICE="mps"
WARMUP_STEPS=1000

# macOS OpenMP fix (critical!)
export KMP_DUPLICATE_LIB_OK=TRUE

echo "============================================================"
echo "P2 SMOKE TEST: AMN Filtered 277k"
echo "============================================================"
echo "Data:       $DATA_FILE"
echo "Output:     $MODEL_DIR"
echo "Epochs:     $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "LR:         $LEARNING_RATE"
echo "Device:     $DEVICE"
echo "Warmup:     $WARMUP_STEPS steps"
echo "============================================================"
echo ""

# Verify data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$MODEL_DIR"

echo "🚀 Starting P2 smoke test..."
echo ""

# Run training from scratch (train_unified.py doesn't support resume)
# Note: Filtered dataset quality should be sufficient without warm-start
./.venv/bin/python app/lvm/train_unified.py \
    --model-type amn \
    --data "$DATA_FILE" \
    --output-dir "$MODEL_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --device $DEVICE \
    --lambda-mse 1.0 \
    --lambda-info 0.0 \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ P2 SMOKE TEST COMPLETE"
    echo "============================================================"
    echo "Model saved to: $MODEL_DIR"
    echo "Log saved to:   $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "1. Check training log for epoch gates:"
    echo "   - Ep1: val_cosine >= 0.50"
    echo "   - Ep5: val_cosine >= 0.53 (target)"
    echo ""
    echo "2. Run OOD evaluation:"
    echo "   ./.venv/bin/python tools/eval_model_ood.py \\"
    echo "     --model $MODEL_DIR/best_model.pt \\"
    echo "     --model-type amn \\"
    echo "     --device $DEVICE"
    echo ""
    echo "3. Check OOD gate:"
    echo "   - OOD cosine >= 0.58 → GREEN (proceed to full training)"
    echo "   - OOD cosine 0.50-0.57 → AMBER (consider ctx=7)"
    echo "   - OOD cosine < 0.50 → RED (stick with 584k)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ P2 SMOKE TEST FAILED"
    echo "============================================================"
    echo "Exit code: $EXIT_CODE"
    echo "Check log: $LOG_FILE"
    exit $EXIT_CODE
fi
