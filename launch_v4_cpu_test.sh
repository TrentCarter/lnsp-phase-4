#!/bin/bash
# V4 Training - CPU Test (verify pipeline works)
# Just 3 epochs to test the complete pipeline

set -e

echo "=== V4 TWO-TOWER TRAINING (CPU TEST - 3 EPOCHS) ==="
echo "Started: $(date)"
echo ""
echo "🎯 Purpose: Verify training pipeline completes successfully"
echo "   Device: CPU (slow but stable)"
echo "   Epochs: 3 (quick test)"
echo ""

# Minimal test profile
BATCH_SIZE=8
ACCUM_STEPS=2
EPOCHS=3
LR=5e-5
LR_MIN=1e-6
WD=0.01
TAU=0.05
MARGIN=0.03
MEMORY_BANK_SIZE=10000
MINE_SCHEDULE="0-1:none;2-3:4@0.82-0.92"
FILTER_THRESHOLD=0.98

# No async mining for CPU test (keep it simple)
GRAD_CLIP=1.0

PAIRS="artifacts/twotower/pairs_v4_synth.npz"
BANK="artifacts/wikipedia_500k_corrected_vectors.npz"
OUT_DIR="runs/twotower_v4_cpu_test"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/twotower_v4_cpu_test_$TIMESTAMP.log"

mkdir -p logs
mkdir -p "$OUT_DIR"

echo "Configuration:"
echo "  Batch size: $BATCH_SIZE × $ACCUM_STEPS = $((BATCH_SIZE * ACCUM_STEPS)) effective"
echo "  Device: CPU"
echo "  Training pairs: 35,901"
echo "  Epochs: $EPOCHS"
echo "  Async mining: DISABLED (synchronous for stability)"
echo "  Log: $LOG_FILE"
echo ""

PYTHONUNBUFFERED=1 \
./.venv/bin/python3 tools/train_twotower_v4.py \
  --pairs "$PAIRS" \
  --bank "$BANK" \
  --out "$OUT_DIR" \
  --bs $BATCH_SIZE \
  --accum $ACCUM_STEPS \
  --epochs $EPOCHS \
  --lr $LR \
  --lr-min $LR_MIN \
  --wd $WD \
  --tau $TAU \
  --margin $MARGIN \
  --memory-bank-size $MEMORY_BANK_SIZE \
  --mine-schedule "$MINE_SCHEDULE" \
  --filter-threshold $FILTER_THRESHOLD \
  --device cpu \
  --grad-clip $GRAD_CLIP \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "==========================================="
  echo "✅ CPU TEST COMPLETED SUCCESSFULLY!"
  echo "==========================================="
  echo ""
  echo "Saved checkpoints:"
  ls -lh "$OUT_DIR/checkpoints/"
  echo ""
else
  echo ""
  echo "==========================================="
  echo "❌ CPU TEST FAILED (Exit code: $EXIT_CODE)"
  echo "==========================================="
  echo ""
  echo "Last 50 lines of log:"
  tail -50 "$LOG_FILE"
  echo ""
  exit $EXIT_CODE
fi
