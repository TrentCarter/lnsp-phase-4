#!/bin/bash
# V4 Training - MPS (Apple GPU) - EXPERIMENTAL
# NOTE: MPS is faster but less stable than CPU

set -e

echo "=== V4 TWO-TOWER TRAINING (MPS/GPU - EXPERIMENTAL) ==="
echo "Started: $(date)"
echo ""
echo "⚠️  WARNING: MPS backend is experimental"
echo "   If training crashes, use ./launch_v4_cpu.sh instead"
echo ""

# NOTE: KMP_DUPLICATE_LIB_OK not needed for MPS (doesn't use OpenMP)
# The OpenMP fix is CPU-specific

BATCH_SIZE=32
ACCUM_STEPS=16
EPOCHS=30
LR=5e-5
LR_MIN=1e-6
WD=0.01
TAU=0.05
MARGIN=0.03
MEMORY_BANK_SIZE=50000
MINE_SCHEDULE="0-5:none;6-10:8@0.82-0.92;11-30:16@0.84-0.96"
FILTER_THRESHOLD=0.98

PAIRS="artifacts/twotower/pairs_v4_synth.npz"
BANK="artifacts/wikipedia_500k_corrected_vectors.npz"
OUT_DIR="runs/twotower_v4_mps"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/twotower_v4_mps_$TIMESTAMP.log"

mkdir -p logs
mkdir -p "$OUT_DIR"

echo "Configuration:"
echo "  Batch size: $BATCH_SIZE × $ACCUM_STEPS = $((BATCH_SIZE * ACCUM_STEPS)) effective"
echo "  Device: MPS (Apple GPU - 40 cores)"
echo "  Training pairs: 35,901"
echo "  Epochs: $EPOCHS"
echo "  Expected: 2x faster than CPU (~30-45 min) IF stable"
echo "  Log: $LOG_FILE"
echo ""
echo "If MPS crashes, the CPU version is running in parallel as backup"
echo ""

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
  --device mps \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "✅ MPS TRAINING COMPLETED SUCCESSFULLY!"
  echo "=========================================="
  echo ""
  echo "MPS is stable on your system - use it for future runs!"
  echo "Model saved: $OUT_DIR/best_model.pt"
  echo ""
else
  echo ""
  echo "=========================================="
  echo "❌ MPS TRAINING CRASHED (Exit code: $EXIT_CODE)"
  echo "=========================================="
  echo ""
  echo "This is expected - MPS backend is experimental."
  echo "Use CPU training instead: ./launch_v4_cpu.sh"
  echo ""
  echo "Last 50 lines of log:"
  tail -50 "$LOG_FILE"
  echo ""
  exit $EXIT_CODE
fi
