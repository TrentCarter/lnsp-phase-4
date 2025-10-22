#!/bin/bash
# V4 Training - MPS without gradient accumulation (test)

set -e

echo "=== V4 MPS TEST (NO GRADIENT ACCUMULATION) ==="
echo "Testing if gradient accum is causing MPS crashes..."
echo ""

BATCH_SIZE=16
ACCUM_STEPS=1  # No accumulation
EPOCHS=5  # Just test first 5 epochs
LR=5e-5
LR_MIN=1e-6
WD=0.01
TAU=0.05
MARGIN=0.03
MEMORY_BANK_SIZE=50000
MINE_SCHEDULE="0-5:none"  # No hard negatives for test
FILTER_THRESHOLD=0.98

PAIRS="artifacts/twotower/pairs_v4_synth.npz"
BANK="artifacts/wikipedia_500k_corrected_vectors.npz"
OUT_DIR="runs/twotower_v4_test"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/twotower_v4_test_$TIMESTAMP.log"

mkdir -p logs runs/twotower_v4_test

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
  > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "✓ Test launched (PID: $TRAIN_PID, batch=16, accum=1, epochs=5)"
echo "  Log: $LOG_FILE"
echo "  Monitor: tail -f $LOG_FILE"
