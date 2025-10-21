#!/bin/bash
# Launch v4 Two-Tower Training (Overnight Run)
# Uses curriculum-based hard negative mining

set -e

echo "=== LAUNCHING V4 TWO-TOWER TRAINING (OVERNIGHT) ==="
echo "Started: $(date)"
echo ""

# Configuration
PAIRS="artifacts/twotower/pairs_v4_synth.npz"
BANK="artifacts/wikipedia_500k_corrected_vectors.npz"
OUT_DIR="runs/twotower_v4"
LOG_DIR="logs"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUT_DIR"

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f "$PAIRS" ]; then
    echo "✗ Training pairs not found: $PAIRS"
    echo "  Run: ./tools/expand_pairs_to_v4.py first"
    exit 1
fi
echo "✓ Training pairs found: $PAIRS"

if [ ! -f "$BANK" ]; then
    echo "✗ Vector bank not found: $BANK"
    exit 1
fi
echo "✓ Vector bank found: $BANK"

# Training parameters (from your spec)
BATCH_SIZE=32
ACCUM_STEPS=16        # Effective batch = 512
EPOCHS=30
LR=5e-5
LR_MIN=1e-6
WD=0.01
TAU=0.05
MARGIN=0.03
MEMORY_BANK_SIZE=50000

# Curriculum mining schedule
# Epochs 1-5: in-batch + memory only (no hard negs)
# Epochs 6-10: add 8 hard negs (cos 0.82-0.92)
# Epochs 11-30: 16 hard negs (cos 0.84-0.96)
MINE_SCHEDULE="0-5:none;6-10:8@0.82-0.92;11-30:16@0.84-0.96"
FILTER_THRESHOLD=0.98

# Log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/twotower_v4_$TIMESTAMP.log"

echo ""
echo "Configuration:"
echo "  Batch size: $BATCH_SIZE × $ACCUM_STEPS accum = $((BATCH_SIZE * ACCUM_STEPS)) effective"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR → $LR_MIN (cosine decay)"
echo "  Temperature: $TAU"
echo "  Margin: $MARGIN"
echo "  Memory bank size: $MEMORY_BANK_SIZE"
echo "  Mining schedule: $MINE_SCHEDULE"
echo "  Filter threshold: $FILTER_THRESHOLD"
echo "  Output: $OUT_DIR"
echo "  Log: $LOG_FILE"
echo ""

# Launch training in background
echo "Launching training in background..."

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
  > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

echo "✓ Training launched in background"
echo "  PID: $TRAIN_PID"
echo "  Log: $LOG_FILE"
echo ""

# Save PID for later monitoring
echo $TRAIN_PID > "$OUT_DIR/train.pid"

echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""

echo "To check if still running:"
echo "  ps -p $TRAIN_PID"
echo ""

echo "To stop training:"
echo "  kill $TRAIN_PID"
echo ""

echo "=== V4 TRAINING STARTED ==="
echo "Launched: $(date)"
