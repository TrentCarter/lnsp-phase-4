#!/bin/bash
# Launch Phase 2 Two-Tower Training (Overnight Run)
# This script starts Phase 2 training in the background with hard negatives

set -e

echo "=== LAUNCHING PHASE 2 TWO-TOWER TRAINING (OVERNIGHT) ==="
echo "Started: $(date)"
echo ""

# Configuration
PAIRS="artifacts/twotower/pairs_v3_synth.npz"
BANK="artifacts/wikipedia_500k_corrected_vectors.npz"
INIT_CKPT="runs/twotower_v3_phase1/checkpoints/best_recall500.pt"
OUT_DIR="runs/twotower_v3_phase2"
LOG_DIR="logs"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUT_DIR"

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f "$PAIRS" ]; then
    echo "✗ Training pairs not found: $PAIRS"
    exit 1
fi
echo "✓ Training pairs found: $PAIRS"

if [ ! -f "$BANK" ]; then
    echo "✗ Vector bank not found: $BANK"
    exit 1
fi
echo "✓ Vector bank found: $BANK"

if [ ! -f "$INIT_CKPT" ]; then
    echo "⚠️  Phase 1 checkpoint not found: $INIT_CKPT"
    echo "   Will train from scratch instead"
    INIT_ARG=""
else
    echo "✓ Phase 1 checkpoint found: $INIT_CKPT"
    INIT_ARG="--init-ckpt $INIT_CKPT"
fi

# Training parameters
BATCH_SIZE=32
ACCUM_STEPS=8
EPOCHS=50
LR=1e-5
WD=0.01
TAU=0.07
MEMORY_BANK_SIZE=20000
MINE_EVERY=2
NUM_HARD_NEGS=16

# Log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/twotower_phase2_$TIMESTAMP.log"

echo ""
echo "Configuration:"
echo "  Batch size: $BATCH_SIZE × $ACCUM_STEPS accum = $((BATCH_SIZE * ACCUM_STEPS)) effective"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Memory bank size: $MEMORY_BANK_SIZE"
echo "  Mine hard negatives every: $MINE_EVERY epochs"
echo "  Hard negatives per sample: $NUM_HARD_NEGS"
echo "  Output: $OUT_DIR"
echo "  Log: $LOG_FILE"
echo ""

# Launch training in background
echo "Launching training in background..."

./.venv/bin/python3 tools/train_twotower_phase2.py \
  --pairs "$PAIRS" \
  --bank "$BANK" \
  $INIT_ARG \
  --bs $BATCH_SIZE \
  --accum $ACCUM_STEPS \
  --epochs $EPOCHS \
  --lr $LR \
  --wd $WD \
  --tau $TAU \
  --memory-bank-size $MEMORY_BANK_SIZE \
  --mine-every $MINE_EVERY \
  --num-hard-negs $NUM_HARD_NEGS \
  --out "$OUT_DIR" \
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
echo "  ./tools/monitor_training.sh $OUT_DIR"
echo ""

echo "To check if still running:"
echo "  ps -p $TRAIN_PID"
echo ""

echo "To stop training:"
echo "  kill $TRAIN_PID"
echo ""

echo "=== PHASE 2 TRAINING STARTED ==="
echo "Launched: $(date)"
