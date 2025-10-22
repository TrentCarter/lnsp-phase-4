#!/bin/bash
# V4 Training - Safer MPS Profile (low accum, frequent sync, fp32)

set -e

echo "=== V4 TWO-TOWER TRAINING (MPS SAFE MODE) ==="
echo "Started: $(date)"
echo ""

# Safer MPS configuration
BATCH_SIZE=24          # Increased from 16
ACCUM_STEPS=4          # Reduced from 32 (keeps effective batch ~96-128)
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
OUT_DIR="runs/twotower_v4"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/twotower_v4_safe_$TIMESTAMP.log"

mkdir -p logs

echo "Configuration (SAFE MPS):"
echo "  Batch size: $BATCH_SIZE × $ACCUM_STEPS = $((BATCH_SIZE * ACCUM_STEPS)) effective"
echo "  Device: MPS (safe mode: fp32, frequent sync, low accum)"
echo "  Training pairs: 35,901"
echo "  Epochs: $EPOCHS"
echo "  Log: $LOG_FILE"
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
  > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > "$OUT_DIR/train.pid"

echo "✓ Training launched (PID: $TRAIN_PID)"
echo "  Monitor: tail -f $LOG_FILE"
echo "  Progress: ./check_v4_progress.sh"
echo ""
echo "Safe MPS mode features:"
echo "  • Low gradient accumulation (4 steps vs 32)"
echo "  • FP32 numerics (no autocast on MPS)"
echo "  • MPS cache clearing before FAISS ops"
echo "  • Smaller effective batch to reduce memory pressure"
