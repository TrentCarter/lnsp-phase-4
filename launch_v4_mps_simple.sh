#!/bin/bash
# V4 Training - MPS Simple Profile (NO async mining)
# Clean synchronous training for maximum stability

set -e

echo "=== V4 TWO-TOWER TRAINING (MPS SIMPLE - NO ASYNC) ==="
echo "Started: $(date)"
echo ""
echo "🎯 Profile: SIMPLE (--bs 16 --accum 4 = effective 64)"
echo "   NO async mining (clean synchronous training)"
echo "   Expected: stable completion, ~25-35 min for 30 epochs"
echo ""

# Simple profile: 16×4 = 64 effective batch, NO async mining
BATCH_SIZE=16
ACCUM_STEPS=4
EPOCHS=30
LR=5e-5
LR_MIN=1e-6
WD=0.01
TAU=0.05
MARGIN=0.03
MEMORY_BANK_SIZE=50000
MINE_SCHEDULE="0-5:none;6-10:8@0.82-0.92;11-30:16@0.84-0.96"
FILTER_THRESHOLD=0.98

# MPS stability toggles
GRAD_CLIP=1.0         # Gradient clipping max norm
SYNC_MPS_EVERY=100    # Synchronize MPS every N steps

PAIRS="artifacts/twotower/pairs_v4_synth.npz"
BANK="artifacts/wikipedia_500k_corrected_vectors.npz"
OUT_DIR="runs/twotower_v4_mps_simple"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/twotower_v4_mps_simple_$TIMESTAMP.log"

mkdir -p logs
mkdir -p "$OUT_DIR"

echo "Configuration:"
echo "  Batch size: $BATCH_SIZE × $ACCUM_STEPS = $((BATCH_SIZE * ACCUM_STEPS)) effective"
echo "  Device: MPS (Apple GPU - 40 cores)"
echo "  Training pairs: 35,901"
echo "  Epochs: $EPOCHS"
echo "  Async mining: DISABLED (synchronous for stability)"
echo "  MPS stability: grad-clip=$GRAD_CLIP, sync-every=$SYNC_MPS_EVERY"
echo "  Log: $LOG_FILE"
echo ""

# 🚨 CRITICAL: Fix OpenMP conflict on macOS (FAISS + PyTorch both use OpenMP)
export KMP_DUPLICATE_LIB_OK=TRUE

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
  --device mps \
  --grad-clip $GRAD_CLIP \
  --sync-mps-every $SYNC_MPS_EVERY \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "✅ SIMPLE TRAINING COMPLETED!"
  echo "=========================================="
  echo ""
  echo "Model saved: $OUT_DIR/best.pt"
  echo ""
else
  echo ""
  echo "=========================================="
  echo "❌ TRAINING CRASHED (Exit code: $EXIT_CODE)"
  echo "=========================================="
  echo ""
  echo "Last 50 lines of log:"
  tail -50 "$LOG_FILE"
  echo ""
  exit $EXIT_CODE
fi
