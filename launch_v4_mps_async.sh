#!/bin/bash
# V4 Training - MPS with Async Mining
# Overlaps FAISS mining with training for better GPU utilization

set -e

echo "=== V4 TWO-TOWER TRAINING (MPS + ASYNC MINING) ==="
echo "Started: $(date)"
echo ""
echo "🚀 Using async FAISS mining (overlaps with training)"
echo "   Expected speedup: 2-3x over synchronous mining"
echo ""

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

# Async mining parameters (optimized for throughput)
MINE_K=128          # Candidate pool size
MINE_QBATCH=2048    # FAISS query batch size (aggressive batching)
MINE_PREFETCH=3     # Queue depth (deeper prefetch for FAISS hiccups)
MINE_TTL=5          # Reuse mined negatives for N steps (reduce FAISS calls)
HARDNEG_FRAC=0.5    # 50% from FAISS, 50% from memory/in-batch
MINE_REFRESH_STEPS=3000  # Refresh mining every 3k steps (not every 1k)

PAIRS="artifacts/twotower/pairs_v4_synth.npz"
BANK="artifacts/wikipedia_500k_corrected_vectors.npz"
OUT_DIR="runs/twotower_v4_mps_async"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/twotower_v4_mps_async_$TIMESTAMP.log"

mkdir -p logs
mkdir -p "$OUT_DIR"

echo "Configuration:"
echo "  Batch size: $BATCH_SIZE × $ACCUM_STEPS = $((BATCH_SIZE * ACCUM_STEPS)) effective"
echo "  Device: MPS (Apple GPU - 40 cores)"
echo "  Training pairs: 35,901"
echo "  Epochs: $EPOCHS"
echo "  Async mining: enabled (k=$MINE_K, qbatch=$MINE_QBATCH, ttl=$MINE_TTL)"
echo "  Expected: ~20-30 min (GPU overlaps FAISS + training)"
echo "  Log: $LOG_FILE"
echo ""

# 🚨 CRITICAL: Fix OpenMP conflict on macOS (FAISS + PyTorch both use OpenMP)
export KMP_DUPLICATE_LIB_OK=TRUE

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
  --async-mining \
  --mine-k $MINE_K \
  --mine-qbatch $MINE_QBATCH \
  --mine-prefetch $MINE_PREFETCH \
  --mine-ttl $MINE_TTL \
  --hardneg-frac $HARDNEG_FRAC \
  --mine-refresh-steps $MINE_REFRESH_STEPS \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "✅ ASYNC MINING TRAINING COMPLETED!"
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
