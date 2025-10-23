#!/usr/bin/env bash
set -euo pipefail

# 🔴 CRITICAL: macOS OpenMP fix (prevents "Abort trap: 6")
export KMP_DUPLICATE_LIB_OK=TRUE

# Stable Two-Tower Training Launch (CPU + Sync FAISS)
echo "════════════════════════════════════════════════════════════"
echo "  🚀 LAUNCHING STABLE TWO-TOWER TRAINING"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Device: CPU (stable mode)"
echo "  Miner: Synchronous FAISS (no multiprocessing)"
echo "  Data: Wikipedia 771k vectors"
echo "  Epochs: 5 (short run for validation)"
echo "  Batch: 8 × 2 accum = 16 effective"
echo "  OpenMP: KMP_DUPLICATE_LIB_OK=TRUE ✓"
echo ""
echo "Estimated time: ~2-3 hours (5 epochs)"
echo ""

# Set output directory with timestamp
OUTDIR="runs/stable_sync_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/training.log"

echo "  Output: $OUTDIR"
echo ""
echo "════════════════════════════════════════════════════════════"

# Launch training (with -u for unbuffered output)
export OUTDIR
PYTHONPATH=. ./.venv/bin/python3 -u tools/train_stable_sync.py > "$LOGFILE" 2>&1 &

# Get background PID
TRAINING_PID=$!
echo ""
echo "Training launched in background (PID: $TRAINING_PID)"
echo ""
echo "Monitor with:"
echo "  tail -f $LOGFILE"
echo "  ps aux | grep $TRAINING_PID"
echo ""
echo "Kill with:"
echo "  kill $TRAINING_PID"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ Training started!"
echo "════════════════════════════════════════════════════════════"

# Save PID for later monitoring
echo $TRAINING_PID > /tmp/lnsp_training.pid
echo "$OUTDIR" > /tmp/lnsp_training_outdir.txt
echo "$LOGFILE" > /tmp/lnsp_training_log.txt
