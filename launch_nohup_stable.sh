#!/usr/bin/env bash
set -euo pipefail

# 🚀 STABLE TWO-TOWER TRAINING (nohup + caffeinate)
# Alternative launcher using nohup (if tmux not available)
# Created: 2025-10-22

echo "════════════════════════════════════════════════════════════"
echo "  🚀 STABLE TWO-TOWER TRAINING (nohup launcher)"
echo "════════════════════════════════════════════════════════════"
echo ""

# Set resource limits
ulimit -n 4096

# Set environment for stability
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export FAISS_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

# Create output directory
OUTDIR="runs/stable_sync_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
mkdir -p logs
LOGFILE="logs/train_sync_$(date +%F_%H%M).out"

export OUTDIR

echo "Configuration:"
echo "  Device: CPU (stable mode)"
echo "  Miner: Synchronous FAISS"
echo "  Data: Wikipedia 771k vectors"
echo "  Epochs: 5"
echo "  Batch: 8 × 2 accum = 16 effective"
echo ""
echo "Environment:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
echo "  PYTHONFAULTHANDLER=$PYTHONFAULTHANDLER"
echo ""
echo "Output:"
echo "  Directory: $OUTDIR"
echo "  Log: $LOGFILE"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

# Launch with caffeinate + nohup
caffeinate -dims nohup ./.venv/bin/python3 -X faulthandler -u tools/train_stable_sync.py \
    1>"$LOGFILE" 2>&1 &

# Get PID
TRAINING_PID=$!

# Disown to prevent terminal close from killing it
disown

# Save PID and paths
echo $TRAINING_PID > /tmp/lnsp_training.pid
echo "$OUTDIR" > /tmp/lnsp_training_outdir.txt
echo "$LOGFILE" > /tmp/lnsp_training_log.txt

echo "Training launched in background (PID: $TRAINING_PID)"
echo ""
echo "Monitor with:"
echo "  tail -f $LOGFILE"
echo "  ps -p $TRAINING_PID"
echo ""
echo "Check progress:"
echo "  grep -E '\\[.*\\].*Epoch' $LOGFILE | tail -10"
echo ""
echo "On-demand stack dump:"
echo "  kill -USR1 $TRAINING_PID"
echo ""
echo "Kill training:"
echo "  kill $TRAINING_PID"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ Training started!"
echo "════════════════════════════════════════════════════════════"
