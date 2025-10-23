#!/usr/bin/env bash
set -euo pipefail

# 🚀 SATT TRAINING LAUNCHER
# Sequence-Aware Two-Tower with Multi-task Objective

echo "════════════════════════════════════════════════════════════"
echo "  🚀 SATT: SEQUENCE-AWARE TWO-TOWER TRAINING"
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
OUTDIR="runs/satt_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
mkdir -p logs
LOGFILE="logs/satt_$(date +%F_%H%M).out"

export OUTDIR

echo "Configuration:"
echo "  Objective: L_seq + 0.3 * L_sim"
echo "  L_seq: Exact next vector prediction"
echo "  L_sim: Semantic similarity (auxiliary)"
echo "  Device: CPU (stable mode)"
echo "  Miner: Synchronous FAISS (nprobe=12)"
echo "  Data: Wikipedia 771k vectors"
echo "  Epochs: 5"
echo "  Batch: 8 × 2 accum = 16 effective"
echo "  Hard negatives: After 2k warmup steps"
echo ""
echo "Output:"
echo "  Directory: $OUTDIR"
echo "  Log: $LOGFILE"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

# Launch with caffeinate + nohup
caffeinate -dims nohup ./.venv/bin/python3 -X faulthandler -u tools/train_satt.py \
    1>"$LOGFILE" 2>&1 &

# Get PID
TRAINING_PID=$!

# Disown
disown

# Save PID and paths
echo $TRAINING_PID > /tmp/lnsp_satt_training.pid
echo "$OUTDIR" > /tmp/lnsp_satt_training_outdir.txt
echo "$LOGFILE" > /tmp/lnsp_satt_training_log.txt

echo "SATT training launched in background (PID: $TRAINING_PID)"
echo ""
echo "Monitor with:"
echo "  tail -f $LOGFILE"
echo "  ps -p $TRAINING_PID"
echo ""
echo "Check progress:"
echo "  grep -E 'L_seq=' $LOGFILE | tail -10"
echo ""
echo "On-demand stack dump:"
echo "  kill -USR1 $TRAINING_PID"
echo ""
echo "Kill training:"
echo "  kill $TRAINING_PID"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ SATT Training started!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Expected behavior:"
echo "  - Warmup (steps 0-2000): Random negatives"
echo "  - After warmup: Hard negatives with same-doc confounders"
echo "  - Loss should decrease steadily"
echo "  - L_seq should be primary (larger than L_sim)"
echo ""
echo "After training completes (~2 minutes):"
echo "  python tools/evaluate_two_tower.py --checkpoint $OUTDIR/epoch_005.pt"
echo ""
