#!/usr/bin/env bash
set -euo pipefail

# 🚀 STABLE TWO-TOWER TRAINING (tmux + caffeinate)
# This launcher uses tmux to avoid macOS background process issues
# Created: 2025-10-22

echo "════════════════════════════════════════════════════════════"
echo "  🚀 STABLE TWO-TOWER TRAINING (tmux launcher)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "This will start training in a tmux session to avoid:"
echo "  • App Nap (macOS background throttling)"
echo "  • TTY/file descriptor edge cases"
echo "  • stdout buffering issues"
echo ""
echo "Session name: tt_sync"
echo "Detach with: Ctrl-B then D"
echo "Reattach with: tmux attach -t tt_sync"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ ERROR: tmux not found"
    echo ""
    echo "Install with: brew install tmux"
    echo ""
    exit 1
fi

# Check if session already exists
if tmux has-session -t tt_sync 2>/dev/null; then
    echo "⚠️  Session 'tt_sync' already exists"
    echo ""
    read -p "Kill and restart? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t tt_sync
        echo "✓ Old session killed"
    else
        echo "Exiting. Use 'tmux attach -t tt_sync' to reconnect."
        exit 0
    fi
fi

# Create output directory
OUTDIR="runs/stable_sync_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/training.log"

echo "Output directory: $OUTDIR"
echo "Log file: $LOGFILE"
echo ""
echo "Starting tmux session..."
echo ""

# Create tmux session with caffeinate and proper environment
tmux new-session -d -s tt_sync bash --login -c "
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
export OUTDIR='$OUTDIR'

# Activate venv
cd '$(pwd)'
source ./.venv/bin/activate

# Run training with caffeinate (prevents sleep) and faulthandler
echo '════════════════════════════════════════════════════════════'
echo '  TRAINING STARTING'
echo '════════════════════════════════════════════════════════════'
echo ''
echo 'Environment:'
echo \"  PYTHONUNBUFFERED=\$PYTHONUNBUFFERED\"
echo \"  OMP_NUM_THREADS=\$OMP_NUM_THREADS\"
echo \"  Output: \$OUTDIR\"
echo ''
echo 'Press Ctrl-C to stop training'
echo 'Press Ctrl-B then D to detach (training continues)'
echo ''
echo '════════════════════════════════════════════════════════════'
echo ''

caffeinate -dims python3 -X faulthandler -u tools/train_stable_sync.py 2>&1 | tee '$LOGFILE'

echo ''
echo '════════════════════════════════════════════════════════════'
echo '  TRAINING FINISHED'
echo '════════════════════════════════════════════════════════════'
echo ''
read -p 'Press Enter to close session...'
"

# Save session info
echo "tt_sync" > /tmp/lnsp_training_session.txt
echo "$OUTDIR" > /tmp/lnsp_training_outdir.txt
echo "$LOGFILE" > /tmp/lnsp_training_log.txt

echo "════════════════════════════════════════════════════════════"
echo "  ✅ Training session started!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Commands:"
echo "  Attach to session:  tmux attach -t tt_sync"
echo "  Check if running:   tmux list-sessions | grep tt_sync"
echo "  Monitor log:        tail -f $LOGFILE"
echo "  Kill session:       tmux kill-session -t tt_sync"
echo ""
echo "Inside tmux:"
echo "  Detach (keeps running):  Ctrl-B then D"
echo "  Stop training:           Ctrl-C"
echo ""
echo "Session will automatically close when training completes."
echo ""
echo "════════════════════════════════════════════════════════════"
