#!/bin/bash
# Monitor two-tower training progress
# Usage: ./monitor_training.sh <run_dir>

RUN_DIR="${1:-runs/twotower_v3_phase1}"
LOG_FILE="$RUN_DIR/training.log"

echo "=== TWO-TOWER TRAINING MONITOR ==="
echo "Run: $RUN_DIR"
echo "Started: $(date)"
echo ""

# Check if process is running
if ps aux | grep "train_twotower.py" | grep -v grep > /dev/null; then
    PID=$(ps aux | grep "train_twotower.py" | grep -v grep | awk '{print $2}')
    echo "✓ Training process running (PID: $PID)"
    ps aux | grep "train_twotower.py" | grep -v grep | awk '{print "  CPU: "$3"%, Mem: "$4"%, Time: "$10}'
else
    echo "✗ Training process not found"
    exit 1
fi

echo ""

# Extract latest metrics from log
if [ -f "$LOG_FILE" ]; then
    echo "=== LATEST METRICS ==="

    # Find all epoch results
    grep -E "Epoch [0-9]+/[0-9]+" "$LOG_FILE" | tail -5
    echo ""

    # Find latest recall metrics
    grep -E "Recall@" "$LOG_FILE" | tail -4
    echo ""

    # Find best model saves
    grep "New best" "$LOG_FILE" | tail -3
else
    echo "Log file not found: $LOG_FILE"
fi

echo ""
echo "Monitor completed: $(date)"
