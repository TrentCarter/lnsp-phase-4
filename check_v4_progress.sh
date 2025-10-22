#!/bin/bash
# Quick progress check for v4 training

LOG=$(ls -t logs/twotower_v4*.log 2>/dev/null | head -1)

if [ -z "$LOG" ]; then
    echo "❌ No log file found"
    exit 1
fi

echo "=== V4 TRAINING STATUS ==="
echo "Log: $LOG"
echo ""

# Check if running
if pgrep -f "train_twotower_v4.py" > /dev/null; then
    PID=$(pgrep -f "train_twotower_v4.py")
    echo "✅ Training RUNNING (PID: $PID)"
else
    echo "⚠️  Training NOT running"
fi

echo ""
echo "=== LATEST PROGRESS ===" 
tail -30 "$LOG" | grep -E "Epoch|Recall@500|Loss|Training:" | tail -15
