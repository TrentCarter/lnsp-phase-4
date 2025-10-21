#!/bin/bash
# V4 Training Gate Monitor
# Checks Recall@500 and margin at key decision points

LOG_FILE=$(ls -t logs/twotower_v4_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ No v4 log file found"
    exit 1
fi

echo "=== V4 TRAINING GATE MONITOR ==="
echo "Log: $LOG_FILE"
echo ""

# Check if training is running
if pgrep -f "train_twotower_v4.py" > /dev/null; then
    PID=$(pgrep -f "train_twotower_v4.py")
    echo "✅ Training running (PID: $PID)"
    ps -p $PID -o pid,%cpu,%mem,rss,etime | tail -1
else
    echo "⚠️  Training not running"
fi

echo ""
echo "=== PROGRESS ==="

# Extract epoch and recall info
grep -E "Epoch [0-9]+/30|Recall@500|margin|Best Recall@500" "$LOG_FILE" | tail -50 | while IFS= read -r line; do
    # Highlight key epochs
    if echo "$line" | grep -q "Epoch 5/30"; then
        echo "🎯 GATE 1 (Epoch 5): $line"
    elif echo "$line" | grep -q "Epoch 10/30"; then
        echo "🎯 GATE 2 (Epoch 10): $line"
    else
        echo "$line"
    fi
done

echo ""
echo "=== LATEST METRICS ==="
grep -E "Recall@500|Best Recall@500" "$LOG_FILE" | tail -5

echo ""
echo "=== DECISION GATES ==="
echo "📍 Gate 1 (Epoch 5): Target Recall@500 ≥ 20-25%, margin Δ ≥ 0.03"
echo "📍 Gate 2 (Epoch 10): Target Recall@500 ≥ 35-40%, margin Δ ≥ 0.05"
echo ""
echo "Run: watch -n 60 ./monitor_v4_gates.sh"
