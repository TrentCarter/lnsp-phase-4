#!/bin/bash
# Monitor async mining training progress

LOG_FILE="logs/twotower_v4_mps_async_20251021_223608.log"

echo "=== ASYNC MINING TRAINING MONITOR ==="
echo ""
echo "Watching: $LOG_FILE"
echo "Press Ctrl+C to stop monitoring"
echo ""
echo "Key metrics to watch:"
echo "  - Throughput: Should stay ~13 it/s even when mining starts (epoch 6+)"
echo "  - Queue health: out_q should be > 0 when mining active"
echo "  - Separation Δ: Should grow to ≥0.05 by epoch 10"
echo ""
echo "============================================================"
echo ""

# Check if process is running
if ! ps aux | grep -q "[t]rain_twotower_v4.py.*async"; then
    echo "❌ Training process not running!"
    exit 1
fi

# Monitor loop
while true; do
    echo "--- $(date +%H:%M:%S) ---"

    # Get current epoch
    EPOCH=$(grep -o "Epoch [0-9]*" "$LOG_FILE" | tail -1)
    echo "Current: $EPOCH"

    # Get latest training speed
    SPEED=$(grep -o "[0-9.]*it/s" "$LOG_FILE" | tail -1)
    echo "Speed: $SPEED"

    # Get queue health (if mining started)
    QUEUE=$(grep "Queue Health" "$LOG_FILE" | tail -1)
    if [ -n "$QUEUE" ]; then
        echo "$QUEUE"
    fi

    # Get latest separation margin (if available)
    MARGIN=$(grep "Separation Δ" "$LOG_FILE" | tail -1)
    if [ -n "$MARGIN" ]; then
        echo "$MARGIN"
    fi

    # Check if training finished
    if grep -q "TRAINING COMPLETE" "$LOG_FILE"; then
        echo ""
        echo "✅ Training completed!"
        tail -20 "$LOG_FILE"
        exit 0
    fi

    # Check if training crashed
    if grep -q "CRASH" "$LOG_FILE"; then
        echo ""
        echo "❌ Training crashed!"
        tail -50 "$LOG_FILE"
        exit 1
    fi

    echo ""
    sleep 30
done
