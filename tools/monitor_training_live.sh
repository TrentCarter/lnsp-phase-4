#!/bin/bash
# Live Training Monitor - Track Key Metrics in Real-Time
# ========================================================
#
# Usage:
#   bash tools/monitor_training_live.sh artifacts/lvm/models/amn_790k_production_*/training.log
#
# Displays:
#   - Latest epoch metrics (val cosine, train loss)
#   - Loss breakdown (MSE, InfoNCE, moment, variance)
#   - Trend direction (improving/degrading)
#   - Stoplight gate status

LOG_FILE="$1"

if [ -z "$LOG_FILE" ]; then
    echo "Usage: $0 <training_log_file>"
    echo "Example: $0 artifacts/lvm/models/amn_790k_*/training.log"
    exit 1
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

echo "=========================================="
echo "🔍 TRAINING MONITOR (Live)"
echo "=========================================="
echo "Watching: $LOG_FILE"
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "🔍 LVM TRAINING MONITOR"
    echo "=========================================="
    echo "Log: $(basename $LOG_FILE)"
    echo "Updated: $(date +%H:%M:%S)"
    echo ""

    # Latest epoch info
    echo "📊 LATEST EPOCH:"
    LATEST_EPOCH=$(grep -E "Epoch [0-9]+/" "$LOG_FILE" 2>/dev/null | tail -1)
    if [ -n "$LATEST_EPOCH" ]; then
        echo "$LATEST_EPOCH"
    else
        echo "   (waiting for first epoch...)"
    fi
    echo ""

    # Val cosine trend (last 5 epochs)
    echo "📈 VAL COSINE TREND (last 5 epochs):"
    grep "val_cosine" "$LOG_FILE" 2>/dev/null | tail -5 | while read line; do
        echo "   $line"
    done
    echo ""

    # Loss breakdown (latest)
    echo "🔧 LOSS BREAKDOWN (latest):"
    echo "   MSE:      $(grep "train_loss_mse" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo 'N/A')"
    echo "   InfoNCE:  $(grep "train_loss_info" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo 'N/A')"
    echo "   Moment:   $(grep "loss_moment" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9e+-]+' || echo 'N/A')"
    echo "   Variance: $(grep "loss_variance" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9e+-]+' || echo 'N/A')"
    echo ""

    # Stoplight gates
    echo "🚦 STOPLIGHT GATES:"
    CURRENT_EPOCH=$(grep -oE "Epoch [0-9]+" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
    CURRENT_VAL=$(grep "val_cosine" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+")

    if [ -n "$CURRENT_EPOCH" ] && [ -n "$CURRENT_VAL" ]; then
        echo "   Current: Epoch $CURRENT_EPOCH, Val Cosine $CURRENT_VAL"
        echo ""

        # Check gates
        if [ $CURRENT_EPOCH -ge 3 ]; then
            if (( $(echo "$CURRENT_VAL >= 0.48" | bc -l) )); then
                echo "   ✅ Epoch 3 gate: PASS (val ≥ 0.48)"
            else
                echo "   ⚠️  Epoch 3 gate: FAIL (val < 0.48)"
            fi
        fi

        if [ $CURRENT_EPOCH -ge 6 ]; then
            if (( $(echo "$CURRENT_VAL >= 0.50" | bc -l) )); then
                echo "   ✅ Epoch 6 gate: PASS (val ≥ 0.50)"
            else
                echo "   ⚠️  Epoch 6 gate: FAIL (val < 0.50)"
            fi
        fi

        if [ $CURRENT_EPOCH -ge 10 ]; then
            if (( $(echo "$CURRENT_VAL >= 0.50" | bc -l) )); then
                echo "   ✅ Epoch 10 gate: PASS (val ≥ 0.50)"
            else
                echo "   ⚠️  Epoch 10 gate: FAIL (val < 0.50)"
            fi
        fi

        if [ $CURRENT_EPOCH -ge 20 ]; then
            if (( $(echo "$CURRENT_VAL >= 0.54" | bc -l) )); then
                echo "   ✅ Epoch 20 gate: PASS (val ≥ 0.54)"
            else
                echo "   ⚠️  Epoch 20 gate: FAIL (val < 0.54)"
            fi
        fi
    else
        echo "   (waiting for epoch data...)"
    fi
    echo ""

    # ETA estimate
    ELAPSED_TIME=$(find "$(dirname $LOG_FILE)" -name "training.log" -printf "%T@\n" 2>/dev/null)
    if [ -n "$ELAPSED_TIME" ]; then
        START_TIME=$(date -r "$(dirname $LOG_FILE)" +%s 2>/dev/null || echo "0")
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))

        if [ $ELAPSED -gt 0 ] && [ -n "$CURRENT_EPOCH" ] && [ $CURRENT_EPOCH -gt 0 ]; then
            TIME_PER_EPOCH=$((ELAPSED / CURRENT_EPOCH))
            REMAINING_EPOCHS=$((30 - CURRENT_EPOCH))
            ETA_SECONDS=$((TIME_PER_EPOCH * REMAINING_EPOCHS))
            ETA_HOURS=$((ETA_SECONDS / 3600))
            ETA_MINS=$(((ETA_SECONDS % 3600) / 60))

            echo "⏱️  TIMING:"
            echo "   Elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
            echo "   Time per epoch: $((TIME_PER_EPOCH / 60))m"
            echo "   ETA: ${ETA_HOURS}h ${ETA_MINS}m"
        fi
    fi
    echo ""
    echo "=========================================="
    echo "Refreshing every 30s... (Ctrl+C to exit)"

    sleep 30
done
