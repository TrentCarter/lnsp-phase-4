#!/bin/bash
#
# Monitor V3 training progress in real-time
# Watches training logs and highlights key metrics
#
# Usage: ./scripts/monitor_training_v3.sh [log_file]
#
# If no log file specified, watches most recent training output
#

LOG_FILE="${1:-artifacts/lvm/models/transformer_directional_v3/training.log}"

if [ ! -f "$LOG_FILE" ]; then
    echo "⚠️  Log file not found: $LOG_FILE"
    echo ""
    echo "Waiting for training to start..."
    echo "(Run: ./scripts/train_transformer_directional_v3.sh > $LOG_FILE 2>&1 &)"
    echo ""
    # Wait for file to appear
    while [ ! -f "$LOG_FILE" ]; do
        sleep 2
    done
fi

echo "=========================================="
echo "Monitoring V3 Training"
echo "=========================================="
echo "Log: $LOG_FILE"
echo ""
echo "Key Metrics to Watch:"
echo "  🎯 Margin(+1 vs last): Should turn POSITIVE by epoch 7-8"
echo "  📈 Val Cosine: Should stay HIGH (≥ 0.50)"
echo "  🔍 Loss phases: Warm-up → Ramp → Full"
echo ""
echo "Starting tail..."
echo ""

tail -f "$LOG_FILE" | while read line; do
    # Highlight epoch starts
    if echo "$line" | grep -q "^Epoch [0-9]"; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo -e "\033[1;36m$line\033[0m"  # Cyan bold
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    # Highlight phase transitions
    elif echo "$line" | grep -q "\[Warm-up\]"; then
        echo -e "\033[1;33m$line\033[0m"  # Yellow bold
    elif echo "$line" | grep -q "\[Ramp\]"; then
        echo -e "\033[1;34m$line\033[0m"  # Blue bold
    elif echo "$line" | grep -q "\[Full\]"; then
        echo -e "\033[1;35m$line\033[0m"  # Magenta bold
    # Highlight positive margin
    elif echo "$line" | grep -q "✅.*Positive margin"; then
        echo -e "\033[1;32m$line\033[0m"  # Green bold
    # Highlight negative margin warning
    elif echo "$line" | grep -q "⚠️.*Negative margin"; then
        echo -e "\033[1;31m$line\033[0m"  # Red bold
    # Highlight validation results
    elif echo "$line" | grep -q "Val Loss:"; then
        echo -e "\033[1;37m$line\033[0m"  # White bold
    # Highlight directional stats
    elif echo "$line" | grep -q "Directional:"; then
        # Extract margin value
        margin=$(echo "$line" | sed -n 's/.*Margin(+1 vs last)=\([^ ]*\).*/\1/p')
        if [ ! -z "$margin" ]; then
            # Check if margin is positive (starts with digit or +)
            if echo "$margin" | grep -q "^[0-9]"; then
                echo -e "\033[1;32m$line\033[0m"  # Green if positive
            else
                echo -e "\033[1;31m$line\033[0m"  # Red if negative
            fi
        else
            echo "$line"
        fi
    else
        echo "$line"
    fi
done
