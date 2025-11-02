#!/bin/bash
#
# Quick Training Status Check
#

LOG_DIR="artifacts/lvm/models/transformer_5cat_20251030_230805"
TRAINING_LOG="$LOG_DIR/training.log"

if [ ! -f "$TRAINING_LOG" ]; then
    echo "❌ Training log not found: $TRAINING_LOG"
    exit 1
fi

echo "============================================"
echo "Training Status - $(date +%H:%M:%S)"
echo "============================================"
echo ""

# Check if training is still running
TRAIN_PID=$(ps aux | grep "train_with_5cat_validation.sh transformer" | grep -v grep | awk '{print $2}')
if [ -n "$TRAIN_PID" ]; then
    echo "✅ Training is RUNNING (PID: $TRAIN_PID)"
else
    echo "⚠️  Training process not found (may be completed or failed)"
fi
echo ""

# Show latest epoch progress
echo "📊 Latest Progress:"
tail -20 "$TRAINING_LOG" | grep -E "Epoch [0-9]+/|Batch [0-9]+/|Loss:|Cosine:|5CAT|Margin" | tail -10
echo ""

# Check for 5CAT results
echo "🧪 5CAT Validation Results:"
grep -A 8 "🧪 Running 5CAT Validation" "$TRAINING_LOG" | tail -15 || echo "   No 5CAT results yet (runs every 5 epochs)"
echo ""

# Check for warnings
WARNINGS=$(grep -c "WARNING.*BACKWARD BIAS" "$TRAINING_LOG" 2>/dev/null || echo "0")
if [ "$WARNINGS" -gt 0 ]; then
    echo "🚨 ALERTS: $WARNINGS backward bias warnings detected!"
else
    echo "✅ No backward bias warnings (good!)"
fi
