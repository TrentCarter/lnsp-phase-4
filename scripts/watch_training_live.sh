#!/bin/bash
# Live training monitor with heartbeat updates

echo "=========================================="
echo "LIVE TRAINING MONITOR"
echo "=========================================="
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "EPOCH 4 TRAINING - LIVE VIEW"
    echo "=========================================="
    date
    echo ""

    # Process status
    if ps aux | grep 99328 | grep -v grep > /dev/null; then
        ps aux | grep 99328 | grep -v grep | awk '{print "🏃 Training: PID", $2, "| CPU:", $3"% | Mem:", $4"% | Runtime:", $10}'
    else
        echo "⚠️  Training process not found"
        exit 1
    fi
    echo ""

    # Heartbeat
    if [ -f artifacts/lvm/train_heartbeat.json ]; then
        echo "📊 Progress:"
        cat artifacts/lvm/train_heartbeat.json | jq -r @json | python3 -c "
import json, sys
data = json.load(sys.stdin)
epoch = data['epoch']
step = data['step']
total = data['steps_total']
loss = data['loss']
avg_loss = data['avg_loss']
pct = (step / total * 100) if total > 0 else 0
print(f'  Epoch: {epoch}')
print(f'  Step:  {step:,} / {total:,} ({pct:.1f}%)')
print(f'  Loss:  {loss:.4f}')
print(f'  Avg:   {avg_loss:.4f}')
"

        # Progress bar
        STEP=$(cat artifacts/lvm/train_heartbeat.json | jq -r '.step')
        TOTAL=$(cat artifacts/lvm/train_heartbeat.json | jq -r '.steps_total')
        PCT=$(echo "scale=0; $STEP * 100 / $TOTAL" | bc)
        FILLED=$(echo "scale=0; $PCT / 2" | bc)
        BAR=$(printf "█%.0s" $(seq 1 $FILLED))
        EMPTY=$(printf "░%.0s" $(seq 1 $((50 - FILLED))))
        echo ""
        echo "  [$BAR$EMPTY] $PCT%"
    else
        echo "⏳ Initializing (loading data, building indices...)"
    fi
    echo ""

    # Checkpoint status
    if [ -f artifacts/lvm/models/twotower_fast/epoch4.pt ]; then
        echo "✅ CHECKPOINT COMPLETE!"
        ls -lh artifacts/lvm/models/twotower_fast/epoch4.pt | awk '{print "   Size:", $5, "| Created:", $6, $7, $8}'
        echo ""
        echo "🎯 Evaluation running automatically..."
        break
    fi

    echo "=========================================="
    echo "Refreshing every 5 seconds..."
    sleep 5
done

echo ""
echo "Training complete! Check evaluation results:"
echo "  cat artifacts/lvm/eval_epoch4/metrics.json | jq"
