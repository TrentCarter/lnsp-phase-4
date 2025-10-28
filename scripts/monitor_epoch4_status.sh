#!/bin/bash
# Quick status check for Epoch 4 training and evaluation

echo "=========================================="
echo "EPOCH 4 TRAINING STATUS"
echo "=========================================="
date
echo ""

# Training process
echo "Training Process:"
if ps aux | grep 99328 | grep -v grep > /dev/null; then
    ps aux | grep 99328 | grep -v grep | awk '{print "  ✅ Active - PID:", $2, "| CPU:", $3"% | Memory:", $4"% | Runtime:", $10}'
else
    echo "  ⚠️  Training process not found"
fi
echo ""

# Heartbeat
echo "Latest Heartbeat:"
if [ -f artifacts/lvm/train_heartbeat.json ]; then
    cat artifacts/lvm/train_heartbeat.json | jq -r '"  Epoch: \(.epoch) | Step: \(.step)/\(.steps_total) | Loss: \(.loss | tostring | .[0:6]) | Avg: \(.avg_loss | tostring | .[0:6])"'
else
    echo "  (No heartbeat yet - training initializing)"
fi
echo ""

# Checkpoint
echo "Checkpoint Status:"
if [ -f artifacts/lvm/models/twotower_fast/epoch4.pt ]; then
    ls -lh artifacts/lvm/models/twotower_fast/epoch4.pt | awk '{print "  ✅ COMPLETE -", $5, "-", $6, $7, $8}'
else
    echo "  ⏳ Not yet created"
fi
echo ""

# Evaluation pipeline
echo "Evaluation Pipeline:"
if ps aux | grep "eval_epoch4_pipeline.sh" | grep -v grep > /dev/null; then
    echo "  ✅ Waiting for checkpoint (will auto-run)"
    if [ -f artifacts/lvm/eval_epoch4/metrics.json ]; then
        echo ""
        echo "  📊 RESULTS AVAILABLE:"
        cat artifacts/lvm/eval_epoch4/metrics.json | jq -r '"    R@5:        \(.["R@5"] * 100 | tostring | .[0:5])%\n    R@10:       \(.["R@10"] * 100 | tostring | .[0:5])%\n    Contain@50: \(.["contain@50"] * 100 | tostring | .[0:5])%\n    MRR:        \(.MRR | tostring | .[0:6])"'
    fi
else
    echo "  ⚠️  Not running"
fi
echo ""

# Quality gates
if [ -f artifacts/lvm/eval_epoch4/metrics.json ]; then
    echo "Quality Gates:"
    R5=$(cat artifacts/lvm/eval_epoch4/metrics.json | jq -r '.["R@5"]')
    MRR=$(cat artifacts/lvm/eval_epoch4/metrics.json | jq -r '.MRR')

    if (( $(echo "$R5 >= 0.30" | bc -l) )); then
        echo "  ✅ R@5 ≥ 30% (PASS)"
    elif (( $(echo "$MRR >= 0.20" | bc -l) )); then
        echo "  ✅ MRR ≥ 0.20 (PASS)"
    else
        echo "  ⚠️  R@5 < 30% and MRR < 0.20 (consider reranker)"
    fi
    echo ""
fi

echo "=========================================="
echo "Quick Commands:"
echo "  Watch heartbeat:  watch -n 5 'cat artifacts/lvm/train_heartbeat.json | jq'"
echo "  View eval log:    tail -f /tmp/eval_epoch4.log"
echo "  View results:     cat artifacts/lvm/eval_epoch4/metrics.json | jq"
echo "=========================================="
