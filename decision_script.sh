#!/bin/bash
# Automated decision based on test results

LOG=$(ls -t logs/twotower_v4_test*.log 2>/dev/null | head -1)

echo "=== MPS TEST DECISION POINT ==="
echo ""

# Check if training completed epoch 1
if grep -q "Recall@500" "$LOG" 2>/dev/null; then
    RECALL=$(grep "Recall@500" "$LOG" | tail -1)
    echo "✅ TEST PASSED: MPS stable without gradient accumulation"
    echo "   $RECALL"
    echo ""
    echo "🎯 RECOMMENDATION: Use safer MPS mode for full run"
    echo ""
    echo "Launch command:"
    echo "  ./launch_v4_mps_safe.sh"
    echo ""
    echo "Benefits:"
    echo "  • 2-3 hour completion (vs 4-5 hours CPU)"
    echo "  • Low gradient accumulation (4 steps vs 32)"
    echo "  • Effective batch ~96-128 (vs 512, but still good)"
    echo "  • FP32 numerics for MPS stability"
    exit 0
else
    # Check if still running
    if pgrep -f "train_twotower_v4.py" > /dev/null; then
        echo "⏳ Test still running, not yet at validation"
        echo "   Current batch: $(grep -oP '\d+/2244' "$LOG" 2>/dev/null | tail -1)"
        echo ""
        echo "Wait 5-10 more minutes for epoch 1 completion"
        exit 2
    else
        echo "❌ TEST FAILED: MPS still crashes even without gradient accumulation"
        echo ""
        echo "🎯 RECOMMENDATION: Switch to CPU for guaranteed completion"
        echo ""
        echo "Launch command:"
        echo "  ./launch_v4_cpu.sh"
        echo ""
        echo "Details:"
        echo "  • 4-5 hour completion time"
        echo "  • Batch 8 × 64 accum = 512 effective"
        echo "  • Guaranteed stable (no MPS bugs)"
        echo "  • Can run overnight"
        exit 1
    fi
fi
