#!/bin/bash
# Monitor MPS diagnostic test

LOG=$(ls -t logs/twotower_v4_test*.log 2>/dev/null | head -1)

echo "=== MPS TEST MONITOR ==="
echo "Checking every 30 seconds for 10 minutes..."
echo ""

for i in {1..20}; do
    echo "Check $i/20 ($(date +%H:%M:%S)):"
    
    if pgrep -f "train_twotower_v4.py" > /dev/null; then
        PID=$(pgrep -f "train_twotower_v4.py")
        echo "  ✓ Running (PID $PID)"
        
        # Show progress
        PROGRESS=$(grep -oP '\d+/2244' "$LOG" 2>/dev/null | tail -1)
        EPOCH=$(grep -oP 'Epoch \d+/5' "$LOG" 2>/dev/null | tail -1)
        echo "  Progress: $EPOCH, Batch: $PROGRESS"
        
        # Check for validation
        if grep -q "Recall@500" "$LOG" 2>/dev/null; then
            RECALLS=$(grep "Recall@500" "$LOG" | tail -3)
            echo "  Validation results found:"
            echo "$RECALLS" | sed 's/^/    /'
        fi
    else
        echo "  ✗ Process stopped"
        echo ""
        echo "Final log:"
        tail -30 "$LOG" | grep -E "Epoch|Recall|Error|error" | tail -10
        exit 1
    fi
    
    echo ""
    sleep 30
done

echo "✅ TEST PASSED: MPS stable without gradient accumulation"
echo ""
echo "Next step: Launch safer MPS full run with low accumulation"
echo "  ./launch_v4_mps_safe.sh"
