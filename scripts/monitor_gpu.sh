#!/bin/bash
#
# Monitor GPU Usage During Training
# ==================================
#
# Run this in a separate terminal while training to monitor GPU utilization.
#
# Usage:
#   ./scripts/monitor_gpu.sh

echo "============================================"
echo "GPU Monitoring (Press Ctrl+C to stop)"
echo "============================================"
echo ""

# Check if training process is running
TRAINING_PIDS=$(ps aux | grep "train_with_5cat_validation\|train_unified.py" | grep -v grep | awk '{print $2}')

if [ -z "$TRAINING_PIDS" ]; then
    echo "⚠️  No training process detected"
    echo "   Start training first: ./scripts/train_with_5cat_validation.sh transformer"
    echo ""
fi

# Monitor loop
while true; do
    clear
    echo "============================================"
    echo "GPU Monitoring - $(date +%H:%M:%S)"
    echo "============================================"
    echo ""

    # Show Python processes using GPU
    echo "📊 Python Processes:"
    ps aux | grep -E "python.*train|python.*lvm" | grep -v grep | awk '{printf "   PID: %-7s CPU: %3s%%  MEM: %4s%%  CMD: %s\n", $2, $3, $4, substr($0, index($0,$11))}'
    echo ""

    # Show GPU metrics via system profiler (macOS)
    echo "🖥️  GPU Activity:"

    # Get GPU power usage (if available)
    if command -v powermetrics &> /dev/null && [ "$EUID" -eq 0 ]; then
        sudo powermetrics --samplers gpu_power -i 1000 -n 1 2>/dev/null | grep -A 5 "GPU Power"
    else
        # Fallback: Check if MPS is active via PyTorch
        ./.venv/bin/python3 -c "
import torch
if torch.backends.mps.is_available():
    try:
        # Create a test tensor on MPS
        x = torch.randn(100, 100, device='mps')
        print('   ✅ MPS device is active')
    except:
        print('   ⚠️  MPS available but not in use')
else:
    print('   ❌ MPS not available')
" 2>/dev/null
    fi

    # Check Activity Monitor for GPU usage
    echo ""
    echo "💻 System Resources:"
    top -l 1 | grep -E "PhysMem|CPU usage"

    echo ""
    echo "📈 Training Progress (if running):"
    # Try to tail the latest training log
    LATEST_LOG=$(find artifacts/lvm/models/*_5cat_*/training.log 2>/dev/null | tail -1)
    if [ -n "$LATEST_LOG" ]; then
        tail -3 "$LATEST_LOG" 2>/dev/null | sed 's/^/   /'
    else
        echo "   No training log found"
    fi

    echo ""
    echo "Press Ctrl+C to stop monitoring..."
    sleep 5
done
