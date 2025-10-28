#!/bin/bash
# Monitor Mamba Phase-5 Training Progress

echo "================================================================================
Mamba Phase-5 Training Monitor
================================================================================
"

# Check if processes are running
check_process() {
    local name=$1
    local pid_file=$2

    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "✅ $name (PID: $pid) - RUNNING"
            return 0
        else
            echo "❌ $name (PID: $pid) - STOPPED"
            return 1
        fi
    else
        echo "⚠️  $name - PID file not found"
        return 1
    fi
}

echo "Process Status:"
check_process "Mamba-XL       " "/tmp/mamba_xl.pid"
check_process "Mamba-Sandwich " "/tmp/mamba_sandwich.pid"
check_process "Mamba-GR       " "/tmp/mamba_gr.pid"

echo ""
echo "================================================================================
Latest Training Progress (last 10 lines per model)
================================================================================
"

# Find latest log files
LOG_DIR="./logs/mamba_phase5"
if [ -d "$LOG_DIR" ]; then
    echo "--- Mamba-XL ---"
    latest_xl=$(ls -t "$LOG_DIR"/mamba_xl_*.log 2>/dev/null | head -1)
    if [ -n "$latest_xl" ]; then
        tail -10 "$latest_xl"
    else
        echo "No log file found"
    fi

    echo ""
    echo "--- Mamba-Sandwich ---"
    latest_sandwich=$(ls -t "$LOG_DIR"/mamba_sandwich_*.log 2>/dev/null | head -1)
    if [ -n "$latest_sandwich" ]; then
        tail -10 "$latest_sandwich"
    else
        echo "No log file found"
    fi

    echo ""
    echo "--- Mamba-GR ---"
    latest_gr=$(ls -t "$LOG_DIR"/mamba_gr_*.log 2>/dev/null | head -1)
    if [ -n "$latest_gr" ]; then
        tail -10 "$latest_gr"
    else
        echo "No log file found"
    fi
else
    echo "Log directory not found: $LOG_DIR"
fi

echo ""
echo "================================================================================
Checkpoints Status
================================================================================
"

# Check for model checkpoints
for model in mamba_xl mamba_sandwich mamba_gr; do
    model_dir="./artifacts/lvm/models/$model"
    if [ -d "$model_dir" ]; then
        echo "✅ $model checkpoint exists"
        if [ -f "$model_dir/best.pt" ]; then
            size=$(du -h "$model_dir/best.pt" | cut -f1)
            echo "   Best model: $size"
        fi
        if [ -f "$model_dir/history.json" ]; then
            epochs=$(jq 'length' "$model_dir/history.json" 2>/dev/null || echo "?")
            echo "   Epochs completed: $epochs"
        fi
    else
        echo "⚠️  $model - no checkpoint yet"
    fi
done

echo ""
echo "================================================================================
Usage:
  Watch live: tail -f $LOG_DIR/mamba_xl_*.log
  Re-run monitor: ./scripts/monitor_mamba_training.sh
  Kill all: pkill -f train_mamba_unified
================================================================================
"
