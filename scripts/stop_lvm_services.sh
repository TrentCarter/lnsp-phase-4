#!/bin/bash

# Stop All LVM Chat Services + Orchestrator Encode/Decode

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

LOG_DIR="/tmp/lvm_api_logs"

echo "============================================"
echo "Stopping LVM Chat Services"
echo "============================================"
echo ""

# Function to stop service
stop_service() {
    local name="$1"
    local pid_file="$LOG_DIR/${name}.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "🛑 Stopping ${name} (PID: $pid)..."
            kill "$pid"
            sleep 1

            # Force kill if still running
            if ps -p "$pid" > /dev/null 2>&1; then
                echo -e "${RED}   Force killing${NC}..."
                kill -9 "$pid" 2>/dev/null
            fi

            echo -e "${GREEN}✅ ${name} stopped${NC}"
        else
            echo -e "⏭️  ${name} not running"
        fi
        rm "$pid_file"
    else
        echo -e "⏭️  ${name} not found"
    fi
}

stop_service "LVM Evaluation Dashboard"
stop_service "Orchestrator Encoder"
stop_service "Orchestrator Decoder"
stop_service "Master Chat"
stop_service "AMN"
stop_service "Transformer Baseline"
stop_service "GRU"
stop_service "LSTM"
stop_service "Vec2Text Direct"
stop_service "Transformer Optimized"
stop_service "Transformer Experimental"

# Also kill any remaining processes on all ports (orchestrator + LVM + dashboard)
echo ""
echo "Cleaning up any remaining processes..."
for port in 7001 7002 8999 9000 9001 9002 9003 9004 9005 9006 9007; do
    pid=$(lsof -t -i:$port 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "  Killing process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null
    fi
done

echo ""
echo "============================================"
echo "✅ All LVM services stopped"
echo "============================================"
