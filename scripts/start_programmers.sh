#!/usr/bin/env bash
#
# Start all 10 programmers from the pool
# Each programmer gets its own port and LLM configuration from programmer_pool.yaml
#
# Usage: ./scripts/start_programmers.sh [start|stop|status|restart]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_PYTHON="./.venv/bin/python"
UVICORN="./.venv/bin/uvicorn"
APP_MODULE="services.tools.aider_rpc.app:app"
LOG_DIR="artifacts/logs/programmers"
PID_DIR="artifacts/pids/programmers"

# Ensure directories exist
mkdir -p "$LOG_DIR" "$PID_DIR"

# Programmer IDs (001-010)
PROGRAMMERS=("001" "002" "003" "004" "005" "006" "007" "008" "009" "010")

start_programmer() {
    local prog_id="$1"
    local pid_file="$PID_DIR/programmer_${prog_id}.pid"
    local log_file="$LOG_DIR/programmer_${prog_id}.log"

    # Calculate port: 6150 + prog_id (001 -> 6151, 010 -> 6160)
    local port=$((6150 + 10#$prog_id))

    # Check if already running
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "⚠️  Programmer-$prog_id already running (PID: $pid)"
            return 0
        else
            # Stale PID file
            rm "$pid_file"
        fi
    fi

    # Start programmer with PROGRAMMER_ID environment variable
    echo "🚀 Starting Programmer-$prog_id on port $port..."
    PROGRAMMER_ID="$prog_id" nohup "$UVICORN" "$APP_MODULE" \
        --host 127.0.0.1 \
        --port "$port" \
        --log-level info \
        > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"

    # Wait a moment to check if it started successfully
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
        echo "✅ Programmer-$prog_id started (PID: $pid)"
    else
        echo "❌ Failed to start Programmer-$prog_id"
        rm "$pid_file"
        return 1
    fi
}

stop_programmer() {
    local prog_id="$1"
    local pid_file="$PID_DIR/programmer_${prog_id}.pid"

    if [ ! -f "$pid_file" ]; then
        echo "⚠️  Programmer-$prog_id not running (no PID file)"
        return 0
    fi

    local pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
        echo "🛑 Stopping Programmer-$prog_id (PID: $pid)..."
        kill "$pid"
        sleep 1

        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            echo "⚠️  Force killing Programmer-$prog_id..."
            kill -9 "$pid" 2>/dev/null || true
        fi

        rm "$pid_file"
        echo "✅ Programmer-$prog_id stopped"
    else
        echo "⚠️  Programmer-$prog_id not running (stale PID file)"
        rm "$pid_file"
    fi
}

status_programmer() {
    local prog_id="$1"
    local pid_file="$PID_DIR/programmer_${prog_id}.pid"

    if [ ! -f "$pid_file" ]; then
        echo "❌ Programmer-$prog_id: Not running"
        return 1
    fi

    local pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
        # Try to get port from config
        local port=$(grep -A 20 "id: \"$prog_id\"" configs/pas/programmer_pool.yaml | grep "port:" | head -1 | awk '{print $2}')
        if [ -n "$port" ]; then
            # Check if port is listening
            if lsof -ti:$port > /dev/null 2>&1; then
                echo "✅ Programmer-$prog_id: Running (PID: $pid, Port: $port)"
            else
                echo "⚠️  Programmer-$prog_id: Running but port $port not listening (PID: $pid)"
            fi
        else
            echo "✅ Programmer-$prog_id: Running (PID: $pid)"
        fi
        return 0
    else
        echo "❌ Programmer-$prog_id: Not running (stale PID file)"
        rm "$pid_file"
        return 1
    fi
}

start_all() {
    echo "=== Starting Programmer Pool (10 instances) ==="
    for prog_id in "${PROGRAMMERS[@]}"; do
        start_programmer "$prog_id"
    done
    echo ""
    echo "=== Programmer Pool Started ==="
    sleep 2
    status_all
}

stop_all() {
    echo "=== Stopping Programmer Pool ==="
    for prog_id in "${PROGRAMMERS[@]}"; do
        stop_programmer "$prog_id"
    done
    echo ""
    echo "=== Programmer Pool Stopped ==="
}

status_all() {
    echo "=== Programmer Pool Status ==="
    local running=0
    local stopped=0

    for prog_id in "${PROGRAMMERS[@]}"; do
        if status_programmer "$prog_id"; then
            ((running++))
        else
            ((stopped++))
        fi
    done

    echo ""
    echo "=== Summary: $running running, $stopped stopped ==="
}

restart_all() {
    echo "=== Restarting Programmer Pool ==="
    stop_all
    sleep 2
    start_all
}

# Main command dispatcher
case "${1:-start}" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    status)
        status_all
        ;;
    restart)
        restart_all
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        echo ""
        echo "Commands:"
        echo "  start    - Start all 10 programmers"
        echo "  stop     - Stop all programmers"
        echo "  status   - Check status of all programmers"
        echo "  restart  - Restart all programmers"
        exit 1
        ;;
esac
