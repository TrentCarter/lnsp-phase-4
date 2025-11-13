#!/bin/bash
#
# Start HMI Server (Flask Web Dashboard)
# Port 6101
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Starting HMI Server (Flask Web Dashboard)..."
echo "Port: 6101"
echo ""
echo "Access the dashboard at:"
echo "  Main Dashboard:  http://localhost:6101/"
echo "  Tree View:       http://localhost:6101/tree"
echo "  Sequencer:       http://localhost:6101/sequencer"
echo "  Actions:         http://localhost:6101/actions"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Clean up port 6101 if in use
echo "🛑 Checking for existing HMI service on port 6101..."
PID_6101=$(lsof -ti:6101 2>/dev/null || echo "")
if [ -n "$PID_6101" ]; then
    echo "   Found process(es) on port 6101: $PID_6101"
    for pid in $PID_6101; do
        echo "   Attempting graceful shutdown of PID $pid..."
        kill -TERM "$pid" 2>/dev/null || true
        sleep 2
        if kill -0 "$pid" 2>/dev/null; then
            echo "   Force killing PID $pid..."
            kill -KILL "$pid" 2>/dev/null || true
            sleep 1
        fi
    done
    
    # Final verification
    sleep 1
    if lsof -ti:6101 >/dev/null 2>&1; then
        echo "❌ ERROR: Port 6101 still in use after cleanup"
        exit 1
    else
        echo "✅ Port 6101 cleared"
    fi
else
    echo "✅ Port 6101 is available"
fi

cd "$PROJECT_ROOT"
exec ./.venv/bin/python services/webui/hmi_app.py
