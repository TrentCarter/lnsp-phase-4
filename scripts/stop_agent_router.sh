#!/bin/bash
#
# Stop Agent Router Service (Port 6119)
#

set -e

PORT=6119
SERVICE_NAME="agent_router"
ARTIFACTS_DIR="artifacts/router"
PID_FILE="$ARTIFACTS_DIR/${SERVICE_NAME}.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "⚠️  PID file not found: $PID_FILE"
    echo "   Service may not be running, or was started manually"

    # Try to find process by port
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        PID=$(lsof -Pi :$PORT -sTCP:LISTEN -t)
        echo "   Found process on port $PORT (PID: $PID)"
        kill $PID
        echo "✅ Stopped Agent Router (PID: $PID)"
        exit 0
    else
        echo "   No process found on port $PORT"
        exit 0
    fi
fi

PID=$(cat "$PID_FILE")

if ps -p $PID > /dev/null 2>&1; then
    kill $PID
    echo "✅ Stopped Agent Router (PID: $PID)"
    rm "$PID_FILE"
else
    echo "⚠️  Process $PID not running"
    rm "$PID_FILE"
fi
