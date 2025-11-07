#!/bin/bash
#
# Start Agent Router Service (Port 6119)
#
# This service routes requests to agents based on capabilities.
#

set -e

PORT=6119
SERVICE_NAME="agent_router"
LOG_DIR="/tmp/pas_logs"
ARTIFACTS_DIR="artifacts/router"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$ARTIFACTS_DIR"

# Check if Registry is running (required dependency)
if ! curl -s http://localhost:6121/health > /dev/null 2>&1; then
    echo "❌ Registry service (6121) is not running"
    echo "   Start it first: ./scripts/start_phase0_services.sh"
    exit 1
fi

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    PID=$(lsof -Pi :$PORT -sTCP:LISTEN -t)
    echo "⚠️  Port $PORT already in use by PID $PID"
    echo "   Stop it first: kill $PID"
    exit 1
fi

echo "Starting Agent Router on port $PORT..."

# Get absolute path for artifacts directory
ARTIFACTS_DIR_ABS="$(pwd)/$ARTIFACTS_DIR"

# Start service in background
cd services/router
nohup ../../.venv/bin/uvicorn agent_router:app \
    --host 127.0.0.1 \
    --port $PORT \
    > "$LOG_DIR/${SERVICE_NAME}.log" 2>&1 &

PID=$!
echo $PID > "$ARTIFACTS_DIR_ABS/${SERVICE_NAME}.pid"
cd ../..

# Wait for service to be ready
echo -n "Waiting for service to start..."
for i in {1..30}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo " ✅"
        echo "Agent Router started (PID: $PID)"
        echo "  Health: http://localhost:$PORT/health"
        echo "  API Docs: http://localhost:$PORT/docs"
        echo "  Logs: $LOG_DIR/${SERVICE_NAME}.log"
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo " ❌"
echo "Failed to start Agent Router (timeout)"
echo "Check logs: $LOG_DIR/${SERVICE_NAME}.log"
exit 1
