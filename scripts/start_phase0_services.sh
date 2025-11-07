#!/bin/bash
# Start Phase 0 services: Registry + Heartbeat Monitor

set -e

REPO_ROOT="/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4"
cd "$REPO_ROOT"

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please create .venv first."
    exit 1
fi

source .venv/bin/activate

# Kill existing services on ports 6121 and 6109
echo "🛑 Stopping existing services..."
lsof -ti:6121 | xargs kill -9 2>/dev/null || true
lsof -ti:6109 | xargs kill -9 2>/dev/null || true
sleep 2

# Create log directory
mkdir -p /tmp/pas_logs

# Start Registry (6121)
echo "🚀 Starting Registry service on port 6121..."
./.venv/bin/uvicorn services.registry.registry_service:app \
    --host 127.0.0.1 \
    --port 6121 \
    --log-level info \
    > /tmp/pas_logs/registry.log 2>&1 &

REGISTRY_PID=$!
echo "   Registry PID: $REGISTRY_PID"

# Wait for Registry to start
sleep 3

# Check Registry health
if curl -s http://localhost:6121/health | grep -q "ok"; then
    echo "✓ Registry service healthy"
else
    echo "❌ Registry service failed to start"
    cat /tmp/pas_logs/registry.log
    exit 1
fi

# Start Heartbeat Monitor (6109)
echo "🚀 Starting Heartbeat Monitor on port 6109..."
./.venv/bin/uvicorn services.heartbeat_monitor.heartbeat_monitor:app \
    --host 127.0.0.1 \
    --port 6109 \
    --log-level info \
    > /tmp/pas_logs/heartbeat_monitor.log 2>&1 &

MONITOR_PID=$!
echo "   Heartbeat Monitor PID: $MONITOR_PID"

# Wait for Heartbeat Monitor to start
sleep 3

# Check Heartbeat Monitor health
if curl -s http://localhost:6109/health | grep -q "ok"; then
    echo "✓ Heartbeat Monitor service healthy"
else
    echo "❌ Heartbeat Monitor failed to start"
    cat /tmp/pas_logs/heartbeat_monitor.log
    exit 1
fi

echo ""
echo "✅ Phase 0 services started successfully!"
echo ""
echo "Registry:          http://localhost:6121"
echo "Heartbeat Monitor: http://localhost:6109"
echo ""
echo "Logs:"
echo "  Registry:          tail -f /tmp/pas_logs/registry.log"
echo "  Heartbeat Monitor: tail -f /tmp/pas_logs/heartbeat_monitor.log"
echo ""
echo "To stop services:"
echo "  ./scripts/stop_phase0_services.sh"
echo ""
