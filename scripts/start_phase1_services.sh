#!/bin/bash
# Start Phase 1 services: Resource Manager + Token Governor
# Requires Phase 0 services (Registry + Heartbeat Monitor) to be running

set -e

REPO_ROOT="/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4"
cd "$REPO_ROOT"

# Check Phase 0 services are running
if ! curl -s http://localhost:6121/health >/dev/null 2>&1; then
    echo "❌ Registry (Phase 0) not running. Please start Phase 0 first:"
    echo "   ./scripts/start_phase0_services.sh"
    exit 1
fi

if ! curl -s http://localhost:6109/health >/dev/null 2>&1; then
    echo "❌ Heartbeat Monitor (Phase 0) not running. Please start Phase 0 first:"
    echo "   ./scripts/start_phase0_services.sh"
    exit 1
fi

echo "✓ Phase 0 services confirmed running"
echo ""

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please create .venv first."
    exit 1
fi

source .venv/bin/activate

# Kill existing Phase 1 services
echo "🛑 Stopping existing Phase 1 services..."
lsof -ti:6104 | xargs kill -9 2>/dev/null || true
lsof -ti:6105 | xargs kill -9 2>/dev/null || true
sleep 2

# Create log directory
mkdir -p /tmp/pas_logs

# Start Resource Manager (6104)
echo "🚀 Starting Resource Manager on port 6104..."
./.venv/bin/uvicorn services.resource_manager.resource_manager:app \
    --host 127.0.0.1 \
    --port 6104 \
    --log-level info \
    > /tmp/pas_logs/resource_manager.log 2>&1 &

RM_PID=$!
echo "   Resource Manager PID: $RM_PID"

# Wait for Resource Manager to start
sleep 3

# Check Resource Manager health
if curl -s http://localhost:6104/health | grep -q "ok"; then
    echo "✓ Resource Manager service healthy"
else
    echo "❌ Resource Manager failed to start"
    cat /tmp/pas_logs/resource_manager.log
    exit 1
fi

# Start Token Governor (6105)
echo "🚀 Starting Token Governor on port 6105..."
./.venv/bin/uvicorn services.token_governor.token_governor:app \
    --host 127.0.0.1 \
    --port 6105 \
    --log-level info \
    > /tmp/pas_logs/token_governor.log 2>&1 &

TG_PID=$!
echo "   Token Governor PID: $TG_PID"

# Wait for Token Governor to start
sleep 3

# Check Token Governor health
if curl -s http://localhost:6105/health | grep -q "ok"; then
    echo "✓ Token Governor service healthy"
else
    echo "❌ Token Governor failed to start"
    cat /tmp/pas_logs/token_governor.log
    exit 1
fi

echo ""
echo "✅ Phase 1 services started successfully!"
echo ""
echo "Resource Manager:  http://localhost:6104"
echo "Token Governor:    http://localhost:6105"
echo ""
echo "Logs:"
echo "  Resource Manager:  tail -f /tmp/pas_logs/resource_manager.log"
echo "  Token Governor:    tail -f /tmp/pas_logs/token_governor.log"
echo ""
echo "All services running:"
echo "  Phase 0: Registry (6121), Heartbeat Monitor (6109)"
echo "  Phase 1: Resource Manager (6104), Token Governor (6105)"
echo ""
echo "To stop Phase 1 services:"
echo "  ./scripts/stop_phase1_services.sh"
echo ""
