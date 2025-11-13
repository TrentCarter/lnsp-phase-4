#!/bin/bash
# Start all PAS services (Phase 0 + Phase 1)

set -e

echo "=========================================="
echo "Starting PAS Services (Phase 0 + Phase 1)"
echo "=========================================="
echo ""

# Start Phase 0
echo "Starting Phase 0 services..."
./scripts/start_phase0_services.sh

echo ""
echo "Waiting 5s for Phase 0 to stabilize..."
sleep 5
echo ""

# Start Phase 1
echo "Starting Phase 1 services..."
./scripts/start_phase1_services.sh

echo ""
echo "Waiting 5s for Phase 1 to stabilize..."
sleep 5
echo ""

# Start Model Pool Manager
echo "Starting Model Pool Manager..."

# Clean up port 8050 if in use
echo "🛑 Checking for existing Model Pool Manager on port 8050..."
PID_8050=$(lsof -ti:8050 2>/dev/null || echo "")
if [ -n "$PID_8050" ]; then
    echo "   Found process(es) on port 8050: $PID_8050"
    for pid in $PID_8050; do
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
    if lsof -ti:8050 >/dev/null 2>&1; then
        echo "❌ ERROR: Port 8050 still in use after cleanup"
        exit 1
    else
        echo "✅ Port 8050 cleared"
    fi
else
    echo "✅ Port 8050 is available"
fi

./.venv/bin/python -m uvicorn services.model_pool_manager.model_pool_manager:app \
  --host 127.0.0.1 --port 8050 \
  > artifacts/logs/model_pool_manager.log 2>&1 &
echo "Model Pool Manager started on port 8050"

echo ""
echo "Waiting 3s for Model Pool Manager to load warmup models..."
sleep 3
echo ""

# Start HMI Service
echo "Starting HMI Service..."

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

./scripts/start_hmi_server.sh &

echo ""
echo "=========================================="
echo "All PAS Services Running!"
echo "=========================================="
echo ""
echo "Phase 0 (Core Infrastructure):"
echo "  Registry:           http://localhost:6121"
echo "  Heartbeat Monitor:  http://localhost:6109"
echo ""
echo "Phase 1 (Management Agents):"
echo "  Resource Manager:   http://localhost:6104"
echo "  Token Governor:     http://localhost:6105"
echo ""
echo "LLM Services:"
echo "  Model Pool Manager: http://localhost:8050"
echo "  Qwen 2.5 Coder:     http://localhost:8051  (warmup)"
echo "  Llama 3.1 8B:       http://localhost:8052  (warmup)"
echo ""
echo "HMI (Web Dashboard):"
echo "  Main Dashboard:     http://localhost:6101/"
echo "  Tree View:          http://localhost:6101/tree"
echo "  Sequencer:          http://localhost:6101/sequencer"
echo "  Actions:            http://localhost:6101/actions"
echo "  Model Pool:         http://localhost:6101/settings  (click Model Pool)"
echo ""
echo "Interactive API Docs:"
echo "  http://localhost:6121/docs  (Registry)"
echo "  http://localhost:6109/docs  (Heartbeat Monitor)"
echo "  http://localhost:6104/docs  (Resource Manager)"
echo "  http://localhost:6105/docs  (Token Governor)"
echo "  http://localhost:8050/docs  (Model Pool Manager)"
echo ""
echo "To stop all services:"
echo "  ./scripts/stop_all_pas_services.sh"
echo ""
