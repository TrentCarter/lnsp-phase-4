#!/bin/bash
# Stop Phase 5 LLM Services
# Ports: 8050 (Llama 3.1 8B), 8051 (TinyLlama), 8052 (TLC Classifier)

set -e

echo "========================================"
echo "Stopping Phase 5 LLM Services"
echo "========================================"
echo ""

STOPPED_COUNT=0

# Stop services by port
for port in 8050 8051 8052; do
    if lsof -ti:$port >/dev/null 2>&1; then
        echo "Stopping service on port $port..."
        PID=$(lsof -ti:$port)
        kill -15 $PID 2>/dev/null || true
        sleep 1

        # Force kill if still running
        if lsof -ti:$port >/dev/null 2>&1; then
            echo "  Force killing PID $PID..."
            kill -9 $PID 2>/dev/null || true
        fi

        STOPPED_COUNT=$((STOPPED_COUNT + 1))
        echo "  ✅ Stopped"
    else
        echo "Port $port: No service running"
    fi
done

echo ""
echo "========================================"

if [ $STOPPED_COUNT -gt 0 ]; then
    echo "Stopped $STOPPED_COUNT service(s)"
else
    echo "No services were running"
fi

echo "========================================"
echo ""

# Verify all stopped
echo "Verifying services stopped..."
ALL_STOPPED=true
for port in 8050 8051 8052; do
    if lsof -ti:$port >/dev/null 2>&1; then
        echo "  ⚠️  Port $port still has a process running"
        ALL_STOPPED=false
    fi
done

if [ "$ALL_STOPPED" = true ]; then
    echo "✅ All Phase 5 services stopped"
    exit 0
else
    echo "❌ Some services still running"
    exit 1
fi
