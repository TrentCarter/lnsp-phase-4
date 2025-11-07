#!/bin/bash
# Start Phase 5 LLM Services
# Ports: 8050 (Llama 3.1 8B), 8051 (TinyLlama), 8052 (TLC Classifier)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
LOG_DIR="/tmp"

echo "========================================"
echo "Starting Phase 5 LLM Services"
echo "========================================"
echo ""

# Check if Python venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Python venv not found at $VENV_PYTHON"
    echo "   Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Check if Ollama is running
echo "Checking Ollama service..."
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ Ollama not running on port 11434"
    echo "   Start with: ollama serve"
    echo ""
    echo "   Or run Ollama in background:"
    echo "   ollama serve > /tmp/ollama.log 2>&1 &"
    exit 1
fi
echo "✅ Ollama running on port 11434"
echo ""

# Check if required models are available
echo "Checking Ollama models..."
MODELS_JSON=$(curl -s http://localhost:11434/api/tags)

if ! echo "$MODELS_JSON" | grep -q "llama3.1:8b"; then
    echo "⚠️  Model 'llama3.1:8b' not found"
    echo "   Install with: ollama pull llama3.1:8b"
fi

if ! echo "$MODELS_JSON" | grep -q "tinyllama:1.1b"; then
    echo "⚠️  Model 'tinyllama:1.1b' not found"
    echo "   Install with: ollama pull tinyllama:1.1b"
fi

echo "✅ Ollama models checked"
echo ""

# Stop existing services on these ports
echo "Stopping any existing services on ports 8050-8052..."
for port in 8050 8051 8052; do
    if lsof -ti:$port >/dev/null 2>&1; then
        echo "  Stopping service on port $port..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
done
echo "✅ Ports cleared"
echo ""

# Start Llama 3.1 8B Service (Port 8050)
echo "Starting Llama 3.1 8B Service (Port 8050)..."
PYTHONPATH="$PROJECT_ROOT" $VENV_PYTHON \
  -m uvicorn services.llm.llama31_8b_service:app \
  --host 127.0.0.1 --port 8050 \
  > "$LOG_DIR/llm_llama31_8b.log" 2>&1 &
LLM_LLAMA31_PID=$!
echo "  PID: $LLM_LLAMA31_PID"
echo "  Log: $LOG_DIR/llm_llama31_8b.log"

# Start TinyLlama Service (Port 8051)
echo "Starting TinyLlama Service (Port 8051)..."
PYTHONPATH="$PROJECT_ROOT" $VENV_PYTHON \
  -m uvicorn services.llm.tinyllama_service:app \
  --host 127.0.0.1 --port 8051 \
  > "$LOG_DIR/llm_tinyllama.log" 2>&1 &
LLM_TINYLLAMA_PID=$!
echo "  PID: $LLM_TINYLLAMA_PID"
echo "  Log: $LOG_DIR/llm_tinyllama.log"

# Start TLC Domain Classifier (Port 8052)
echo "Starting TLC Domain Classifier (Port 8052)..."
PYTHONPATH="$PROJECT_ROOT" $VENV_PYTHON \
  -m uvicorn services.llm.tlc_classifier_service:app \
  --host 127.0.0.1 --port 8052 \
  > "$LOG_DIR/llm_tlc_classifier.log" 2>&1 &
LLM_TLC_PID=$!
echo "  PID: $LLM_TLC_PID"
echo "  Log: $LOG_DIR/llm_tlc_classifier.log"

echo ""
echo "Waiting for services to start..."
sleep 5

# Health check
echo ""
echo "========================================"
echo "Health Check"
echo "========================================"
ALL_HEALTHY=true

for port in 8050 8051 8052; do
    echo -n "Port $port: "
    if curl -s http://localhost:$port/health >/dev/null 2>&1; then
        STATUS=$(curl -s http://localhost:$port/health | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")
        if [ "$STATUS" = "healthy" ]; then
            echo "✅ Healthy"
        else
            echo "⚠️  $STATUS"
            ALL_HEALTHY=false
        fi
    else
        echo "❌ Not responding"
        ALL_HEALTHY=false
    fi
done

echo ""
echo "========================================"
echo "Phase 5 LLM Services Started"
echo "========================================"
echo ""
echo "Service URLs:"
echo "  Llama 3.1 8B:     http://localhost:8050"
echo "    - Docs:         http://localhost:8050/docs"
echo "    - Health:       curl http://localhost:8050/health"
echo "    - Info:         curl http://localhost:8050/info"
echo ""
echo "  TinyLlama:        http://localhost:8051"
echo "    - Docs:         http://localhost:8051/docs"
echo "    - Health:       curl http://localhost:8051/health"
echo "    - Info:         curl http://localhost:8051/info"
echo ""
echo "  TLC Classifier:   http://localhost:8052"
echo "    - Docs:         http://localhost:8052/docs"
echo "    - Health:       curl http://localhost:8052/health"
echo "    - Info:         curl http://localhost:8052/info"
echo ""
echo "Logs:"
echo "  tail -f $LOG_DIR/llm_llama31_8b.log"
echo "  tail -f $LOG_DIR/llm_tinyllama.log"
echo "  tail -f $LOG_DIR/llm_tlc_classifier.log"
echo ""
echo "Stop services:"
echo "  ./scripts/stop_phase5_llm_services.sh"
echo ""

if [ "$ALL_HEALTHY" = true ]; then
    echo "✅ All services healthy!"
    exit 0
else
    echo "⚠️  Some services not healthy - check logs"
    exit 1
fi
