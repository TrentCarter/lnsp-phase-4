#!/bin/bash

# Start All LVM Chat Services
# Port assignments: 9001=AMN, 9002=Transformer, 9003=GRU, 9004=LSTM, 9005=Mamba

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Log directory
LOG_DIR="/tmp/lvm_api_logs"
mkdir -p "$LOG_DIR"

# Function to check if port is in use
port_in_use() {
    lsof -i ":$1" > /dev/null 2>&1
}

# Function to start LVM service
start_lvm_service() {
    local name="$1"
    local port="$2"
    local model_type="$3"
    local model_path="$4"
    local device="$5"

    if port_in_use "$port"; then
        echo -e "${YELLOW}⚠️  ${name} (port ${port}) already running${NC}"
        return 0
    fi

    echo -e "${GREEN}🚀 Starting ${name} on port ${port}...${NC}"

    # Set environment variables for this service
    export LVM_MODEL_TYPE="$model_type"
    export LVM_MODEL_PATH="$model_path"
    export LVM_DEVICE="$device"
    export LVM_PORT="$port"

    # Start service in background
    PYTHONPATH=. ./.venv/bin/python -c "
import sys
sys.path.insert(0, '.')

# Configure model before importing app
from app.api import lvm_inference
lvm_inference.config.model_type = '$model_type'
lvm_inference.config.model_path = '$model_path'
lvm_inference.config.device = '$device'

# Start server
import uvicorn
uvicorn.run(
    'app.api.lvm_inference:app',
    host='127.0.0.1',
    port=$port,
    log_level='info'
)
" > "$LOG_DIR/${name}.log" 2>&1 &

    local pid=$!
    echo "$pid" > "$LOG_DIR/${name}.pid"

    sleep 2

    # Check if process is still running
    if ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ ${name} started (PID: $pid)${NC}"
    else
        echo -e "${RED}❌ ${name} failed to start${NC}"
        cat "$LOG_DIR/${name}.log" | tail -20
    fi
}

# Function to check service health
check_health() {
    local name="$1"
    local url="$2"

    if curl -s "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ ${name}${NC}"
    else
        echo -e "${RED}❌ ${name}${NC}"
    fi
}

echo "============================================"
echo "Starting LVM Chat Services"
echo "============================================"
echo ""

# Detect device (prefer MPS on macOS, then CUDA, fallback to CPU)
if python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    DEVICE="mps"
    echo "✅ Using MPS (Apple Silicon GPU)"
elif python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
    echo "✅ Using CUDA GPU"
else
    DEVICE="cpu"
    echo "⚠️  Using CPU (no GPU detected)"
fi
echo ""

# Find actual model paths (handle different directory structures)
AMN_PATH=$(find artifacts/lvm/models -name "*amn*" -name "*.pt" | head -1)
TRANSFORMER_PATH=$(find artifacts/lvm/models -name "*transformer*" -name "*.pt" | head -1)
GRU_PATH=$(find artifacts/lvm/models -name "*gru*" -name "*.pt" | head -1)
LSTM_PATH=$(find artifacts/lvm/models -name "*lstm*" -name "*.pt" | head -1)

# Fallback paths if specific models not found
[ -z "$AMN_PATH" ] && AMN_PATH="artifacts/lvm/models/amn_v0.pt"
[ -z "$TRANSFORMER_PATH" ] && TRANSFORMER_PATH="artifacts/lvm/models/transformer_v0.pt"
[ -z "$GRU_PATH" ] && GRU_PATH="artifacts/lvm/models/gru_v0.pt"
[ -z "$LSTM_PATH" ] && LSTM_PATH="artifacts/lvm/models/lstm_v0.pt"

# Start services
start_lvm_service "AMN Chat" 9001 "amn" "$AMN_PATH" "$DEVICE"
start_lvm_service "Transformer Chat" 9002 "transformer" "$TRANSFORMER_PATH" "$DEVICE"
start_lvm_service "GRU Chat" 9003 "gru" "$GRU_PATH" "$DEVICE"
start_lvm_service "LSTM Chat" 9004 "lstm" "$LSTM_PATH" "$DEVICE"

echo ""
echo "============================================"
echo "Health Check (waiting 5 seconds...)"
echo "============================================"
sleep 5

check_health "AMN Chat (9001)" "http://localhost:9001/health"
check_health "Transformer Chat (9002)" "http://localhost:9002/health"
check_health "GRU Chat (9003)" "http://localhost:9003/health"
check_health "LSTM Chat (9004)" "http://localhost:9004/health"

echo ""
echo "============================================"
echo "🎉 LVM Chat Services Ready!"
echo "============================================"
echo ""
echo "Chat interfaces:"
echo "  AMN:         http://localhost:9001/chat"
echo "  Transformer: http://localhost:9002/chat"
echo "  GRU:         http://localhost:9003/chat"
echo "  LSTM:        http://localhost:9004/chat"
echo ""
echo "API endpoints:"
echo "  POST /chat  - Chat-style inference (text → text)"
echo "  POST /infer - Low-level inference (vectors → vector)"
echo "  GET /info   - Model information"
echo ""
echo "Logs: $LOG_DIR/"
echo ""
echo "To stop services: ./scripts/stop_lvm_services.sh"
echo "============================================"
