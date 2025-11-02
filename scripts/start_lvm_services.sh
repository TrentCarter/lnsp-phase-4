#!/bin/bash
# Start All LVM Services (Fixed - Using Proper lvm_inference.py)

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

LOG_DIR="/tmp/lvm_api_logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "Starting ALL LVM Services"
echo "============================================"
echo ""

# Detect device
if python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    DEVICE="mps"
    echo "✅ Using MPS (Apple Silicon GPU)"
elif python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
    echo "✅ Using CUDA GPU"
else
    DEVICE="cpu"
    echo "⚠️  Using CPU"
fi
echo ""

# Start Orchestrator Encoder (7001)
echo -e "${GREEN}🚀 Starting Orchestrator Encoder (7001)...${NC}"
./.venv/bin/uvicorn app.api.orchestrator_encoder_server:app \
    --host 127.0.0.1 --port 7001 --log-level error \
    > "$LOG_DIR/orchestrator_encoder.log" 2>&1 &
echo $! > "$LOG_DIR/orchestrator_encoder.pid"
sleep 1

# Start Orchestrator Decoder (7002)
echo -e "${GREEN}🚀 Starting Orchestrator Decoder (7002)...${NC}"
./.venv/bin/uvicorn app.api.orchestrator_decoder_server:app \
    --host 127.0.0.1 --port 7002 --log-level error \
    > "$LOG_DIR/orchestrator_decoder.log" 2>&1 &
echo $! > "$LOG_DIR/orchestrator_decoder.pid"
sleep 1

# Start LVM Evaluation Dashboard (8999)
echo -e "${GREEN}🚀 Starting LVM Evaluation Dashboard (8999)...${NC}"
./.venv/bin/python lvm_eval/test_web_gui.py > "$LOG_DIR/lvm_eval_dashboard.log" 2>&1 &
echo $! > "$LOG_DIR/LVM_Evaluation_Dashboard.pid"
sleep 2

# Start Master Chat (9000)
echo -e "${GREEN}🚀 Starting Master Chat (9000)...${NC}"
./.venv/bin/uvicorn app.api.master_chat:app \
    --host 127.0.0.1 --port 9000 --log-level error \
    > "$LOG_DIR/master_chat.log" 2>&1 &
echo $! > "$LOG_DIR/master_chat.pid"
sleep 1

# Helper function to start LVM service with config injection
start_lvm() {
    local name="$1"
    local port="$2"
    local model_type="$3"
    local model_path="$4"
    local passthrough="${5:-false}"

    echo -e "${GREEN}🚀 Starting $name ($port)...${NC}"

    # Convert bash boolean to Python boolean (case-insensitive)
    local py_passthrough="False"
    local passthrough_lower=$(echo "$passthrough" | tr '[:upper:]' '[:lower:]')
    if [[ "$passthrough_lower" == "true" ]]; then
        py_passthrough="True"
    fi

    # Create Python wrapper that configures and starts service
    cat > "/tmp/start_${port}.py" << EOF
import sys
sys.path.insert(0, '.')

# Import and configure BEFORE FastAPI starts
from app.api import lvm_inference
lvm_inference.config.model_type = "${model_type}"
lvm_inference.config.model_path = "${model_path}"
lvm_inference.config.device = "${DEVICE}"
lvm_inference.config.passthrough = ${py_passthrough}

# Now start uvicorn with configured app
import uvicorn
uvicorn.run(lvm_inference.app, host="127.0.0.1", port=${port}, log_level="error")
EOF

    ./.venv/bin/python "/tmp/start_${port}.py" \
        > "$LOG_DIR/${name// /_}.log" 2>&1 &
    echo $! > "$LOG_DIR/${name// /_}.pid"
    sleep 2

    # Check if started
    if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name started${NC}"
    else
        echo -e "${RED}❌ $name failed${NC}"
        tail -10 "$LOG_DIR/${name// /_}.log"
    fi
}

# Find model paths
AMN_PATH="artifacts/lvm/models/amn_790k_production_20251030_123212/best_model.pt"
GRU_PATH=$(find artifacts/lvm/models -name "*gru*" -name "*.pt" | head -1)
LSTM_PATH=$(find artifacts/lvm/models -name "*lstm*" -name "*.pt" | head -1)
TRANS_BASE="artifacts/lvm/models/transformer_v0.pt"
TRANS_OPT="artifacts/lvm/models/transformer_optimized_v0.pt"
TRANS_584K="artifacts/lvm/models/transformer_584k_stable/best_model.pt"

[ -z "$GRU_PATH" ] && GRU_PATH="artifacts/lvm/models/gru_v0.pt"
[ -z "$LSTM_PATH" ] && LSTM_PATH="artifacts/lvm/models/lstm_v0.pt"

# Start all LVM services
start_lvm "AMN" 9001 "amn" "$AMN_PATH"
start_lvm "Transformer Baseline" 9002 "transformer" "$TRANS_BASE"
start_lvm "GRU" 9003 "gru" "$GRU_PATH"
start_lvm "LSTM" 9004 "lstm" "$LSTM_PATH"
start_lvm "Vec2Text Direct" 9005 "vec2text" "$AMN_PATH" "True"
start_lvm "Transformer Optimized" 9006 "transformer" "$TRANS_OPT"
start_lvm "Transformer Experimental" 9007 "transformer" "$TRANS_584K"

echo ""
echo "============================================"
echo "✅ Health Checks"
echo "============================================"
sleep 3

for port in 7001 7002 8999 9000 9001 9002 9003 9004 9005 9006 9007; do
    if [ "$port" = "8999" ]; then
        # Dashboard uses HTTP without /health endpoint
        if curl -s "http://localhost:${port}/" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Port $port (Dashboard)${NC}"
        else
            echo -e "${RED}❌ Port $port (Dashboard)${NC}"
        fi
    else
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Port $port${NC}"
        else
            echo -e "${RED}❌ Port $port${NC}"
        fi
    fi
done

echo ""
echo "============================================"
echo "🎉 Services Ready!"
echo "============================================"
echo ""
echo "🎯 LVM Evaluation Dashboard: http://localhost:8999"
echo ""
echo "Master Chat:  http://localhost:9000/chat"
echo "AMN:          http://localhost:9001/chat"
echo "Transformer:  http://localhost:9002/chat"
echo "GRU:          http://localhost:9003/chat"
echo "LSTM:         http://localhost:9004/chat"
echo "Vec2Text:     http://localhost:9005/chat"
echo "Trans (Opt):  http://localhost:9006/chat"
echo "Trans (Exp):  http://localhost:9007/chat"
echo ""
echo "Logs: $LOG_DIR/"
echo "============================================"
