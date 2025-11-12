#!/usr/bin/env bash
#
# Start Multi-Tier PAS - Full Build
#
# Starts all PAS services in correct order:
# 1. Architect (port 6110)
# 2. Directors: Code (6111), Models (6112), Data (6113), DevSecOps (6114), Docs (6115)
# 3. PAS Root (port 6100)
# 4. Gateway (port 6120)
#
# Usage:
#   bash scripts/start_multitier_pas.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Starting Multi-Tier PAS ==="
echo ""

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: No venv found"
    exit 1
fi

# Check if services are already running
echo "Checking for running services..."
RUNNING_PORTS=""

for port in 6110 6111 6112 6113 6114 6115 6100 6120; do
    if lsof -ti:$port > /dev/null 2>&1; then
        RUNNING_PORTS="$RUNNING_PORTS $port"
    fi
done

if [ ! -z "$RUNNING_PORTS" ]; then
    echo -e "${YELLOW}Warning: Services already running on ports:$RUNNING_PORTS${NC}"
    echo "Run './scripts/stop_multitier_pas.sh' first to stop them"
    exit 1
fi

# Set LLM configuration
export ARCHITECT_LLM_PROVIDER="${ARCHITECT_LLM_PROVIDER:-anthropic}"
export ARCHITECT_LLM="${ARCHITECT_LLM:-claude-sonnet-4-5-20250929}"

export DIR_CODE_LLM_PROVIDER="${DIR_CODE_LLM_PROVIDER:-google}"
export DIR_CODE_LLM="${DIR_CODE_LLM:-gemini-2.5-flash}"

export DIR_MODELS_LLM_PROVIDER="${DIR_MODELS_LLM_PROVIDER:-anthropic}"
export DIR_MODELS_LLM="${DIR_MODELS_LLM:-claude-sonnet-4-5-20250929}"

export DIR_DATA_LLM_PROVIDER="${DIR_DATA_LLM_PROVIDER:-anthropic}"
export DIR_DATA_LLM="${DIR_DATA_LLM:-claude-sonnet-4-5-20250929}"

export DIR_DEVSECOPS_LLM_PROVIDER="${DIR_DEVSECOPS_LLM_PROVIDER:-google}"
export DIR_DEVSECOPS_LLM="${DIR_DEVSECOPS_LLM:-gemini-2.5-flash}"

export DIR_DOCS_LLM_PROVIDER="${DIR_DOCS_LLM_PROVIDER:-anthropic}"
export DIR_DOCS_LLM="${DIR_DOCS_LLM:-claude-sonnet-4-5-20250929}"

# Set service URLs
export ARCHITECT_URL="http://127.0.0.1:6110"
export DIR_CODE_URL="http://127.0.0.1:6111"
export DIR_MODELS_URL="http://127.0.0.1:6112"
export DIR_DATA_URL="http://127.0.0.1:6113"
export DIR_DEVSECOPS_URL="http://127.0.0.1:6114"
export DIR_DOCS_URL="http://127.0.0.1:6115"

# Create log directory
mkdir -p logs/pas

echo ""
echo "=== Starting Services ==="

# 1. Start Architect
echo -e "${GREEN}[1/8]${NC} Starting Architect (port 6110)..."
python -m uvicorn services.pas.architect.app:app \
    --host 127.0.0.1 --port 6110 \
    > logs/pas/architect.log 2>&1 &
ARCHITECT_PID=$!
sleep 2

# 2. Start Director-Code
echo -e "${GREEN}[2/8]${NC} Starting Director-Code (port 6111)..."
python -m uvicorn services.pas.director_code.app:app \
    --host 127.0.0.1 --port 6111 \
    > logs/pas/director_code.log 2>&1 &
DIRECTOR_CODE_PID=$!
sleep 1

# 3. Start Director-Models
echo -e "${GREEN}[3/8]${NC} Starting Director-Models (port 6112)..."
python -m uvicorn services.pas.director_models.app:app \
    --host 127.0.0.1 --port 6112 \
    > logs/pas/director_models.log 2>&1 &
DIRECTOR_MODELS_PID=$!
sleep 1

# 4. Start Director-Data
echo -e "${GREEN}[4/8]${NC} Starting Director-Data (port 6113)..."
python -m uvicorn services.pas.director_data.app:app \
    --host 127.0.0.1 --port 6113 \
    > logs/pas/director_data.log 2>&1 &
DIRECTOR_DATA_PID=$!
sleep 1

# 5. Start Director-DevSecOps
echo -e "${GREEN}[5/8]${NC} Starting Director-DevSecOps (port 6114)..."
python -m uvicorn services.pas.director_devsecops.app:app \
    --host 127.0.0.1 --port 6114 \
    > logs/pas/director_devsecops.log 2>&1 &
DIRECTOR_DEVSECOPS_PID=$!
sleep 1

# 6. Start Director-Docs
echo -e "${GREEN}[6/8]${NC} Starting Director-Docs (port 6115)..."
python -m uvicorn services.pas.director_docs.app:app \
    --host 127.0.0.1 --port 6115 \
    > logs/pas/director_docs.log 2>&1 &
DIRECTOR_DOCS_PID=$!
sleep 1

# 7. Start PAS Root
echo -e "${GREEN}[7/8]${NC} Starting PAS Root (port 6100)..."
python -m uvicorn services.pas.root.app:app \
    --host 127.0.0.1 --port 6100 \
    > logs/pas/root.log 2>&1 &
PAS_ROOT_PID=$!
sleep 2

# 8. Start Gateway
echo -e "${GREEN}[8/8]${NC} Starting Gateway (port 6120)..."
python -m uvicorn services.gateway.app:app \
    --host 127.0.0.1 --port 6120 \
    > logs/pas/gateway.log 2>&1 &
GATEWAY_PID=$!
sleep 2

echo ""
echo "=== Health Checks ==="

# Health check function
check_health() {
    local service=$1
    local port=$2
    local max_retries=10
    local retry=0

    while [ $retry -lt $max_retries ]; do
        if curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} $service (port $port)"
            return 0
        fi
        retry=$((retry + 1))
        sleep 1
    done

    echo -e "${YELLOW}✗${NC} $service (port $port) - NOT RESPONDING"
    return 1
}

ALL_HEALTHY=true

check_health "Architect" 6110 || ALL_HEALTHY=false
check_health "Director-Code" 6111 || ALL_HEALTHY=false
check_health "Director-Models" 6112 || ALL_HEALTHY=false
check_health "Director-Data" 6113 || ALL_HEALTHY=false
check_health "Director-DevSecOps" 6114 || ALL_HEALTHY=false
check_health "Director-Docs" 6115 || ALL_HEALTHY=false
check_health "PAS Root" 6100 || ALL_HEALTHY=false
check_health "Gateway" 6120 || ALL_HEALTHY=false

echo ""

if [ "$ALL_HEALTHY" = true ]; then
    echo -e "${GREEN}=== Multi-Tier PAS Started Successfully ===${NC}"
    echo ""
    echo "Services:"
    echo "  Architect:       http://127.0.0.1:6110"
    echo "  Director-Code:   http://127.0.0.1:6111"
    echo "  Director-Models: http://127.0.0.1:6112"
    echo "  Director-Data:   http://127.0.0.1:6113"
    echo "  Director-DevSecOps: http://127.0.0.1:6114"
    echo "  Director-Docs:   http://127.0.0.1:6115"
    echo "  PAS Root:        http://127.0.0.1:6100"
    echo "  Gateway:         http://127.0.0.1:6120"
    echo ""
    echo "Logs: logs/pas/"
    echo ""
    echo "To stop: ./scripts/stop_multitier_pas.sh"
else
    echo -e "${YELLOW}=== Some services failed to start ===${NC}"
    echo "Check logs in logs/pas/"
    exit 1
fi
