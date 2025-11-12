#!/bin/bash
#
# Start All Manager Services
#
# Starts 7 Manager services (Tier 4):
# - Manager-Code-01, 02, 03 (ports 6141-6143)
# - Manager-Models-01 (port 6144)
# - Manager-Data-01 (port 6145)
# - Manager-DevSecOps-01 (port 6146)
# - Manager-Docs-01 (port 6147)
#

set -e

# Load environment variables from .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Manager definitions: (name, port, service_dir)
MANAGERS=(
    "Manager-Code-01:6141:manager_code_01"
    "Manager-Code-02:6142:manager_code_02"
    "Manager-Code-03:6143:manager_code_03"
    "Manager-Models-01:6144:manager_models_01"
    "Manager-Data-01:6145:manager_data_01"
    "Manager-DevSecOps-01:6146:manager_devsecops_01"
    "Manager-Docs-01:6147:manager_docs_01"
)

echo "=========================================="
echo "Starting All Manager Services (Tier 4)"
echo "=========================================="
echo ""

# Kill existing Manager processes
echo "Stopping existing Managers..."
for port in 6141 6142 6143 6144 6145 6146 6147; do
    pid=$(lsof -ti:$port 2>/dev/null || true)
    if [ ! -z "$pid" ]; then
        echo "  Killing process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null || true
        sleep 0.5
    fi
done

sleep 2

# Start each Manager
echo ""
echo "Starting Managers..."
count=1
total=${#MANAGERS[@]}

for manager_def in "${MANAGERS[@]}"; do
    IFS=':' read -r name port service_dir <<< "$manager_def"

    echo "[$count/$total] Starting $name (port $port)..."

    # Start Manager service in background
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" $VENV_PYTHON -m uvicorn \
        services.pas.${service_dir}.app:app \
        --host 127.0.0.1 \
        --port $port \
        --log-level warning \
        > "artifacts/logs/manager_$(echo $service_dir).log" 2>&1 &

    PID=$!
    echo "  Started $name (PID: $PID)"

    # Brief wait before starting next service
    sleep 1

    ((count++))
done

sleep 3

# Health checks
echo ""
echo "=========================================="
echo "Health Checks"
echo "=========================================="

all_healthy=true
for manager_def in "${MANAGERS[@]}"; do
    IFS=':' read -r name port service_dir <<< "$manager_def"

    response=$(curl -s http://127.0.0.1:$port/health 2>/dev/null || echo "")

    if [ ! -z "$response" ]; then
        agent=$(echo $response | jq -r '.agent' 2>/dev/null || echo "Unknown")
        llm=$(echo $response | jq -r '.llm_model' 2>/dev/null || echo "Unknown")
        echo -e "${GREEN}✓${NC} $name (port $port)"
        echo "  Agent: $agent"
        echo "  LLM: $llm"
    else
        echo -e "${RED}✗${NC} $name (port $port) - Not responding"
        all_healthy=false
    fi
done

echo ""
echo "=========================================="

if [ "$all_healthy" = true ]; then
    echo -e "${GREEN}✓ All Manager services started successfully${NC}"
    echo ""
    echo "Manager services are running:"
    echo "  - Code Lane: Manager-Code-01 (6141), Manager-Code-02 (6142), Manager-Code-03 (6143)"
    echo "  - Models Lane: Manager-Models-01 (6144)"
    echo "  - Data Lane: Manager-Data-01 (6145)"
    echo "  - DevSecOps Lane: Manager-DevSecOps-01 (6146)"
    echo "  - Docs Lane: Manager-Docs-01 (6147)"
    echo ""
    echo "Logs: artifacts/logs/manager_*.log"
else
    echo -e "${RED}✗ Some Manager services failed to start${NC}"
    echo "Check logs in artifacts/logs/manager_*.log"
    exit 1
fi

echo "=========================================="
