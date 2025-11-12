#!/bin/bash
#
# Start All Programmers Script
#
# Starts all 10 Programmer FastAPI services (ports 6151-6160) with:
# - Health checks
# - Log redirection
# - Graceful shutdown of existing processes
# - Environment variable loading (.env)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
SERVICES_DIR="$PROJECT_ROOT/services/tools"
LOGS_DIR="$PROJECT_ROOT/artifacts/logs"
PIDS_FILE="$PROJECT_ROOT/artifacts/pids/programmers.pids"

# Load environment variables from .env if exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${BLUE}Loading environment from .env${NC}"
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
else
    echo -e "${YELLOW}Warning: .env not found, API keys may not be available${NC}"
fi

# Create directories
mkdir -p "$LOGS_DIR"
mkdir -p "$(dirname "$PIDS_FILE")"

# Programmer services configuration
declare -a PROGRAMMERS=(
    "001:6151:Programmer-001"
    "002:6152:Programmer-002"
    "003:6153:Programmer-003"
    "004:6154:Programmer-004"
    "005:6155:Programmer-005"
    "006:6156:Programmer-006"
    "007:6157:Programmer-007"
    "008:6158:Programmer-008"
    "009:6159:Programmer-009"
    "010:6160:Programmer-010"
)

# Function to check if port is in use
is_port_in_use() {
    local port=$1
    lsof -ti:$port > /dev/null 2>&1
}

# Function to kill process on port
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port 2>/dev/null || true)

    if [ -n "$pid" ]; then
        echo -e "  ${YELLOW}Killing existing process on port $port (PID: $pid)${NC}"
        kill -9 $pid 2>/dev/null || true
        sleep 0.5
    fi
}

# Function to start a Programmer service
start_programmer() {
    local num=$1
    local port=$2
    local name=$3

    local service_dir="$SERVICES_DIR/programmer_$num"
    local log_file="$LOGS_DIR/programmer_$num.log"

    # Kill existing process on port
    kill_port $port

    # Start service
    echo -e "  ${BLUE}Starting $name on port $port...${NC}"

    cd "$service_dir"
    nohup "$VENV_PYTHON" app.py > "$log_file" 2>&1 &
    local pid=$!

    # Save PID
    echo "$pid" >> "$PIDS_FILE"

    # Wait for service to start
    local attempts=0
    local max_attempts=30

    while [ $attempts -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo -e "  ${GREEN}✓ $name started (PID: $pid, Port: $port)${NC}"

            # Get health status
            local health_json=$(curl -s "http://localhost:$port/health")
            local agent_id=$(echo "$health_json" | jq -r '.agent // "unknown"')
            local llm_mode=$(echo "$health_json" | jq -r '.llm_mode // "unknown"')

            echo -e "    Agent: $agent_id, LLM: $llm_mode"
            return 0
        fi

        sleep 0.5
        attempts=$((attempts + 1))
    done

    echo -e "  ${RED}✗ $name failed to start (check $log_file)${NC}"
    return 1
}

# Main
echo "============================================================"
echo "Starting All Programmer Services"
echo "============================================================"

# Clear old PIDs file
> "$PIDS_FILE"

# Start all Programmers
echo ""
for programmer in "${PROGRAMMERS[@]}"; do
    IFS=':' read -r num port name <<< "$programmer"
    start_programmer "$num" "$port" "$name"
done

echo ""
echo "============================================================"
echo "Programmer Services Summary"
echo "============================================================"

# Check all services
all_healthy=true
for programmer in "${PROGRAMMERS[@]}"; do
    IFS=':' read -r num port name <<< "$programmer"

    if is_port_in_use $port; then
        echo -e "${GREEN}✓${NC} $name (port $port)"
    else
        echo -e "${RED}✗${NC} $name (port $port)"
        all_healthy=false
    fi
done

echo ""

if $all_healthy; then
    echo -e "${GREEN}✓ All 10 Programmer services are running${NC}"
else
    echo -e "${YELLOW}⚠ Some Programmer services failed to start${NC}"
    echo -e "  Check logs in: $LOGS_DIR"
fi

echo ""
echo "Logs: $LOGS_DIR/programmer_*.log"
echo "PIDs: $PIDS_FILE"
echo ""
echo "To stop all Programmers: bash scripts/stop_all_programmers.sh"
echo ""
