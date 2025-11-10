#!/usr/bin/env bash
#
# run_stack.sh - Start all P0 services
#
# Services:
#   1. Aider-LCO RPC (port 6130)
#   2. PAS Root (port 6100)
#   3. Gateway (port 6120)
#   4. HMI (port 6101) - optional
#
# Usage:
#   bash scripts/run_stack.sh [--no-hmi]
#

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to repo root
cd "$REPO_ROOT"

# Check if --no-hmi flag is set
START_HMI=true
if [[ "${1:-}" == "--no-hmi" ]]; then
    START_HMI=false
fi

# Store PIDs
declare -a PIDS=()

# Trap to kill all services on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait
    echo -e "${GREEN}All services stopped${NC}"
}
trap cleanup EXIT INT TERM

# Function to start a service
start_service() {
    local name="$1"
    local cmd="$2"
    local port="$3"

    echo -e "${GREEN}[$name]${NC} Starting on port $port..."
    $cmd &
    local pid=$!
    PIDS+=("$pid")

    # Wait 2s and check if process is still alive
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${GREEN}[$name]${NC} ✓ Started (PID: $pid)"
    else
        echo -e "${RED}[$name]${NC} ✗ Failed to start"
        exit 1
    fi
}

# Start services in dependency order

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  Starting P0 Service Stack${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# 1. Aider-LCO RPC (port 6130)
start_service "Aider-LCO" \
    "./.venv/bin/uvicorn services.tools.aider_rpc.app:app --host 127.0.0.1 --port 6130" \
    "6130"

# 2. PAS Root (port 6100)
start_service "PAS Root" \
    "./.venv/bin/uvicorn services.pas.root.app:app --host 127.0.0.1 --port 6100" \
    "6100"

# 3. Gateway (port 6120)
start_service "Gateway" \
    "./.venv/bin/uvicorn services.gateway.app:app --host 127.0.0.1 --port 6120" \
    "6120"

# 4. HMI (port 6101) - optional
if [ "$START_HMI" = true ]; then
    if [ -f "services/webui/hmi_app.py" ]; then
        start_service "HMI" \
            "./.venv/bin/python services/webui/hmi_app.py" \
            "6101"
    else
        echo -e "${YELLOW}[HMI]${NC} Skipping (hmi_app.py not found)"
    fi
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  All Services Running${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}Gateway:${NC}     http://127.0.0.1:6120"
echo -e "${GREEN}PAS Root:${NC}    http://127.0.0.1:6100"
echo -e "${GREEN}Aider-LCO:${NC}   http://127.0.0.1:6130"
if [ "$START_HMI" = true ]; then
    echo -e "${GREEN}HMI:${NC}         http://127.0.0.1:6101"
fi
echo ""
echo -e "${YELLOW}Test with:${NC}"
echo -e "  ./bin/verdict health"
echo -e "  ./bin/verdict send --title 'Test' --goal '...'"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for all services
wait
