#!/bin/bash
# Start Phase 2 PAS Services (Event Stream + HMI)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Starting Phase 2 PAS Services"
echo "========================================="

# Create log directory if it doesn't exist
mkdir -p /tmp/pas_logs

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to start a service
start_service() {
    local name=$1
    local script=$2
    local port=$3
    local logfile=$4

    echo -n "Starting $name (port $port)... "

    if check_port $port; then
        echo -e "${YELLOW}already running${NC}"
        return 0
    fi

    # Start service in background
    nohup $script > "$logfile" 2>&1 &
    local pid=$!

    # Wait up to 5 seconds for service to start
    for i in {1..10}; do
        if check_port $port; then
            echo -e "${GREEN}started (PID: $pid)${NC}"
            return 0
        fi
        sleep 0.5
    done

    echo -e "${RED}failed to start${NC}"
    return 1
}

# Verify Python virtual environment
if [ ! -f ".venv/bin/python" ]; then
    echo -e "${RED}Error: Python virtual environment not found${NC}"
    echo "Please run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Check if Phase 0+1 services are running
echo ""
echo "Checking Phase 0+1 services..."
PHASE01_OK=true

for port in 6121 6109 6104 6105; do
    if ! check_port $port; then
        echo -e "${YELLOW}Warning: Port $port not responding${NC}"
        PHASE01_OK=false
    fi
done

if [ "$PHASE01_OK" = false ]; then
    echo -e "${YELLOW}Some Phase 0+1 services are not running${NC}"
    echo "Consider starting them with: ./scripts/start_all_pas_services.sh"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Starting Phase 2 services..."

# Start Event Stream (port 6102)
start_service \
    "Event Stream" \
    "./.venv/bin/python services/event_stream/event_stream.py" \
    6102 \
    "/tmp/pas_logs/event_stream.log"

# Start Flask HMI (port 6101)
start_service \
    "Flask HMI" \
    "./.venv/bin/python services/webui/hmi_app.py" \
    6101 \
    "/tmp/pas_logs/hmi_app.log"

echo ""
echo "========================================="
echo -e "${GREEN}Phase 2 services started successfully!${NC}"
echo "========================================="
echo ""
echo "Service Status:"
echo "  Event Stream:  http://localhost:6102/health"
echo "  Flask HMI:     http://localhost:6101"
echo ""
echo "Open HMI Dashboard:"
echo "  open http://localhost:6101"
echo ""
echo "View logs:"
echo "  tail -f /tmp/pas_logs/event_stream.log"
echo "  tail -f /tmp/pas_logs/hmi_app.log"
echo ""
echo "To stop services:"
echo "  ./scripts/stop_phase2_services.sh"
echo ""
