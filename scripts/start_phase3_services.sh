#!/bin/bash

# Start Phase 3 PAS Services
# - Provider Router (6103)
# - Gateway (6120)

set -e

echo "========================================="
echo "Starting Phase 3 PAS Services"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Log directory
LOG_DIR="/tmp/pas_logs"
mkdir -p "$LOG_DIR"

# Artifacts directory
mkdir -p artifacts/provider_router
mkdir -p artifacts/costs

# Check if Phase 0+1+2 services are running
echo "Checking Phase 0+1+2 services..."
REQUIRED_PORTS=(6121 6109 6104 6105 6102 6101)
MISSING_SERVICES=()

for port in "${REQUIRED_PORTS[@]}"; do
    if ! curl -s http://localhost:$port/health > /dev/null 2>&1; then
        MISSING_SERVICES+=($port)
    fi
done

if [ ${#MISSING_SERVICES[@]} -ne 0 ]; then
    echo -e "${YELLOW}Warning: Some prerequisite services are not running${NC}"
    echo "Missing services on ports: ${MISSING_SERVICES[@]}"
    echo "Start them with: ./scripts/start_all_pas_services.sh"
    echo ""
fi

# Function to check if a port is in use
port_in_use() {
    lsof -i :$1 > /dev/null 2>&1
}

# Function to start a service
start_service() {
    local name=$1
    local port=$2
    local script=$3
    local log_file=$4

    echo -n "Starting $name (port $port)... "

    if port_in_use $port; then
        echo -e "${YELLOW}already running${NC}"
    else
        # Start service in background
        nohup ./.venv/bin/python $script > "$log_file" 2>&1 &
        local pid=$!

        # Wait for service to start (max 5 seconds)
        for i in {1..10}; do
            if curl -s http://localhost:$port/health > /dev/null 2>&1; then
                echo -e "${GREEN}started (PID $pid)${NC}"
                return 0
            fi
            sleep 0.5
        done

        echo -e "${RED}failed to start${NC}"
        return 1
    fi
}

echo "Starting Phase 3 services..."

# Start Provider Router
start_service \
    "Provider Router" \
    6103 \
    "services/provider_router/provider_router.py" \
    "$LOG_DIR/provider_router.log"

# Start Gateway
start_service \
    "Gateway" \
    6120 \
    "services/gateway/gateway.py" \
    "$LOG_DIR/gateway.log"

echo ""
echo "========================================="
echo -e "${GREEN}Phase 3 services started successfully!${NC}"
echo "========================================="
echo ""
echo "Service Status:"
echo "  Provider Router: http://localhost:6103/health"
echo "  Gateway:         http://localhost:6120/health"
echo ""
echo "API Documentation:"
echo "  Provider Router: http://localhost:6103/docs"
echo "  Gateway:         http://localhost:6120/docs"
echo ""
echo "View logs:"
echo "  tail -f $LOG_DIR/provider_router.log"
echo "  tail -f $LOG_DIR/gateway.log"
echo ""
echo "To stop services:"
echo "  ./scripts/stop_phase3_services.sh"
