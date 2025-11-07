#!/bin/bash

# Stop Phase 3 PAS Services
# - Provider Router (6103)
# - Gateway (6120)

echo "========================================="
echo "Stopping Phase 3 PAS Services"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to stop service by port
stop_service() {
    local name=$1
    local port=$2

    echo -n "Stopping $name (port $port)... "

    # Find PID listening on port
    local pid=$(lsof -ti :$port 2>/dev/null)

    if [ -z "$pid" ]; then
        echo -e "${YELLOW}not running${NC}"
    else
        kill $pid 2>/dev/null
        # Wait for process to stop (max 5 seconds)
        for i in {1..10}; do
            if ! kill -0 $pid 2>/dev/null; then
                echo -e "${GREEN}stopped${NC}"
                return 0
            fi
            sleep 0.5
        done

        # Force kill if still running
        kill -9 $pid 2>/dev/null
        echo -e "${GREEN}stopped (forced)${NC}"
    fi
}

# Stop services in reverse order
stop_service "Gateway" 6120
stop_service "Provider Router" 6103

echo ""
echo "========================================="
echo -e "${GREEN}Phase 3 services stopped${NC}"
echo "========================================="
