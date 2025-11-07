#!/bin/bash
# Stop Phase 2 PAS Services (Event Stream + HMI)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Stopping Phase 2 PAS Services"
echo "========================================="

# Function to stop service on a port
stop_service() {
    local name=$1
    local port=$2

    echo -n "Stopping $name (port $port)... "

    # Find PIDs using the port
    PIDS=$(lsof -ti :$port 2>/dev/null || true)

    if [ -z "$PIDS" ]; then
        echo -e "${YELLOW}not running${NC}"
        return 0
    fi

    # Kill the processes
    echo $PIDS | xargs kill -15 2>/dev/null || true

    # Wait up to 5 seconds for graceful shutdown
    for i in {1..10}; do
        PIDS=$(lsof -ti :$port 2>/dev/null || true)
        if [ -z "$PIDS" ]; then
            echo -e "${GREEN}stopped${NC}"
            return 0
        fi
        sleep 0.5
    done

    # Force kill if still running
    PIDS=$(lsof -ti :$port 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo -n "(force) "
        echo $PIDS | xargs kill -9 2>/dev/null || true
        sleep 1
    fi

    PIDS=$(lsof -ti :$port 2>/dev/null || true)
    if [ -z "$PIDS" ]; then
        echo -e "${GREEN}stopped${NC}"
    else
        echo -e "${RED}failed to stop${NC}"
    fi
}

# Stop services
stop_service "Flask HMI" 6101
stop_service "Event Stream" 6102

echo ""
echo "========================================="
echo -e "${GREEN}Phase 2 services stopped${NC}"
echo "========================================="
echo ""
