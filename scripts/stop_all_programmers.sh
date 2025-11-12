#!/bin/bash
#
# Stop All Programmers Script
#
# Gracefully stops all Programmer FastAPI services (ports 6151-6160)
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
PIDS_FILE="$PROJECT_ROOT/artifacts/pids/programmers.pids"

# Programmer ports
PORTS=(6151 6152 6153 6154 6155 6156 6157 6158 6159 6160)

echo "============================================================"
echo "Stopping All Programmer Services"
echo "============================================================"
echo ""

# Function to kill process on port
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port 2>/dev/null || true)

    if [ -n "$pid" ]; then
        echo -e "${BLUE}Stopping process on port $port (PID: $pid)${NC}"
        kill -15 $pid 2>/dev/null || true  # SIGTERM (graceful)
        sleep 1

        # Force kill if still running
        if lsof -ti:$port > /dev/null 2>&1; then
            echo -e "${YELLOW}Force killing process on port $port${NC}"
            kill -9 $pid 2>/dev/null || true
        fi

        echo -e "${GREEN}✓ Stopped port $port${NC}"
    else
        echo -e "${YELLOW}⊘ No process on port $port${NC}"
    fi
}

# Stop all Programmers by port
for port in "${PORTS[@]}"; do
    kill_port $port
done

# Clean up PIDs file
if [ -f "$PIDS_FILE" ]; then
    rm "$PIDS_FILE"
    echo -e "\n${GREEN}✓ Cleaned up PIDs file${NC}"
fi

echo ""
echo "============================================================"
echo "All Programmer Services Stopped"
echo "============================================================"
echo ""
