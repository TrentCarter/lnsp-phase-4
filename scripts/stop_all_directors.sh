#!/bin/bash
# Stop All Director Services for P0 Stack
# Created: 2025-11-12

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log directory
LOG_DIR="logs/pas"

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}=== P0 Director Services Shutdown ===${NC}"
echo ""

# Function to stop a service by port
stop_by_port() {
    local name="$1"
    local port="$2"
    local name_lower=$(echo "$name" | tr '[:upper:]' '[:lower:]')
    local pid_file="$LOG_DIR/${name_lower}.pid"

    echo -e "${BLUE}🛑 Stopping $name (port $port)...${NC}"

    # Try to stop by PID file first
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            kill "$pid" 2>/dev/null || true
            sleep 1
            if ps -p "$pid" > /dev/null 2>&1; then
                kill -9 "$pid" 2>/dev/null || true
            fi
            rm -f "$pid_file"
            echo -e "${GREEN}✅ $name stopped (PID $pid)${NC}"
        else
            echo -e "${YELLOW}⚠️  $name PID $pid not running${NC}"
            rm -f "$pid_file"
        fi
    fi

    # Also kill by port (in case PID file is stale)
    local port_pid=$(lsof -ti ":$port" 2>/dev/null || true)
    if [ -n "$port_pid" ]; then
        echo -e "${BLUE}   Killing process on port $port (PID: $port_pid)${NC}"
        kill "$port_pid" 2>/dev/null || true
        sleep 1
        if lsof -ti ":$port" >/dev/null 2>&1; then
            kill -9 "$port_pid" 2>/dev/null || true
        fi
        echo -e "${GREEN}✅ $name stopped (port $port)${NC}"
    else
        if [ ! -f "$pid_file" ]; then
            echo -e "${YELLOW}⚠️  $name not running${NC}"
        fi
    fi
}

# Stop all Directors
stop_by_port "Dir-Code" 6111
stop_by_port "Dir-Models" 6112
stop_by_port "Dir-Data" 6113
stop_by_port "Dir-DevSecOps" 6114
stop_by_port "Dir-Docs" 6115

echo ""
echo -e "${GREEN}=== All Director services stopped ===${NC}"
echo ""
echo "To restart: ./scripts/start_all_directors.sh"
