#!/bin/bash
# Stop All FastAPI Services for LNSP Pipeline
# Created: 2025-10-18

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

LOG_DIR="/tmp/lnsp_api_logs"

echo -e "${YELLOW}=== Stopping LNSP FastAPI Services ===${NC}"

# Kill services by PID file
stop_service() {
    local name="$1"
    local pid_file="$LOG_DIR/${name}.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "${GREEN}Stopping ${name} (PID: ${pid})...${NC}"
            kill "$pid" 2>/dev/null || true
            sleep 1
            # Force kill if still running
            if ps -p "$pid" > /dev/null 2>&1; then
                echo -e "${YELLOW}Force killing ${name}...${NC}"
                kill -9 "$pid" 2>/dev/null || true
            fi
        else
            echo -e "${YELLOW}${name} not running (stale PID file)${NC}"
        fi
        rm "$pid_file"
    else
        echo -e "${YELLOW}${name}: no PID file found${NC}"
    fi
}

# Stop services in reverse order
stop_service "vec2text_decoder"
stop_service "ingest_api"
stop_service "gtr_t5_embeddings"
stop_service "semantic_chunker"
stop_service "episode_chunker"

# Fallback: kill any remaining uvicorn processes on LNSP ports
echo ""
echo -e "${YELLOW}Checking for orphaned processes...${NC}"

for port in 8900 8001 8767 8004 8766; do
    pids=$(lsof -ti ":$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo -e "${YELLOW}Killing orphaned process on port ${port}: ${pids}${NC}"
        echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
done

echo ""
echo -e "${GREEN}✅ All FastAPI services stopped${NC}"
