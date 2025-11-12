#!/bin/bash
# Start All Director Services for P0 Stack
# Created: 2025-11-12
#
# Director services:
# - Dir-Code (6111): Code Lane Coordinator
# - Dir-Models (6112): Models Lane Coordinator
# - Dir-Data (6113): Data Lane Coordinator
# - Dir-DevSecOps (6114): DevSecOps Lane Coordinator
# - Dir-Docs (6115): Docs Lane Coordinator

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log directory
LOG_DIR="logs/pas"
mkdir -p "$LOG_DIR"

# Project root (assuming script is in scripts/ directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}=== P0 Director Services Startup ===${NC}"
echo "Project root: $PROJECT_ROOT"
echo "Logs: $LOG_DIR"
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Virtual environment not found at .venv${NC}"
    echo "Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Load .env file for API keys
if [ -f ".env" ]; then
    echo -e "${BLUE}Loading environment variables from .env${NC}"
    set -a  # automatically export all variables
    source .env
    set +a
else
    echo -e "${YELLOW}⚠ .env file not found, skipping environment variables${NC}"
fi

# Function to check if port is in use
port_in_use() {
    lsof -i ":$1" >/dev/null 2>&1
}

# Function to start a Director service
start_director() {
    local name="$1"
    local port="$2"
    local service_dir="$3"
    local name_lower=$(echo "$name" | tr '[:upper:]' '[:lower:]')
    local log_file="$LOG_DIR/${name_lower}.log"

    if port_in_use "$port"; then
        echo -e "${YELLOW}⚠️  $name already running on port $port${NC}"
        return 0
    fi

    echo -e "${BLUE}🚀 Starting $name on port $port...${NC}"

    # Start service in background with logs
    (
        cd "$PROJECT_ROOT"
        source .venv/bin/activate
        export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
        python -m uvicorn "$service_dir.app:app" \
            --host 127.0.0.1 \
            --port "$port" \
            --log-level info \
            >> "$log_file" 2>&1
    ) &

    # Save PID
    echo $! > "$LOG_DIR/${name_lower}.pid"

    # Wait a moment and verify it started
    sleep 2

    if port_in_use "$port"; then
        echo -e "${GREEN}✅ $name started successfully (PID: $(cat "$LOG_DIR/${name_lower}.pid"))${NC}"
    else
        echo -e "${RED}❌ $name failed to start (check $log_file)${NC}"
        return 1
    fi
}

# Start all Directors
echo ""
start_director "Dir-Code" 6111 "services.pas.director_code"
start_director "Dir-Models" 6112 "services.pas.director_models"
start_director "Dir-Data" 6113 "services.pas.director_data"
start_director "Dir-DevSecOps" 6114 "services.pas.director_devsecops"
start_director "Dir-Docs" 6115 "services.pas.director_docs"

echo ""
echo -e "${GREEN}=== All Director services started ===${NC}"
echo ""
echo "Service status:"
echo "  Dir-Code:       http://localhost:6111/health"
echo "  Dir-Models:     http://localhost:6112/health"
echo "  Dir-Data:       http://localhost:6113/health"
echo "  Dir-DevSecOps:  http://localhost:6114/health"
echo "  Dir-Docs:       http://localhost:6115/health"
echo ""
echo "Logs:"
echo "  Dir-Code:       $LOG_DIR/dir-code.log"
echo "  Dir-Models:     $LOG_DIR/dir-models.log"
echo "  Dir-Data:       $LOG_DIR/dir-data.log"
echo "  Dir-DevSecOps:  $LOG_DIR/dir-devsecops.log"
echo "  Dir-Docs:       $LOG_DIR/dir-docs.log"
echo ""
echo "To stop all Directors: ./scripts/stop_all_directors.sh"
