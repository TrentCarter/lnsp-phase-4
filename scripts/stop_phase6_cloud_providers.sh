#!/bin/bash
#
# Stop Phase 6 Cloud Provider Adapters
#
# This script stops all 4 cloud LLM provider adapters:
# - OpenAI (Port 8100)
# - Anthropic (Port 8101)
# - Gemini (Port 8102)
# - Grok (Port 8103)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Stopping Phase 6: Cloud Provider Adapters${NC}"
echo -e "${BLUE}============================================================${NC}"

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Function to stop a service by name
stop_service() {
    service_name=$1
    port=$2

    echo -e "\n${BLUE}Stopping $service_name (Port $port)...${NC}"

    # Try to stop by PID file
    if [ -f "/tmp/pas_logs/${service_name}.pid" ]; then
        pid=$(cat /tmp/pas_logs/${service_name}.pid)

        if ps -p $pid > /dev/null 2>&1; then
            kill $pid 2>/dev/null || true
            sleep 1

            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                kill -9 $pid 2>/dev/null || true
            fi

            rm -f /tmp/pas_logs/${service_name}.pid
            echo -e "${GREEN}   ✅ $service_name stopped (PID: $pid)${NC}"
        else
            echo -e "${YELLOW}   ⚠️  $service_name not running (stale PID)${NC}"
            rm -f /tmp/pas_logs/${service_name}.pid
        fi
    else
        # Try to stop by port
        pid=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill $pid 2>/dev/null || true
            sleep 1

            # Force kill if still running
            pid=$(lsof -ti:$port 2>/dev/null || true)
            if [ -n "$pid" ]; then
                kill -9 $pid 2>/dev/null || true
            fi

            echo -e "${GREEN}   ✅ $service_name stopped (Port $port)${NC}"
        else
            echo -e "${YELLOW}   ⚠️  $service_name not running${NC}"
        fi
    fi
}

# Stop all Phase 6 services
echo -e "\n${BLUE}Stopping Cloud Provider Adapters...${NC}"

stop_service "openai_adapter" 8100
stop_service "anthropic_adapter" 8101
stop_service "gemini_adapter" 8102
stop_service "grok_adapter" 8103

# Summary
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${GREEN}✅ Phase 6 Cloud Provider Adapters Stopped${NC}"
echo -e "${BLUE}============================================================${NC}"

echo -e "\n${BLUE}Verification:${NC}"
for port in 8100 8101 8102 8103; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "  ${RED}❌ Port $port: Still running${NC}"
    else
        echo -e "  ${GREEN}✅ Port $port: Stopped${NC}"
    fi
done

echo -e "\n${BLUE}============================================================${NC}"
