#!/bin/bash
#
# Start Phase 6 Cloud Provider Adapters
#
# This script starts all 4 cloud LLM provider adapters:
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
echo -e "${BLUE}Starting Phase 6: Cloud Provider Adapters${NC}"
echo -e "${BLUE}============================================================${NC}"

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  Warning: .env file not found!${NC}"
    echo -e "${YELLOW}   Copying .env.template to .env${NC}"
    cp .env.template .env
    echo -e "${RED}   ❌ Please edit .env and add your API keys!${NC}"
    echo -e "${RED}   Then run this script again.${NC}"
    exit 1
fi

# Source .env
source .env

# Create log directory
mkdir -p /tmp/pas_logs

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Virtual environment not found at .venv${NC}"
    echo -e "${YELLOW}   Creating virtual environment...${NC}"
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

# Function to check if port is in use
check_port() {
    port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${YELLOW}⚠️  Port $port already in use${NC}"
        return 1
    else
        return 0
    fi
}

# Function to start a service
start_service() {
    service_name=$1
    port=$2
    module_path=$3

    echo -e "\n${BLUE}Starting $service_name (Port $port)...${NC}"

    if ! check_port $port ; then
        echo -e "${YELLOW}   Skipping (already running)${NC}"
        return
    fi

    # Start service in background
    .venv/bin/python -m $module_path \
        > /tmp/pas_logs/${service_name}.log 2>&1 &

    pid=$!
    echo $pid > /tmp/pas_logs/${service_name}.pid

    # Wait for service to start
    sleep 2

    # Check if service is running
    if ps -p $pid > /dev/null 2>&1; then
        # Check health endpoint
        if curl -s http://localhost:$port/health > /dev/null 2>&1; then
            echo -e "${GREEN}   ✅ $service_name started (PID: $pid)${NC}"
        else
            echo -e "${YELLOW}   ⚠️  $service_name started but health check failed${NC}"
            echo -e "${YELLOW}   Check logs: tail -f /tmp/pas_logs/${service_name}.log${NC}"
        fi
    else
        echo -e "${RED}   ❌ $service_name failed to start${NC}"
        echo -e "${RED}   Check logs: cat /tmp/pas_logs/${service_name}.log${NC}"
    fi
}

# Start all Phase 6 services
echo -e "\n${BLUE}Starting Cloud Provider Adapters...${NC}"

start_service "openai_adapter" 8100 "services.cloud_providers.openai.openai_adapter"
start_service "anthropic_adapter" 8101 "services.cloud_providers.anthropic.anthropic_adapter"
start_service "gemini_adapter" 8102 "services.cloud_providers.gemini.gemini_adapter"
start_service "grok_adapter" 8103 "services.cloud_providers.grok.grok_adapter"

# Summary
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${GREEN}✅ Phase 6 Cloud Provider Adapters Started${NC}"
echo -e "${BLUE}============================================================${NC}"

echo -e "\n${BLUE}Service Status:${NC}"
for port in 8100 8101 8102 8103; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "  ${GREEN}✅ Port $port: Running${NC}"
    else
        echo -e "  ${RED}❌ Port $port: Not running${NC}"
    fi
done

echo -e "\n${BLUE}Access Points:${NC}"
echo -e "  OpenAI Adapter:    ${GREEN}http://localhost:8100/docs${NC}"
echo -e "  Anthropic Adapter: ${GREEN}http://localhost:8101/docs${NC}"
echo -e "  Gemini Adapter:    ${GREEN}http://localhost:8102/docs${NC}"
echo -e "  Grok Adapter:      ${GREEN}http://localhost:8103/docs${NC}"

echo -e "\n${BLUE}Health Checks:${NC}"
for port in 8100 8101 8102 8103; do
    echo -e "  curl http://localhost:$port/health"
done

echo -e "\n${BLUE}Logs:${NC}"
echo -e "  tail -f /tmp/pas_logs/*.log"

echo -e "\n${BLUE}Stop Services:${NC}"
echo -e "  ./scripts/stop_phase6_cloud_providers.sh"

echo -e "\n${BLUE}============================================================${NC}"
