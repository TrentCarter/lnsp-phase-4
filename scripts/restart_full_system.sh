#!/usr/bin/env bash
#
# restart_full_system.sh - Complete system restart
#
# Stops and restarts ALL services:
#   - P0 Stack: Gateway, PAS Root, Aider-LCO
#   - Phase 0: Registry, Heartbeat Monitor
#   - Phase 1: Resource Manager, Token Governor
#   - HMI: Web Dashboard
#
# Usage:
#   bash scripts/restart_full_system.sh
#

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to repo root
cd "$REPO_ROOT"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  Full System Restart${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# ============================================================================
# STEP 1: Stop All Services
# ============================================================================

echo -e "${YELLOW}[1/3] Stopping all services...${NC}"
echo ""

# Stop P0 Stack
echo "P0 Stack:"
lsof -ti:6120 | xargs kill -9 2>/dev/null && echo -e "  ${GREEN}✓${NC} Gateway stopped" || echo "  (Gateway not running)"
lsof -ti:6100 | xargs kill -9 2>/dev/null && echo -e "  ${GREEN}✓${NC} PAS Root stopped" || echo "  (PAS Root not running)"
lsof -ti:6130 | xargs kill -9 2>/dev/null && echo -e "  ${GREEN}✓${NC} Aider-LCO stopped" || echo "  (Aider-LCO not running)"

echo ""

# Stop Phase 0
echo "Phase 0:"
lsof -ti:6121 | xargs kill -9 2>/dev/null && echo -e "  ${GREEN}✓${NC} Registry stopped" || echo "  (Registry not running)"
lsof -ti:6109 | xargs kill -9 2>/dev/null && echo -e "  ${GREEN}✓${NC} Heartbeat Monitor stopped" || echo "  (Heartbeat Monitor not running)"

echo ""

# Stop Phase 1
echo "Phase 1:"
lsof -ti:6104 | xargs kill -9 2>/dev/null && echo -e "  ${GREEN}✓${NC} Resource Manager stopped" || echo "  (Resource Manager not running)"
lsof -ti:6105 | xargs kill -9 2>/dev/null && echo -e "  ${GREEN}✓${NC} Token Governor stopped" || echo "  (Token Governor not running)"

echo ""

# Stop HMI
echo "HMI:"
lsof -ti:6101 | xargs kill -9 2>/dev/null && echo -e "  ${GREEN}✓${NC} Web Dashboard stopped" || echo "  (Web Dashboard not running)"

echo ""

# Stop Model Pool Manager + Model Services
echo "Model Pool:"
lsof -ti:8050 | xargs kill -9 2>/dev/null && echo -e "  ${GREEN}✓${NC} Model Pool Manager stopped" || echo "  (Model Pool Manager not running)"
# Stop all model services (8051-8099)
for port in $(seq 8051 8099); do
    lsof -ti:$port 2>/dev/null | xargs kill -9 2>/dev/null
done
echo -e "  ${GREEN}✓${NC} Model services stopped (ports 8051-8099)"

echo ""
echo -e "${GREEN}✓ All services stopped${NC}"
echo ""

# Wait for ports to be fully released
echo "Waiting 3 seconds for ports to be released..."
sleep 3
echo ""

# ============================================================================
# STEP 2: Start Core Services
# ============================================================================

echo -e "${YELLOW}[2/3] Starting core services...${NC}"
echo ""

# Start Phase 0 (Core Infrastructure)
echo "Starting Phase 0 (Registry, Heartbeat Monitor)..."
if [ -f "./scripts/start_phase0_services.sh" ]; then
    ./scripts/start_phase0_services.sh
    echo -e "${GREEN}✓ Phase 0 started${NC}"
else
    echo -e "${YELLOW}⚠ Phase 0 script not found, starting manually...${NC}"
    cd services/registry && ../../.venv/bin/python registry_service.py &
    sleep 2
    cd ../heartbeat_monitor && ../../.venv/bin/python heartbeat_monitor.py &
    sleep 2
    cd "$REPO_ROOT"
fi

echo ""
echo "Waiting 5s for Phase 0 to stabilize..."
sleep 5
echo ""

# Start Phase 1 (Management Agents)
echo "Starting Phase 1 (Resource Manager, Token Governor)..."
if [ -f "./scripts/start_phase1_services.sh" ]; then
    ./scripts/start_phase1_services.sh
    echo -e "${GREEN}✓ Phase 1 started${NC}"
else
    echo -e "${YELLOW}⚠ Phase 1 script not found, starting manually...${NC}"
    cd services/resource_manager && ../../.venv/bin/python resource_manager.py &
    sleep 2
    cd ../token_governor && ../../.venv/bin/python token_governor.py &
    sleep 2
    cd "$REPO_ROOT"
fi

echo ""
echo "Waiting 5s for Phase 1 to stabilize..."
sleep 5
echo ""

# ============================================================================
# STEP 3: Start P0 Stack + HMI
# ============================================================================

echo -e "${YELLOW}[3/3] Starting P0 Stack + Model Pool + HMI...${NC}"
echo ""

# Start Model Pool Manager (port 8050)
echo "Starting Model Pool Manager (port 8050)..."
./.venv/bin/python services/model_pool_manager/model_pool_manager.py > /tmp/model_pool_manager.log 2>&1 &
sleep 3
if lsof -ti:8050 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Model Pool Manager started (warmup models loading...)${NC}"
else
    echo -e "${RED}✗ Model Pool Manager failed to start${NC}"
fi

echo ""

# Start Aider-LCO (port 6130)
echo "Starting Aider-LCO RPC (port 6130)..."
./.venv/bin/uvicorn services.tools.aider_rpc.app:app --host 127.0.0.1 --port 6130 > /dev/null 2>&1 &
sleep 2
if lsof -ti:6130 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Aider-LCO started${NC}"
else
    echo -e "${RED}✗ Aider-LCO failed to start${NC}"
fi

echo ""

# Start PAS Root (port 6100)
echo "Starting PAS Root (port 6100)..."
./.venv/bin/uvicorn services.pas.root.app:app --host 127.0.0.1 --port 6100 > /dev/null 2>&1 &
sleep 2
if lsof -ti:6100 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PAS Root started${NC}"
else
    echo -e "${RED}✗ PAS Root failed to start${NC}"
fi

echo ""

# Start Gateway (port 6120)
echo "Starting Gateway (port 6120)..."
./.venv/bin/uvicorn services.gateway.app:app --host 127.0.0.1 --port 6120 > /dev/null 2>&1 &
sleep 2
if lsof -ti:6120 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Gateway started${NC}"
else
    echo -e "${RED}✗ Gateway failed to start${NC}"
fi

echo ""

# Start HMI (port 6101)
echo "Starting HMI Web Dashboard (port 6101)..."
cd services/webui && ../../.venv/bin/python hmi_app.py > /dev/null 2>&1 &
cd "$REPO_ROOT"
sleep 3
if lsof -ti:6101 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ HMI started${NC}"
else
    echo -e "${RED}✗ HMI failed to start${NC}"
fi

echo ""

# ============================================================================
# Done!
# ============================================================================

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Full System Restart Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}Model Pool:${NC}"
echo "  Pool Manager:      http://localhost:8050"
echo "  qwen2.5-coder:     http://localhost:8051 (warmup)"
echo "  llama3.1:          http://localhost:8052 (warmup)"
echo ""
echo -e "${GREEN}P0 Stack:${NC}"
echo "  Gateway:           http://localhost:6120"
echo "  PAS Root:          http://localhost:6100"
echo "  Aider-LCO:         http://localhost:6130"
echo ""
echo -e "${GREEN}Phase 0 (Core):${NC}"
echo "  Registry:          http://localhost:6121"
echo "  Heartbeat Monitor: http://localhost:6109"
echo ""
echo -e "${GREEN}Phase 1 (Management):${NC}"
echo "  Resource Manager:  http://localhost:6104"
echo "  Token Governor:    http://localhost:6105"
echo ""
echo -e "${GREEN}HMI (Dashboard):${NC}"
echo "  Main Dashboard:    http://localhost:6101/"
echo "  Tree View:         http://localhost:6101/tree"
echo "  Sequencer:         http://localhost:6101/sequencer"
echo "  Actions:           http://localhost:6101/actions"
echo ""
echo -e "${YELLOW}Test with:${NC}"
echo "  ./bin/verdict health"
echo "  curl http://localhost:6101"
echo ""
