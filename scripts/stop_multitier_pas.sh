#!/usr/bin/env bash
#
# Stop Multi-Tier PAS
#
# Stops all PAS services gracefully
#

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Stopping Multi-Tier PAS ==="
echo ""

# Stop services by port
PORTS=(6120 6100 6115 6114 6113 6112 6111 6110)
SERVICE_NAMES=("Gateway" "PAS Root" "Director-Docs" "Director-DevSecOps" "Director-Data" "Director-Models" "Director-Code" "Architect")

for i in "${!PORTS[@]}"; do
    PORT="${PORTS[$i]}"
    SERVICE="${SERVICE_NAMES[$i]}"

    PID=$(lsof -ti:$PORT 2>/dev/null || true)

    if [ ! -z "$PID" ]; then
        echo -e "${GREEN}✓${NC} Stopping $SERVICE (port $PORT, PID $PID)"
        kill $PID 2>/dev/null || true
        sleep 1
    else
        echo -e "${YELLOW}⊗${NC} $SERVICE (port $PORT) - not running"
    fi
done

echo ""
echo -e "${GREEN}=== Multi-Tier PAS Stopped ===${NC}"
