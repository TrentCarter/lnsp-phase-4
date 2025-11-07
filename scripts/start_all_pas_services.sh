#!/bin/bash
# Start all PAS services (Phase 0 + Phase 1)

set -e

echo "=========================================="
echo "Starting PAS Services (Phase 0 + Phase 1)"
echo "=========================================="
echo ""

# Start Phase 0
echo "Starting Phase 0 services..."
./scripts/start_phase0_services.sh

echo ""
echo "Waiting 5s for Phase 0 to stabilize..."
sleep 5
echo ""

# Start Phase 1
echo "Starting Phase 1 services..."
./scripts/start_phase1_services.sh

echo ""
echo "=========================================="
echo "All PAS Services Running!"
echo "=========================================="
echo ""
echo "Phase 0 (Core Infrastructure):"
echo "  Registry:           http://localhost:6121"
echo "  Heartbeat Monitor:  http://localhost:6109"
echo ""
echo "Phase 1 (Management Agents):"
echo "  Resource Manager:   http://localhost:6104"
echo "  Token Governor:     http://localhost:6105"
echo ""
echo "Interactive API Docs:"
echo "  http://localhost:6121/docs  (Registry)"
echo "  http://localhost:6109/docs  (Heartbeat Monitor)"
echo "  http://localhost:6104/docs  (Resource Manager)"
echo "  http://localhost:6105/docs  (Token Governor)"
echo ""
echo "To stop all services:"
echo "  ./scripts/stop_all_pas_services.sh"
echo ""
