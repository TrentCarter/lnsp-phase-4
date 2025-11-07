#!/bin/bash
# Stop Phase 1 services: Resource Manager + Token Governor

echo "🛑 Stopping Phase 1 services..."

# Stop Resource Manager (6104)
lsof -ti:6104 | xargs kill -9 2>/dev/null && echo "✓ Resource Manager stopped" || echo "  (Resource Manager not running)"

# Stop Token Governor (6105)
lsof -ti:6105 | xargs kill -9 2>/dev/null && echo "✓ Token Governor stopped" || echo "  (Token Governor not running)"

echo "✅ Phase 1 services stopped"
echo ""
echo "Phase 0 services still running:"
echo "  Registry (6121), Heartbeat Monitor (6109)"
echo ""
echo "To stop all services:"
echo "  ./scripts/stop_all_pas_services.sh"
echo ""
