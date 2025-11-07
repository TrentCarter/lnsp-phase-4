#!/bin/bash
# Stop all PAS services (Phase 0 + Phase 1)

echo "🛑 Stopping all PAS services..."
echo ""

# Stop Phase 1
echo "Phase 1:"
lsof -ti:6104 | xargs kill -9 2>/dev/null && echo "  ✓ Resource Manager stopped" || echo "  (Resource Manager not running)"
lsof -ti:6105 | xargs kill -9 2>/dev/null && echo "  ✓ Token Governor stopped" || echo "  (Token Governor not running)"

echo ""

# Stop Phase 0
echo "Phase 0:"
lsof -ti:6121 | xargs kill -9 2>/dev/null && echo "  ✓ Registry stopped" || echo "  (Registry not running)"
lsof -ti:6109 | xargs kill -9 2>/dev/null && echo "  ✓ Heartbeat Monitor stopped" || echo "  (Heartbeat Monitor not running)"

echo ""
echo "✅ All PAS services stopped"
