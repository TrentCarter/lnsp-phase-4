#!/bin/bash
# Stop Phase 0 services: Registry + Heartbeat Monitor

echo "🛑 Stopping Phase 0 services..."

# Stop Registry (6121)
lsof -ti:6121 | xargs kill -9 2>/dev/null && echo "✓ Registry stopped" || echo "  (Registry not running)"

# Stop Heartbeat Monitor (6109)
lsof -ti:6109 | xargs kill -9 2>/dev/null && echo "✓ Heartbeat Monitor stopped" || echo "  (Heartbeat Monitor not running)"

echo "✅ Phase 0 services stopped"
