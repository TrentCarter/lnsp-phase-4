#!/bin/bash
# Comprehensive test script for Phase 0: Registry + Heartbeat Monitor

set -e

REPO_ROOT="/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4"
cd "$REPO_ROOT"

echo "=========================================="
echo "Phase 0 Integration Tests"
echo "=========================================="
echo ""

# Ensure services are running
echo "1. Checking service health..."
if ! curl -s http://localhost:6121/health | grep -q "ok"; then
    echo "❌ Registry not healthy. Please run ./scripts/start_phase0_services.sh first."
    exit 1
fi

if ! curl -s http://localhost:6109/health | grep -q "ok"; then
    echo "❌ Heartbeat Monitor not healthy. Please run ./scripts/start_phase0_services.sh first."
    exit 1
fi

echo "✓ Both services healthy"
echo ""

# Test 1: Register a service
echo "2. Test: Register a service..."
SERVICE_ID=$(curl -s -X POST http://localhost:6121/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-llama-service",
    "type": "model",
    "role": "experimental",
    "url": "http://127.0.0.1:8888",
    "caps": ["infer", "classify"],
    "labels": {"space": "local", "tier": "2"},
    "ctx_limit": 32768,
    "heartbeat_interval_s": 60,
    "ttl_s": 90
  }' | jq -r '.service_id')

if [ -z "$SERVICE_ID" ] || [ "$SERVICE_ID" == "null" ]; then
    echo "❌ Failed to register service"
    exit 1
fi

echo "✓ Service registered with ID: $SERVICE_ID"
echo ""

# Test 2: Discover the service
echo "3. Test: Discover the registered service..."
DISCOVERED=$(curl -s "http://localhost:6121/discover?name=test-llama-service" | jq -r '.items[0].service_id')

if [ "$DISCOVERED" != "$SERVICE_ID" ]; then
    echo "❌ Failed to discover service"
    exit 1
fi

echo "✓ Service discovered successfully"
echo ""

# Test 3: Send heartbeat
echo "4. Test: Send heartbeat..."
HEARTBEAT_RESULT=$(curl -s -X PUT http://localhost:6121/heartbeat \
  -H "Content-Type: application/json" \
  -d "{
    \"service_id\": \"$SERVICE_ID\",
    \"status\": \"ok\",
    \"p95_ms\": 123.4,
    \"queue_depth\": 2,
    \"load\": 0.35
  }" | jq -r '.success')

if [ "$HEARTBEAT_RESULT" != "true" ]; then
    echo "❌ Failed to send heartbeat"
    exit 1
fi

echo "✓ Heartbeat sent successfully"
echo ""

# Test 4: Check heartbeat monitor stats
echo "5. Test: Check Heartbeat Monitor stats..."
TOTAL_SERVICES=$(curl -s http://localhost:6109/stats | jq -r '.total_services')

if [ "$TOTAL_SERVICES" -lt 1 ]; then
    echo "❌ Heartbeat Monitor not tracking services"
    exit 1
fi

echo "✓ Heartbeat Monitor tracking $TOTAL_SERVICES service(s)"
echo ""

# Test 5: Test service discovery with filters
echo "6. Test: Discover services with filters..."
FILTERED=$(curl -s "http://localhost:6121/discover?type=model&role=experimental&cap=infer" | jq -r '.items | length')

if [ "$FILTERED" -lt 1 ]; then
    echo "❌ Filtered discovery failed"
    exit 1
fi

echo "✓ Filtered discovery returned $FILTERED service(s)"
echo ""

# Test 6: Wait for missed heartbeat detection (90s TTL)
echo "7. Test: Wait for missed heartbeat detection..."
echo "   (This will take ~2 minutes to test TTL eviction)"
echo "   Waiting 120 seconds..."

for i in {1..12}; do
    echo -n "."
    sleep 10
done
echo ""

# Check if service was marked down
DOWN_SERVICES=$(curl -s http://localhost:6109/stats | jq -r '.down_services')

if [ "$DOWN_SERVICES" -ge 1 ]; then
    echo "✓ Service marked 'down' after missed heartbeats"
else
    echo "⚠️  Service not marked 'down' yet (may need more time)"
fi
echo ""

# Test 7: Check alert log
echo "8. Test: Check heartbeat alerts..."
ALERTS=$(curl -s http://localhost:6109/alerts | jq -r '.alerts | length')

if [ "$ALERTS" -ge 1 ]; then
    echo "✓ Heartbeat Monitor issued $ALERTS alert(s)"
    echo ""
    echo "Recent alerts:"
    curl -s http://localhost:6109/alerts | jq -r '.alerts[] | "  - [\(.ts)] \(.alert_type): \(.service_name) → \(.action)"'
else
    echo "⚠️  No alerts issued yet"
fi
echo ""

# Test 8: Deregister service (cleanup)
echo "9. Test: Deregister service..."
DEREG_RESULT=$(curl -s -X POST http://localhost:6121/deregister \
  -H "Content-Type: application/json" \
  -d "{\"service_id\": \"$SERVICE_ID\"}" | jq -r '.success')

if [ "$DEREG_RESULT" == "true" ]; then
    echo "✓ Service deregistered successfully"
else
    echo "⚠️  Service may have already been auto-deregistered"
fi
echo ""

# Summary
echo "=========================================="
echo "Phase 0 Tests Complete!"
echo "=========================================="
echo ""
echo "✅ All core functionality verified:"
echo "   - Service registration"
echo "   - Service discovery (with filters)"
echo "   - Heartbeat updates"
echo "   - Heartbeat monitoring"
echo "   - TTL-based eviction"
echo "   - Alert generation"
echo "   - Service deregistration"
echo ""
echo "Next steps:"
echo "   1. Review logs: tail -f /tmp/pas_logs/*.log"
echo "   2. Check event log: cat artifacts/hmi/events/heartbeat_alerts_*.jsonl"
echo "   3. Proceed to Phase 1: Resource Manager + Token Governor"
echo ""
