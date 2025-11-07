#!/bin/bash
# Demo 1: Idempotent Start
# Shows Redis-backed idempotency with replay header
# Duration: <2 minutes

set -e

PLMS_API_BASE_URL="${PLMS_API_BASE_URL:-http://localhost:6100}"
IDEMPOTENCY_KEY="demo1-$(date +%s)"

echo "=== Demo 1: Idempotent Start ==="
echo "API Base URL: $PLMS_API_BASE_URL"
echo "Idempotency Key: $IDEMPOTENCY_KEY"
echo ""

# Check if Redis is running
if ! redis-cli ping >/dev/null 2>&1; then
    echo "⚠️  Warning: Redis not running. Starting Redis..."
    docker run -d -p 6379:6379 redis:alpine
    sleep 2
fi

echo "Step 1: First request (should create new run)"
echo "=========================================="
RESPONSE1=$(curl -s -i -X POST "$PLMS_API_BASE_URL/api/projects/42/start" \
    -H "Idempotency-Key: $IDEMPOTENCY_KEY" \
    -H "Content-Type: application/json" \
    -d '{"run_kind": "baseline"}')

echo "$RESPONSE1"
echo ""

# Extract Idempotent-Replay header
REPLAY_HEADER1=$(echo "$RESPONSE1" | grep -i "idempotent-replay:" | cut -d: -f2 | tr -d ' \r\n')
echo "First request Idempotent-Replay header: [$REPLAY_HEADER1]"

if [ "$REPLAY_HEADER1" != "false" ]; then
    echo "❌ FAIL: Expected Idempotent-Replay: false, got: $REPLAY_HEADER1"
    exit 1
fi

echo "✓ First request correctly returned Idempotent-Replay: false"
echo ""

sleep 1

echo "Step 2: Second request (should return cached response)"
echo "=========================================="
RESPONSE2=$(curl -s -i -X POST "$PLMS_API_BASE_URL/api/projects/42/start" \
    -H "Idempotency-Key: $IDEMPOTENCY_KEY" \
    -H "Content-Type: application/json" \
    -d '{"run_kind": "baseline"}')

echo "$RESPONSE2"
echo ""

# Extract Idempotent-Replay header
REPLAY_HEADER2=$(echo "$RESPONSE2" | grep -i "idempotent-replay:" | cut -d: -f2 | tr -d ' \r\n')
echo "Second request Idempotent-Replay header: [$REPLAY_HEADER2]"

if [ "$REPLAY_HEADER2" != "true" ]; then
    echo "❌ FAIL: Expected Idempotent-Replay: true, got: $REPLAY_HEADER2"
    exit 1
fi

echo "✓ Second request correctly returned Idempotent-Replay: true"
echo ""

# Extract run_id from both responses
RUN_ID1=$(echo "$RESPONSE1" | grep -A 50 "^{" | jq -r '.run_id')
RUN_ID2=$(echo "$RESPONSE2" | grep -A 50 "^{" | jq -r '.run_id')

if [ "$RUN_ID1" != "$RUN_ID2" ]; then
    echo "❌ FAIL: Run IDs don't match!"
    echo "  First:  $RUN_ID1"
    echo "  Second: $RUN_ID2"
    exit 1
fi

echo "✓ Both requests returned the same run_id: $RUN_ID1"
echo ""

echo "=== Demo 1 Complete ==="
echo "✓ Idempotency verified"
echo "✓ First request: Idempotent-Replay: false"
echo "✓ Second request: Idempotent-Replay: true"
echo "✓ Same run_id returned"
echo ""
echo "💡 Value: Multi-pod safe, operator-proof retries"
echo "   HMI shows single run card regardless of duplicate submissions"
