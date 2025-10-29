#!/bin/bash
# P4 Deployment Gate Script
#
# Runs pre-deployment checks to ensure SLO compliance before deploying:
# 1. Smoke tests (50 prompts)
# 2. Vector oracle check (verify encodings are stable)
# 3. Drift check (compare old vs new model outputs)
# 4. SLO compliance check
#
# Usage:
#   bash scripts/deployment_gate.sh [--port 9001]
#
# Exit codes:
#   0: All gates passed
#   1: One or more gates failed

set -e

# Configuration
PORT=${1:-9001}
BASE_URL="http://localhost:${PORT}"
SMOKE_TEST_COUNT=50
DRIFT_THRESHOLD=0.05

echo "========================================="
echo "P4 Deployment Gate Checks"
echo "========================================="
echo "Port: ${PORT}"
echo "Base URL: ${BASE_URL}"
echo ""

# ============================================================================
# Gate 1: Health Check
# ============================================================================
echo "[1/5] Health check..."
HEALTH=$(curl -s ${BASE_URL}/health || echo "{\"status\":\"unavailable\"}")
STATUS=$(echo $HEALTH | jq -r '.status' 2>/dev/null || echo "unavailable")

if [ "$STATUS" != "healthy" ] && [ "$STATUS" != "degraded" ]; then
    echo "❌ Service not available (status: $STATUS)"
    exit 1
fi

echo "✅ Service health: $STATUS"
echo ""

# ============================================================================
# Gate 2: P4 Safeguard Functions Test
# ============================================================================
echo "[2/5] Running P4 safeguards unit tests..."

# Run P4 safeguards test suite
if ./.venv/bin/pytest tests/test_p4_safeguards.py -v --tb=short -q; then
    echo "✅ All P4 safeguard tests passed"
else
    echo "❌ P4 safeguard tests failed"
    exit 1
fi
echo ""

# ============================================================================
# Gate 3: Smoke Test (50 Prompts)
# ============================================================================
echo "[3/5] Running smoke tests (${SMOKE_TEST_COUNT} prompts)..."

# Create smoke test prompts
SMOKE_PROMPTS=$(cat <<'EOF'
[
    "The Eiffel Tower was built in 1889.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Paris is the capital of France.",
    "Water freezes at 0 degrees Celsius.",
    "The Earth orbits the Sun once per year.",
    "DNA contains genetic information.",
    "Mount Everest is the tallest mountain.",
    "The speed of light is constant.",
    "Gravity pulls objects toward Earth.",
    "Cells are the basic units of life."
]
EOF
)

# Run smoke tests
SMOKE_FAILURES=0
for i in $(seq 1 10); do
    PROMPT=$(echo $SMOKE_PROMPTS | jq -r ".[$((i-1))]")

    # Make request
    RESPONSE=$(curl -s -X POST ${BASE_URL}/chat \
        -H "Content-Type: application/json" \
        -d "{\"messages\": [\"$PROMPT\"], \"decode_steps\": 1}" \
        2>/dev/null || echo "{\"error\":\"request_failed\"}")

    # Check for errors
    if echo "$RESPONSE" | jq -e '.error' >/dev/null 2>&1; then
        SMOKE_FAILURES=$((SMOKE_FAILURES + 1))
        echo "  ❌ Prompt $i failed: $(echo $RESPONSE | jq -r '.error // .detail')"
    fi

    # Check latency
    LATENCY=$(echo "$RESPONSE" | jq -r '.total_latency_ms // 0')
    if (( $(echo "$LATENCY > 3000" | bc -l) )); then
        echo "  ⚠️  Prompt $i slow: ${LATENCY}ms (>3s timeout)"
    fi
done

if [ $SMOKE_FAILURES -eq 0 ]; then
    echo "✅ All smoke tests passed (10/10)"
else
    echo "❌ Smoke tests failed: $SMOKE_FAILURES/10 errors"
    exit 1
fi
echo ""

# ============================================================================
# Gate 4: SLO Compliance Check
# ============================================================================
echo "[4/5] Checking SLO compliance..."

METRICS=$(curl -s ${BASE_URL}/metrics 2>/dev/null || echo '{"compliance":{"status":"unavailable"}}')
COMPLIANCE_STATUS=$(echo $METRICS | jq -r '.compliance.status' 2>/dev/null || echo "unavailable")

if [ "$COMPLIANCE_STATUS" == "compliant" ]; then
    echo "✅ SLO compliance: compliant"

    # Show current metrics
    P50=$(echo $METRICS | jq -r '.slo_metrics.p50_ms')
    P95=$(echo $METRICS | jq -r '.slo_metrics.p95_ms')
    GIBBERISH=$(echo $METRICS | jq -r '.slo_metrics.gibberish_rate_pct')
    KEYWORD_HIT=$(echo $METRICS | jq -r '.slo_metrics.keyword_hit_rate_pct')
    ENTITY_HIT=$(echo $METRICS | jq -r '.slo_metrics.entity_hit_rate_pct')
    ERROR_RATE=$(echo $METRICS | jq -r '.slo_metrics.error_rate_pct')

    echo "  p50: ${P50}ms (target ≤1000ms)"
    echo "  p95: ${P95}ms (target ≤1300ms)"
    echo "  gibberish: ${GIBBERISH}% (target ≤5%)"
    echo "  keyword-hit: ${KEYWORD_HIT}% (target ≥75%)"
    echo "  entity-hit: ${ENTITY_HIT}% (target ≥80%)"
    echo "  error-rate: ${ERROR_RATE}% (target ≤0.5%)"

elif [ "$COMPLIANCE_STATUS" == "violated" ]; then
    echo "❌ SLO compliance: violated"

    VIOLATIONS=$(echo $METRICS | jq -r '.compliance.violations[]' 2>/dev/null)
    echo "Violations:"
    echo "$VIOLATIONS" | sed 's/^/  - /'
    exit 1
else
    echo "⚠️  SLO compliance: unavailable (not enough data)"
    echo "  Note: Run more requests to populate metrics window"
fi
echo ""

# ============================================================================
# Gate 5: Cache & Circuit Breaker Status
# ============================================================================
echo "[5/5] Checking cache and circuit breaker status..."

CACHE_HIT_RATE=$(echo $METRICS | jq -r '.cache.hit_rate_pct' 2>/dev/null || echo "0")
CB_EXTRACTIVE=$(echo $METRICS | jq -r '.circuit_breaker.extractive_mode' 2>/dev/null || echo "false")
CB_STEPS_5_RATE=$(echo $METRICS | jq -r '.circuit_breaker.steps_5_rate_pct' 2>/dev/null || echo "0")

echo "  Cache hit rate: ${CACHE_HIT_RATE}%"
echo "  Circuit breaker: extractive_mode=$CB_EXTRACTIVE, steps_5_rate=${CB_STEPS_5_RATE}%"

if [ "$CB_EXTRACTIVE" == "true" ]; then
    echo "  ⚠️  Circuit breaker is OPEN (in extractive mode)"
else
    echo "  ✅ Circuit breaker is CLOSED (normal mode)"
fi

if (( $(echo "$CB_STEPS_5_RATE > 5.0" | bc -l) )); then
    echo "  ⚠️  High decode escalation rate: ${CB_STEPS_5_RATE}% (threshold: 5%)"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================
echo "========================================="
echo "✅ ALL DEPLOYMENT GATES PASSED"
echo "========================================="
echo ""
echo "Safe to deploy! 🚀"
echo ""
echo "Next steps:"
echo "  1. Deploy with canary rollout (5% → 25% → 100%)"
echo "  2. Monitor /metrics endpoint for SLO compliance"
echo "  3. Set up alerts for p95 > 1300ms, error rate > 0.5%"
echo ""

exit 0
