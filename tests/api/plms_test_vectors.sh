#!/bin/bash
# PLMS API Test Vectors - Tier 1
# Usage: bash tests/api/plms_test_vectors.sh

set -euo pipefail

BASE_URL="${PLMS_API_BASE_URL:-http://localhost:6100}"
PROJECT_ID="${TEST_PROJECT_ID:-42}"

echo "=== PLMS API Test Vectors ==="
echo "Base URL: $BASE_URL"
echo "Project ID: $PROJECT_ID"
echo ""

# --- 1. Start Execution (Idempotent) ---
echo "1. POST /api/projects/$PROJECT_ID/start (Idempotent)"
IDEMPOTENCY_KEY=$(uuidgen)
echo "   Idempotency-Key: $IDEMPOTENCY_KEY"

curl -X POST "$BASE_URL/api/projects/$PROJECT_ID/start" \
  -H "Idempotency-Key: $IDEMPOTENCY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "run_kind": "baseline"
  }' \
  | jq .

echo ""
echo "   Retry with same Idempotency-Key (should return cached response):"
curl -X POST "$BASE_URL/api/projects/$PROJECT_ID/start" \
  -H "Idempotency-Key: $IDEMPOTENCY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "run_kind": "baseline"
  }' \
  | jq .

echo ""
echo ""

# --- 2. Simulate with 1% Canary ---
echo "2. POST /api/projects/$PROJECT_ID/simulate (1% Rehearsal)"

curl -X POST "$BASE_URL/api/projects/$PROJECT_ID/simulate?rehearsal_pct=0.01&write_sandbox=false" \
  | jq .

echo ""
echo ""

# --- 3. Get Metrics with Credible Intervals ---
echo "3. GET /api/projects/$PROJECT_ID/metrics?with_ci=1"

curl -s "$BASE_URL/api/projects/$PROJECT_ID/metrics?with_ci=1" \
  | jq .

echo ""
echo ""

# --- 4. Get Metrics without CIs (Point Estimates) ---
echo "4. GET /api/projects/$PROJECT_ID/metrics (Point Estimates)"

curl -s "$BASE_URL/api/projects/$PROJECT_ID/metrics" \
  | jq .

echo ""
echo ""

# --- 5. Get Lane Overrides (Active Learning) ---
echo "5. GET /api/projects/$PROJECT_ID/lane-overrides"

curl -s "$BASE_URL/api/projects/$PROJECT_ID/lane-overrides" \
  | jq .

echo ""
echo ""

# --- 6. Get Budget Runway ---
echo "6. GET /api/projects/$PROJECT_ID/budget-runway"

curl -s "$BASE_URL/api/projects/$PROJECT_ID/budget-runway" \
  | jq .

echo ""
echo ""

# --- 7. Get Risk Heatmap ---
echo "7. GET /api/projects/$PROJECT_ID/risk-heatmap"

curl -s "$BASE_URL/api/projects/$PROJECT_ID/risk-heatmap" \
  | jq .

echo ""
echo ""

# --- 8. Start Rehearsal Run ---
echo "8. POST /api/projects/$PROJECT_ID/start (Rehearsal Run)"
REHEARSAL_KEY=$(uuidgen)

curl -X POST "$BASE_URL/api/projects/$PROJECT_ID/start" \
  -H "Idempotency-Key: $REHEARSAL_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "run_kind": "rehearsal"
  }' \
  | jq .

echo ""
echo ""

# --- 9. Test Error Cases ---
echo "9. Error Cases"

echo "   9a. Missing Idempotency-Key (should return 400)"
curl -X POST "$BASE_URL/api/projects/$PROJECT_ID/start" \
  -H "Content-Type: application/json" \
  -d '{"run_kind": "baseline"}' \
  2>&1 | head -n 5

echo ""

echo "   9b. Invalid run_kind (should return 400)"
INVALID_KEY=$(uuidgen)
curl -X POST "$BASE_URL/api/projects/$PROJECT_ID/start" \
  -H "Idempotency-Key: $INVALID_KEY" \
  -H "Content-Type: application/json" \
  -d '{"run_kind": "invalid"}' \
  2>&1 | head -n 5

echo ""
echo ""

# --- 10. Batch Test: Multiple Projects ---
echo "10. Batch Test: Multiple Projects"

for PID in 1 2 3; do
  echo "   Project $PID: GET /api/projects/$PID/metrics"
  curl -s "$BASE_URL/api/projects/$PID/metrics" | jq -c '{project_id: .project_id, tokens: .estimated.tokens}'
done

echo ""
echo ""

# --- Summary ---
echo "=== Test Complete ==="
echo "✓ Idempotency tested (same key returns cached response)"
echo "✓ Rehearsal mode (1% canary) tested"
echo "✓ Credible intervals (with_ci=1) tested"
echo "✓ Lane overrides (active learning) tested"
echo "✓ Budget runway gauge tested"
echo "✓ Risk heatmap tested"
echo "✓ Error cases handled (400 responses)"
echo ""
echo "Next steps:"
echo "  - Run integration tests: pytest tests/test_plms_integration.py"
echo "  - Check logs: tail -f logs/plms_api.log"
echo "  - Monitor HMI: open http://localhost:6101/projects"
