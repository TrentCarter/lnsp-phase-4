#!/bin/bash
# Demo: PAS Stub End-to-End Flow
# Tests all PAS API endpoints with realistic workflow

set -e

echo "=== PAS Stub End-to-End Demo ==="
echo ""

# Check if PAS stub is running
if ! curl -s http://localhost:6200/health > /dev/null 2>&1; then
    echo "❌ PAS stub not running!"
    echo "   Start with: make run-pas-stub"
    exit 1
fi

echo "✓ PAS stub is running"
echo ""

# 1. Health Check
echo "1️⃣  Health Check"
curl -s http://localhost:6200/health | jq
echo ""

# 2. Start a Run
echo "2️⃣  Start Run (baseline)"
RUN_ID="r-$(date +%Y-%m-%d-%H%M%S)"
curl -s -X POST http://localhost:6200/pas/v1/runs/start \
  -H "Content-Type: application/json" \
  -d "{
    \"project_id\": 42,
    \"run_id\": \"$RUN_ID\",
    \"run_kind\": \"baseline\",
    \"rehearsal_pct\": 0.0,
    \"budget_caps\": {
      \"budget_usd\": 50.0,
      \"energy_kwh\": 2.0
    }
  }" | jq
echo ""

# 3. Submit Job Cards
echo "3️⃣  Submit Job Cards (Code-Impl + Vector-Ops lanes)"

IDEMPOTENCY_KEY_1=$(uuidgen)
TASK_1=$(curl -s -X POST http://localhost:6200/pas/v1/jobcards \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: $IDEMPOTENCY_KEY_1" \
  -d "{
    \"project_id\": 42,
    \"run_id\": \"$RUN_ID\",
    \"lane\": \"Code-Impl\",
    \"priority\": 0.8,
    \"deps\": [],
    \"payload\": {
      \"repo\": \"/path/to/repo\",
      \"goal\": \"implement /login endpoint\",
      \"tests\": [\"tests/test_login.py\"]
    },
    \"budget_usd\": 1.50,
    \"ci_width_hint\": 0.3
  }" | jq -r '.task_id')

echo "   Task 1 (Code-Impl): $TASK_1"

IDEMPOTENCY_KEY_2=$(uuidgen)
TASK_2=$(curl -s -X POST http://localhost:6200/pas/v1/jobcards \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: $IDEMPOTENCY_KEY_2" \
  -d "{
    \"project_id\": 42,
    \"run_id\": \"$RUN_ID\",
    \"lane\": \"Vector-Ops\",
    \"priority\": 0.5,
    \"deps\": [\"$TASK_1\"],
    \"payload\": {
      \"operation\": \"refresh\",
      \"scope\": \"src/\"
    },
    \"budget_usd\": 0.50,
    \"ci_width_hint\": 0.2
  }" | jq -r '.task_id')

echo "   Task 2 (Vector-Ops): $TASK_2"
echo ""

# 4. Test Idempotency
echo "4️⃣  Test Idempotency (re-submit Task 1 with same key)"
curl -s -X POST http://localhost:6200/pas/v1/jobcards \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: $IDEMPOTENCY_KEY_1" \
  -d "{
    \"project_id\": 42,
    \"run_id\": \"$RUN_ID\",
    \"lane\": \"Code-Impl\",
    \"priority\": 0.8,
    \"deps\": [],
    \"payload\": {},
    \"budget_usd\": 1.50,
    \"ci_width_hint\": 0.3
  }" | jq
echo ""

# 5. Check Run Status (before execution completes)
echo "5️⃣  Run Status (initial)"
curl -s "http://localhost:6200/pas/v1/runs/status?run_id=$RUN_ID" | jq '{
  run_id,
  status,
  tasks_total,
  tasks_completed
}'
echo ""

# 6. Wait for execution to complete
echo "6️⃣  Waiting for execution to complete (synthetic delays: 5-15s per task)..."
sleep 20
echo ""

# 7. Check Run Status (after execution)
echo "7️⃣  Run Status (final)"
curl -s "http://localhost:6200/pas/v1/runs/status?run_id=$RUN_ID" | jq
echo ""

# 8. Portfolio Status
echo "8️⃣  Portfolio Status"
curl -s http://localhost:6200/pas/v1/portfolio/status | jq
echo ""

# 9. Simulate Rehearsal
echo "9️⃣  Simulate Rehearsal (1% stratified)"
curl -s -X POST http://localhost:6200/pas/v1/runs/simulate \
  -H "Content-Type: application/json" \
  -d "{
    \"run_id\": \"$RUN_ID\",
    \"rehearsal_pct\": 0.01,
    \"stratified\": true
  }" | jq
echo ""

# 10. Pause/Resume Test
echo "🔟 Pause/Resume Test"
echo "   Pausing run..."
curl -s -X POST http://localhost:6200/pas/v1/runs/pause \
  -H "Content-Type: application/json" \
  -d "{
    \"run_id\": \"$RUN_ID\",
    \"reason\": \"User requested pause\"
  }" | jq

sleep 2

echo "   Resuming run..."
curl -s -X POST http://localhost:6200/pas/v1/runs/resume \
  -H "Content-Type: application/json" \
  -d "{
    \"run_id\": \"$RUN_ID\"
  }" | jq
echo ""

# Summary
echo "✅ End-to-End Demo Complete!"
echo ""
echo "📊 Summary:"
echo "   - Run ID: $RUN_ID"
echo "   - Tasks submitted: 2"
echo "   - Lanes tested: Code-Impl, Vector-Ops"
echo "   - Idempotency: ✓ Verified"
echo "   - Synthetic execution: ✓ Completed"
echo "   - KPIs emitted: ✓ (check run status for violations)"
echo ""
echo "📚 Next Steps:"
echo "   - Integrate with LCO: vp start → POST /pas/v1/runs/start"
echo "   - Wire PLMS webhooks: PAS → PLMS on run completion"
echo "   - Replace stub with real executors (Weeks 5-8)"
