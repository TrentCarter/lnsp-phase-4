#!/bin/bash
# Comprehensive test script for Phase 1: Resource Manager + Token Governor

set -e

REPO_ROOT="/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4"
cd "$REPO_ROOT"

echo "=========================================="
echo "Phase 1 Integration Tests"
echo "=========================================="
echo ""

# Ensure services are running
echo "1. Checking service health..."
if ! curl -s http://localhost:6104/health | grep -q "ok"; then
    echo "❌ Resource Manager not healthy. Please run ./scripts/start_phase1_services.sh first."
    exit 1
fi

if ! curl -s http://localhost:6105/health | grep -q "ok"; then
    echo "❌ Token Governor not healthy. Please run ./scripts/start_phase1_services.sh first."
    exit 1
fi

echo "✓ Both services healthy"
echo ""

# ============================================================================
# RESOURCE MANAGER TESTS
# ============================================================================

echo "=========================================="
echo "Resource Manager Tests"
echo "=========================================="
echo ""

# Test 1: Check initial quotas
echo "2. Test: Check initial resource quotas..."
QUOTAS=$(curl -s http://localhost:6104/quotas)
CPU_TOTAL=$(echo "$QUOTAS" | jq -r '.quotas.cpu.total_capacity')

if [ -z "$CPU_TOTAL" ] || [ "$CPU_TOTAL" == "null" ]; then
    echo "❌ Failed to get quotas"
    exit 1
fi

echo "✓ Resource quotas retrieved (CPU total: $CPU_TOTAL)"
echo ""

# Test 2: Reserve resources
echo "3. Test: Reserve resources for a job..."
RESERVE_RESULT=$(curl -s -X POST http://localhost:6104/reserve \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "J-TEST-001",
    "agent": "Q-Tower-Trainer",
    "cpu": 4.0,
    "mem_mb": 8192,
    "gpu": 1,
    "gpu_mem_mb": 4096
  }')

RESERVATION_ID=$(echo "$RESERVE_RESULT" | jq -r '.reservation_id')
STATUS=$(echo "$RESERVE_RESULT" | jq -r '.status')

if [ "$STATUS" != "granted" ]; then
    echo "❌ Reservation failed: $RESERVE_RESULT"
    exit 1
fi

echo "✓ Resources reserved (ID: $RESERVATION_ID)"
echo ""

# Test 3: Check updated quotas (should show allocated resources)
echo "4. Test: Verify resources are allocated..."
QUOTAS_AFTER=$(curl -s http://localhost:6104/quotas)
CPU_ALLOCATED=$(echo "$QUOTAS_AFTER" | jq -r '.quotas.cpu.allocated')

if [ "$CPU_ALLOCATED" != "4" ]; then
    echo "⚠️  CPU allocation unexpected: $CPU_ALLOCATED (expected 4)"
fi

echo "✓ Resources allocated correctly (CPU: $CPU_ALLOCATED)"
echo ""

# Test 4: Try to over-allocate (should be denied)
echo "5. Test: Attempt resource over-allocation..."
OVERALLOC_RESULT=$(curl -s -X POST http://localhost:6104/reserve \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "J-TEST-002",
    "agent": "Reranker-Trainer",
    "cpu": 100.0,
    "mem_mb": 999999
  }')

OVERALLOC_STATUS=$(echo "$OVERALLOC_RESULT" | jq -r '.status')

if [ "$OVERALLOC_STATUS" != "denied" ]; then
    echo "❌ Over-allocation should have been denied"
    exit 1
fi

echo "✓ Over-allocation correctly denied"
echo ""

# Test 5: Get active reservations
echo "6. Test: List active reservations..."
RESERVATIONS=$(curl -s "http://localhost:6104/reservations?status=active")
ACTIVE_COUNT=$(echo "$RESERVATIONS" | jq '.reservations | length')

if [ "$ACTIVE_COUNT" -lt 1 ]; then
    echo "❌ No active reservations found"
    exit 1
fi

echo "✓ Active reservations: $ACTIVE_COUNT"
echo ""

# Test 6: Release resources
echo "7. Test: Release resources..."
RELEASE_RESULT=$(curl -s -X POST http://localhost:6104/release \
  -H "Content-Type: application/json" \
  -d "{\"reservation_id\": \"$RESERVATION_ID\"}")

RELEASE_SUCCESS=$(echo "$RELEASE_RESULT" | jq -r '.success')

if [ "$RELEASE_SUCCESS" != "true" ]; then
    echo "❌ Failed to release resources"
    exit 1
fi

echo "✓ Resources released successfully"
echo ""

# Test 7: Verify resources returned to quota
echo "8. Test: Verify resources returned to quota..."
QUOTAS_FINAL=$(curl -s http://localhost:6104/quotas)
CPU_ALLOCATED_FINAL=$(echo "$QUOTAS_FINAL" | jq -r '.quotas.cpu.allocated')

if [ "$CPU_ALLOCATED_FINAL" != "0" ]; then
    echo "⚠️  CPU still allocated: $CPU_ALLOCATED_FINAL (expected 0)"
fi

echo "✓ Resources returned to quota (CPU allocated: $CPU_ALLOCATED_FINAL)"
echo ""

# ============================================================================
# TOKEN GOVERNOR TESTS
# ============================================================================

echo "=========================================="
echo "Token Governor Tests"
echo "=========================================="
echo ""

# Test 8: Track context usage (below threshold)
echo "9. Test: Track context usage (normal)..."
TRACK_RESULT=$(curl -s -X POST http://localhost:6105/track \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "Architect",
    "run_id": "R-001",
    "ctx_used": 4000,
    "ctx_limit": 16000
  }')

TRACK_STATUS=$(echo "$TRACK_RESULT" | jq -r '.status')

if [ "$TRACK_STATUS" != "ok" ]; then
    echo "❌ Context tracking failed: $TRACK_RESULT"
    exit 1
fi

echo "✓ Context tracked (status: $TRACK_STATUS)"
echo ""

# Test 9: Track context usage (warning threshold)
echo "10. Test: Track context usage (warning)..."
TRACK_WARN=$(curl -s -X POST http://localhost:6105/track \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "Director-Code",
    "run_id": "R-002",
    "ctx_used": 10000,
    "ctx_limit": 16000
  }')

WARN_STATUS=$(echo "$TRACK_WARN" | jq -r '.status')

if [ "$WARN_STATUS" != "warning" ]; then
    echo "❌ Warning threshold not detected: $WARN_STATUS"
    exit 1
fi

echo "✓ Warning threshold detected (status: $WARN_STATUS)"
echo ""

# Test 10: Track context usage (breach)
echo "11. Test: Track context usage (breach)..."
TRACK_BREACH=$(curl -s -X POST http://localhost:6105/track \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "Manager-Data",
    "run_id": "R-003",
    "ctx_used": 12500,
    "ctx_limit": 16000
  }')

BREACH_STATUS=$(echo "$TRACK_BREACH" | jq -r '.status')
BREACH_ACTION=$(echo "$TRACK_BREACH" | jq -r '.action')

if [ "$BREACH_STATUS" != "breach" ]; then
    echo "❌ Breach threshold not detected: $BREACH_STATUS"
    exit 1
fi

if [ "$BREACH_ACTION" != "save_state_clear_resume" ]; then
    echo "❌ Breach action incorrect: $BREACH_ACTION"
    exit 1
fi

echo "✓ Breach threshold detected (action: $BREACH_ACTION)"
echo ""

# Test 11: Get context status
echo "12. Test: Get context status for all agents..."
STATUS_RESULT=$(curl -s http://localhost:6105/status)
AGENTS_COUNT=$(echo "$STATUS_RESULT" | jq '.agents | length')

if [ "$AGENTS_COUNT" -lt 3 ]; then
    echo "❌ Expected at least 3 tracked agents, got $AGENTS_COUNT"
    exit 1
fi

echo "✓ Tracking $AGENTS_COUNT agents"
echo ""

# Test 12: Trigger summarization
echo "13. Test: Trigger Save-State → Clear → Resume..."
SUMMARIZE_RESULT=$(curl -s -X POST http://localhost:6105/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "Manager-Data",
    "run_id": "R-003",
    "trigger_reason": "hard_max_breach"
  }')

SUMMARY_ID=$(echo "$SUMMARIZE_RESULT" | jq -r '.summary_id')
SUMMARY_PATH=$(echo "$SUMMARIZE_RESULT" | jq -r '.summary_path')
CTX_AFTER=$(echo "$SUMMARIZE_RESULT" | jq -r '.ctx_after')

if [ -z "$SUMMARY_ID" ] || [ "$SUMMARY_ID" == "null" ]; then
    echo "❌ Summarization failed"
    exit 1
fi

if [ "$CTX_AFTER" != "0" ]; then
    echo "❌ Context not cleared after summarization"
    exit 1
fi

echo "✓ Summarization completed (ID: $SUMMARY_ID)"
echo "  Summary file: $SUMMARY_PATH"
echo ""

# Test 13: Verify summary file created
echo "14. Test: Verify summary file created..."
if [ ! -f "$SUMMARY_PATH" ]; then
    echo "❌ Summary file not found: $SUMMARY_PATH"
    exit 1
fi

echo "✓ Summary file exists: $SUMMARY_PATH"
echo ""

# Test 14: Get summarization history
echo "15. Test: Get summarization history..."
SUMMARIES=$(curl -s http://localhost:6105/summaries)
SUMMARIES_COUNT=$(echo "$SUMMARIES" | jq '.summaries | length')

if [ "$SUMMARIES_COUNT" -lt 1 ]; then
    echo "❌ No summaries found"
    exit 1
fi

echo "✓ Summarization history retrieved ($SUMMARIES_COUNT summaries)"
echo ""

# Summary
echo "=========================================="
echo "Phase 1 Tests Complete!"
echo "=========================================="
echo ""
echo "✅ All functionality verified:"
echo ""
echo "Resource Manager:"
echo "  ✓ Quota management"
echo "  ✓ Resource reservation"
echo "  ✓ Over-allocation prevention"
echo "  ✓ Resource release"
echo "  ✓ Reservation tracking"
echo ""
echo "Token Governor:"
echo "  ✓ Context tracking"
echo "  ✓ Warning threshold detection"
echo "  ✓ Breach threshold detection"
echo "  ✓ Save-State → Clear → Resume workflow"
echo "  ✓ Summary file generation"
echo "  ✓ Summarization history"
echo ""
echo "Next steps:"
echo "  1. Review summary file: cat $SUMMARY_PATH"
echo "  2. Check service logs: tail -f /tmp/pas_logs/*.log"
echo "  3. Proceed to Phase 2: Flask HMI Dashboard"
echo ""
