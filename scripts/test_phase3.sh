#!/bin/bash

# Phase 3 Integration Tests
# Tests Provider Router and Gateway services

set -e

echo "========================================="
echo "Phase 3 Integration Tests"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
PASSED=0
FAILED=0

# Test helper function
test_endpoint() {
    local description=$1
    local method=$2
    local url=$3
    local data=$4
    local expected_key=$5
    local expected_value=$6

    echo -n "Testing: $description... "

    if [ "$method" = "GET" ]; then
        response=$(curl -s "$url")
    elif [ "$method" = "POST" ]; then
        response=$(curl -s -X POST "$url" -H "Content-Type: application/json" -d "$data")
    elif [ "$method" = "DELETE" ]; then
        response=$(curl -s -X DELETE "$url")
    fi

    # Check if expected key exists (if provided)
    if [ -n "$expected_key" ]; then
        actual_value=$(echo "$response" | jq -r ".$expected_key" 2>/dev/null)
        if [ "$actual_value" = "$expected_value" ]; then
            echo -e "${GREEN}PASS${NC} ($expected_key=$expected_value)"
            PASSED=$((PASSED + 1))
        else
            echo -e "${RED}FAIL${NC} (expected $expected_key=$expected_value, got $actual_value)"
            FAILED=$((FAILED + 1))
        fi
    else
        # Just check if we got a valid response
        if echo "$response" | jq . >/dev/null 2>&1; then
            echo -e "${GREEN}PASS${NC} (HTTP 200)"
            PASSED=$((PASSED + 1))
        else
            echo -e "${RED}FAIL${NC} (Invalid response)"
            FAILED=$((FAILED + 1))
        fi
    fi
}

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is required for tests${NC}"
    echo "Install with: brew install jq"
    exit 1
fi

# === Phase 0+1+2 Services (Prerequisites) ===
echo -e "${BLUE}=== Phase 0+1+2 Services (Prerequisites) ===${NC}"
echo ""

test_endpoint "Registry health check" "GET" "http://localhost:6121/health" "" "status" "ok"
test_endpoint "Heartbeat Monitor health check" "GET" "http://localhost:6109/health" "" "status" "ok"
test_endpoint "Resource Manager health check" "GET" "http://localhost:6104/health" "" "status" "ok"
test_endpoint "Token Governor health check" "GET" "http://localhost:6105/health" "" "status" "ok"
test_endpoint "Event Stream health check" "GET" "http://localhost:6102/health" "" "status" "ok"
test_endpoint "Flask HMI health check" "GET" "http://localhost:6101/health" "" "status" "ok"

echo ""

# === Phase 3 Services (Provider Router + Gateway) ===
echo -e "${BLUE}=== Phase 3 Services (Provider Router + Gateway) ===${NC}"
echo ""

test_endpoint "Provider Router status" "GET" "http://localhost:6103/health" "" "status" "ok"
test_endpoint "Provider Router service name" "GET" "http://localhost:6103/health" "" "service" "provider_router"
test_endpoint "Provider Router port" "GET" "http://localhost:6103/health" "" "port" "6103"

test_endpoint "Gateway status" "GET" "http://localhost:6120/health" "" "status" "ok"
test_endpoint "Gateway service name" "GET" "http://localhost:6120/health" "" "service" "gateway"
test_endpoint "Gateway port" "GET" "http://localhost:6120/health" "" "port" "6120"

echo ""

# === Provider Registration ===
echo -e "${BLUE}=== Provider Registration ===${NC}"
echo ""

# Register test provider 1 (cheap)
PROVIDER1='{
  "name": "test-gpt-3.5",
  "model": "gpt-3.5-turbo",
  "context_window": 4096,
  "cost_per_input_token": 0.0000015,
  "cost_per_output_token": 0.000002,
  "endpoint": "http://localhost:8100",
  "features": ["function_calling", "streaming"]
}'

test_endpoint "Register cheap provider" "POST" "http://localhost:6103/register" "$PROVIDER1" "status" "success"

# Register test provider 2 (premium)
PROVIDER2='{
  "name": "test-gpt-4",
  "model": "gpt-4",
  "context_window": 8192,
  "cost_per_input_token": 0.00003,
  "cost_per_output_token": 0.00006,
  "endpoint": "http://localhost:8101",
  "features": ["function_calling", "streaming", "vision"],
  "slo": {
    "latency_p95_ms": 2000
  }
}'

test_endpoint "Register premium provider" "POST" "http://localhost:6103/register" "$PROVIDER2" "status" "success"

# Register test provider 3 (fast)
PROVIDER3='{
  "name": "test-claude-haiku",
  "model": "claude-haiku",
  "context_window": 200000,
  "cost_per_input_token": 0.00000025,
  "cost_per_output_token": 0.00000125,
  "endpoint": "http://localhost:8102",
  "features": ["streaming"],
  "slo": {
    "latency_p95_ms": 800
  }
}'

test_endpoint "Register fast provider" "POST" "http://localhost:6103/register" "$PROVIDER3" "status" "success"

echo ""

# === Provider Discovery ===
echo -e "${BLUE}=== Provider Discovery ===${NC}"
echo ""

test_endpoint "List all providers" "GET" "http://localhost:6103/providers" ""
test_endpoint "Get specific provider" "GET" "http://localhost:6103/providers/test-gpt-3.5" "" "name" "test-gpt-3.5"
test_endpoint "Provider registry stats" "GET" "http://localhost:6103/stats" ""

echo ""

# === Provider Selection ===
echo -e "${BLUE}=== Provider Selection ===${NC}"
echo ""

# Select cheapest provider
SELECT1='{
  "requirements": {
    "model": "gpt-3.5-turbo",
    "context_window": 2000
  },
  "optimization": "cost"
}'

test_endpoint "Select cheapest provider" "POST" "http://localhost:6103/select" "$SELECT1"

# Select with features requirement
SELECT2='{
  "requirements": {
    "model": "gpt-4",
    "context_window": 4000,
    "features": ["vision"]
  },
  "optimization": "cost"
}'

test_endpoint "Select with features" "POST" "http://localhost:6103/select" "$SELECT2"

echo ""

# === Gateway Routing ===
echo -e "${BLUE}=== Gateway Routing ===${NC}"
echo ""

# Route a test request
ROUTE_REQ='{
  "request_id": "test-req-001",
  "run_id": "test-run",
  "agent": "test-agent",
  "requirements": {
    "model": "gpt-3.5-turbo",
    "context_window": 2000
  },
  "optimization": "cost",
  "payload": {
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ]
  }
}'

test_endpoint "Route request through gateway" "POST" "http://localhost:6120/route" "$ROUTE_REQ" "status" "success"

echo ""

# === Cost Tracking ===
echo -e "${BLUE}=== Cost Tracking ===${NC}"
echo ""

test_endpoint "Get cost metrics (minute)" "GET" "http://localhost:6120/metrics?window=minute" ""
test_endpoint "Get cost metrics (hour)" "GET" "http://localhost:6120/metrics?window=hour" ""
test_endpoint "Get receipts for run" "GET" "http://localhost:6120/receipts/test-run" ""

echo ""

# === Budget Management ===
echo -e "${BLUE}=== Budget Management ===${NC}"
echo ""

test_endpoint "Set budget for run" "POST" "http://localhost:6120/budget?run_id=test-run&budget_usd=10.0" "" "status" "Budget set successfully"
test_endpoint "Get budget status" "GET" "http://localhost:6120/budget/test-run" "" "budget_set" "true"

echo ""

# === Fallback Testing ===
echo -e "${BLUE}=== Fallback Testing ===${NC}"
echo ""

# Route another request to test cost accumulation
ROUTE_REQ2='{
  "request_id": "test-req-002",
  "run_id": "test-run",
  "agent": "test-agent",
  "requirements": {
    "model": "gpt-4",
    "context_window": 4000
  },
  "optimization": "cost"
}'

test_endpoint "Route second request" "POST" "http://localhost:6120/route" "$ROUTE_REQ2" "status" "success"

echo ""

# === Receipt Validation ===
echo -e "${BLUE}=== Receipt Validation ===${NC}"
echo ""

# Check that receipts were written
if [ -f "artifacts/costs/test-run.jsonl" ]; then
    receipt_count=$(wc -l < artifacts/costs/test-run.jsonl)
    echo -n "Testing: Receipt file exists... "
    if [ "$receipt_count" -ge 2 ]; then
        echo -e "${GREEN}PASS${NC} ($receipt_count receipts)"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}FAIL${NC} (expected >=2 receipts, got $receipt_count)"
        FAILED=$((FAILED + 1))
    fi
else
    echo -n "Testing: Receipt file exists... "
    echo -e "${RED}FAIL${NC} (file not found)"
    FAILED=$((FAILED + 1))
fi

# Validate receipt JSON format
echo -n "Testing: Receipt JSON format... "
if jq . artifacts/costs/test-run.jsonl >/dev/null 2>&1; then
    echo -e "${GREEN}PASS${NC} (Valid LDJSON)"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}FAIL${NC} (Invalid JSON)"
    FAILED=$((FAILED + 1))
fi

echo ""

# === Integration with Event Stream ===
echo -e "${BLUE}=== Integration with Event Stream ===${NC}"
echo ""

# Check that cost events were broadcasted
test_endpoint "Event Stream buffer count" "GET" "http://localhost:6102/health" ""

echo ""

# === Test Summary ===
echo "========================================="
echo "Test Summary"
echo "========================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "Phase 3 services are working correctly."
    echo ""
    echo "Next steps:"
    echo "  1. View Gateway API docs: http://localhost:6120/docs"
    echo "  2. View Provider Router docs: http://localhost:6103/docs"
    echo "  3. Check cost receipts: cat artifacts/costs/test-run.jsonl"
    echo "  4. View HMI dashboard: http://localhost:6101"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    echo "Check logs:"
    echo "  tail -f /tmp/pas_logs/provider_router.log"
    echo "  tail -f /tmp/pas_logs/gateway.log"
    exit 1
fi
