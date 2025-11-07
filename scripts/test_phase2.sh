#!/bin/bash
# Integration tests for Phase 2 PAS Services

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

echo "========================================="
echo "Phase 2 Integration Tests"
echo "========================================="
echo ""

# Function to run a test
run_test() {
    local test_name=$1
    local test_command=$2
    local expected=$3

    echo -n "Testing: $test_name... "

    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Function to test HTTP endpoint
test_http() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}

    echo -n "Testing: $name... "

    local status=$(curl -s -o /dev/null -w "%{http_code}" "$url")

    if [ "$status" -eq "$expected_status" ]; then
        echo -e "${GREEN}PASS${NC} (HTTP $status)"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}FAIL${NC} (Expected HTTP $expected_status, got $status)"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Function to test JSON response field
test_json_field() {
    local name=$1
    local url=$2
    local field=$3
    local expected=$4

    echo -n "Testing: $name... "

    local value=$(curl -s "$url" | jq -r "$field")

    if [ "$value" = "$expected" ]; then
        echo -e "${GREEN}PASS${NC} ($field=$value)"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}FAIL${NC} (Expected $expected, got $value)"
        ((TESTS_FAILED++))
        return 1
    fi
}

echo -e "${BLUE}=== Phase 0+1 Services (Prerequisites) ===${NC}"
echo ""

test_http "Registry health check" "http://localhost:6121/health"
test_http "Heartbeat Monitor health check" "http://localhost:6109/health"
test_http "Resource Manager health check" "http://localhost:6104/health"
test_http "Token Governor health check" "http://localhost:6105/health"

echo ""
echo -e "${BLUE}=== Phase 2 Services (Event Stream + HMI) ===${NC}"
echo ""

test_json_field "Event Stream status" "http://localhost:6102/health" ".status" "ok"
test_json_field "Event Stream service name" "http://localhost:6102/health" ".service" "event_stream"
test_json_field "Event Stream port" "http://localhost:6102/health" ".port" "6102"

test_json_field "Flask HMI status" "http://localhost:6101/health" ".status" "ok"
test_json_field "Flask HMI service name" "http://localhost:6101/health" ".service" "hmi_app"
test_json_field "Flask HMI port" "http://localhost:6101/health" ".port" "6101"

echo ""
echo -e "${BLUE}=== HMI API Endpoints ===${NC}"
echo ""

test_http "HMI dashboard page" "http://localhost:6101/"
test_http "HMI tree view page" "http://localhost:6101/tree"
test_http "HMI services API" "http://localhost:6101/api/services"
test_http "HMI tree API" "http://localhost:6101/api/tree"
test_http "HMI metrics API" "http://localhost:6101/api/metrics"
test_http "HMI alerts API" "http://localhost:6101/api/alerts"

echo ""
echo -e "${BLUE}=== Metrics Validation ===${NC}"
echo ""

test_json_field "Total services count" "http://localhost:6101/api/metrics" ".summary.total_services" "5"
test_json_field "Healthy services count" "http://localhost:6101/api/metrics" ".summary.healthy_services" "5"

# Test health percentage (allow 100 or 100.0)
echo -n "Testing: Health percentage... "
HEALTH_PCT=$(curl -s "http://localhost:6101/api/metrics" | jq -r '.summary.health_percentage')
if [ "$HEALTH_PCT" = "100" ] || [ "$HEALTH_PCT" = "100.0" ]; then
    echo -e "${GREEN}PASS${NC} (.summary.health_percentage=$HEALTH_PCT)"
    ((TESTS_PASSED++))
else
    echo -e "${RED}FAIL${NC} (Expected 100, got $HEALTH_PCT)"
    ((TESTS_FAILED++))
fi

echo ""
echo -e "${BLUE}=== Event Stream Broadcast ===${NC}"
echo ""

# Test broadcasting an event
echo -n "Testing: Event broadcast endpoint... "
HTTP_STATUS=$(curl -s -o /tmp/broadcast_response.json -w "%{http_code}" -X POST http://localhost:6102/broadcast \
    -H "Content-Type: application/json" \
    -d '{
        "event_type": "test_event",
        "data": {
            "message": "Integration test event",
            "test_id": "phase2-integration-test"
        }
    }')

if [ "$HTTP_STATUS" -eq "200" ]; then
    STATUS=$(cat /tmp/broadcast_response.json | jq -r '.status')
    if [ "$STATUS" = "broadcasted" ]; then
        echo -e "${GREEN}PASS${NC} (Event broadcasted)"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}FAIL${NC} (Status: $STATUS)"
        ((TESTS_FAILED++))
    fi
else
    echo -e "${RED}FAIL${NC} (HTTP $HTTP_STATUS)"
    ((TESTS_FAILED++))
fi

# Verify event was buffered
echo -n "Testing: Event buffer count... "
BUFFERED=$(curl -s http://localhost:6102/health | jq -r '.buffered_events')
if [ "$BUFFERED" -ge "1" ]; then
    echo -e "${GREEN}PASS${NC} (Buffered events: $BUFFERED)"
    ((TESTS_PASSED++))
else
    echo -e "${RED}FAIL${NC} (Expected >= 1, got $BUFFERED)"
    ((TESTS_FAILED++))
fi

echo ""
echo -e "${BLUE}=== Tree Structure ===${NC}"
echo ""

# Test tree has root node
test_json_field "Tree root node name" "http://localhost:6101/api/tree" ".name" "PAS Root"

# Test tree root status (can be idle or running)
echo -n "Testing: Tree root node status... "
TREE_STATUS=$(curl -s "http://localhost:6101/api/tree" | jq -r '.status')
if [ "$TREE_STATUS" = "idle" ] || [ "$TREE_STATUS" = "running" ]; then
    echo -e "${GREEN}PASS${NC} (.status=$TREE_STATUS)"
    ((TESTS_PASSED++))
else
    echo -e "${RED}FAIL${NC} (Expected idle or running, got $TREE_STATUS)"
    ((TESTS_FAILED++))
fi

echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "Phase 2 services are working correctly."
    echo ""
    echo "Next steps:"
    echo "  1. Open http://localhost:6101 in your browser"
    echo "  2. View the dashboard and tree visualization"
    echo "  3. Register test agents to see real-time updates"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    echo "Check the logs:"
    echo "  tail -f /tmp/pas_logs/event_stream.log"
    echo "  tail -f /tmp/pas_logs/hmi_app.log"
    echo ""
    exit 1
fi
