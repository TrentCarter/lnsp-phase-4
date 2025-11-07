#!/bin/bash
#
# Phase 6 Integration Tests
# Tests all 4 cloud provider adapters
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Phase 6 Integration Tests: Cloud Provider Adapters${NC}"
echo -e "${BLUE}============================================================${NC}"

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Track test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    test_name=$1
    command=$2

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    echo -e "\n${BLUE}Test $TOTAL_TESTS: $test_name${NC}"

    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}   ✅ PASS${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}   ❌ FAIL${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Function to check JSON response contains key
check_json_key() {
    response=$1
    key=$2

    if echo "$response" | jq -e ".$key" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# ============================================================================
# Health Check Tests (All 4 Adapters)
# ============================================================================

echo -e "\n${BLUE}=== Health Check Tests ===${NC}"

run_test "OpenAI adapter health check" \
    "curl -s http://localhost:8100/health | jq -e '.status == \"healthy\" or .status == \"degraded\"'"

run_test "Anthropic adapter health check" \
    "curl -s http://localhost:8101/health | jq -e '.status == \"healthy\" or .status == \"degraded\"'"

run_test "Gemini adapter health check" \
    "curl -s http://localhost:8102/health | jq -e '.status == \"healthy\" or .status == \"degraded\"'"

run_test "Grok adapter health check" \
    "curl -s http://localhost:8103/health | jq -e '.status == \"healthy\" or .status == \"degraded\"'"

# ============================================================================
# Service Info Tests (All 4 Adapters)
# ============================================================================

echo -e "\n${BLUE}=== Service Info Tests ===${NC}"

run_test "OpenAI adapter service info" \
    "curl -s http://localhost:8100/info | jq -e '.service_name and .provider == \"openai\"'"

run_test "Anthropic adapter service info" \
    "curl -s http://localhost:8101/info | jq -e '.service_name and .provider == \"anthropic\"'"

run_test "Gemini adapter service info" \
    "curl -s http://localhost:8102/info | jq -e '.service_name and .provider == \"gemini\"'"

run_test "Grok adapter service info" \
    "curl -s http://localhost:8103/info | jq -e '.service_name and .provider == \"grok\"'"

# ============================================================================
# Model Info Tests (Context Window & Cost)
# ============================================================================

echo -e "\n${BLUE}=== Model Info Tests ===${NC}"

run_test "OpenAI model context window" \
    "curl -s http://localhost:8100/info | jq -e '.model.context_window >= 16385'"

run_test "Anthropic model cost info" \
    "curl -s http://localhost:8101/info | jq -e '.model.cost_per_input_token and .model.cost_per_output_token'"

run_test "Gemini capabilities" \
    "curl -s http://localhost:8102/info | jq -e '.model.capabilities | length > 0'"

run_test "Grok model info" \
    "curl -s http://localhost:8103/info | jq -e '.model.name and .model.provider == \"grok\"'"

# ============================================================================
# Provider Router Integration Tests
# ============================================================================

echo -e "\n${BLUE}=== Provider Router Integration ===${NC}"

# Check if Provider Router is running
if ! curl -s http://localhost:6103/health > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Provider Router not running - skipping integration tests${NC}"
else
    run_test "OpenAI registered in Provider Router" \
        "curl -s http://localhost:6103/providers | jq -e '.providers[] | select(.name | contains(\"openai\"))'"

    run_test "Anthropic registered in Provider Router" \
        "curl -s http://localhost:6103/providers | jq -e '.providers[] | select(.name | contains(\"anthropic\"))'"

    run_test "Gemini registered in Provider Router" \
        "curl -s http://localhost:6103/providers | jq -e '.providers[] | select(.name | contains(\"gemini\"))'"

    run_test "Grok registered in Provider Router" \
        "curl -s http://localhost:6103/providers | jq -e '.providers[] | select(.name | contains(\"grok\"))'"
fi

# ============================================================================
# API Endpoint Tests
# ============================================================================

echo -e "\n${BLUE}=== API Endpoint Tests ===${NC}"

run_test "OpenAI root endpoint" \
    "curl -s http://localhost:8100/ | jq -e '.service and .provider == \"openai\"'"

run_test "Anthropic docs endpoint" \
    "curl -s http://localhost:8101/docs > /dev/null"

run_test "Gemini OpenAPI schema" \
    "curl -s http://localhost:8102/openapi.json | jq -e '.info.title'"

run_test "Grok endpoints list" \
    "curl -s http://localhost:8103/ | jq -e '.endpoints.chat'"

# ============================================================================
# Test Summary
# ============================================================================

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}============================================================${NC}"

echo -e "\nTotal Tests:  $TOTAL_TESTS"
echo -e "${GREEN}Passed:       $PASSED_TESTS${NC}"
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}Failed:       $FAILED_TESTS${NC}"
else
    echo -e "Failed:       $FAILED_TESTS"
fi

pass_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
echo -e "\nPass Rate:    $pass_rate%"

echo -e "\n${BLUE}============================================================${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo -e "${BLUE}============================================================${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed!${NC}"
    echo -e "${BLUE}============================================================${NC}"
    exit 1
fi
