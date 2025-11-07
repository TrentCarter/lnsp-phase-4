#!/bin/bash
# Integration tests for Phase 5 LLM Services
# Tests all three services (Llama 3.1 8B, TinyLlama, TLC Classifier)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "Phase 5 LLM Services Integration Tests"
echo "========================================"
echo ""

PASS_COUNT=0
FAIL_COUNT=0

# Helper function to run test
run_test() {
    local test_name="$1"
    local test_command="$2"

    echo -n "Testing: $test_name ... "

    if eval "$test_command" >/dev/null 2>&1; then
        echo "✅ PASS"
        PASS_COUNT=$((PASS_COUNT + 1))
        return 0
    else
        echo "❌ FAIL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi
}

# Test 1: Health checks
echo "=== Health Checks ==="
run_test "Llama 3.1 8B health" \
    "curl -sf http://localhost:8050/health | grep -q '\"status\":\"healthy\"'"

run_test "TinyLlama health" \
    "curl -sf http://localhost:8051/health | grep -q '\"status\":\"healthy\"'"

run_test "TLC Classifier health" \
    "curl -sf http://localhost:8052/health | grep -q '\"status\":\"healthy\"'"

echo ""

# Test 2: Service info endpoints
echo "=== Service Info ==="
run_test "Llama 3.1 8B info" \
    "curl -sf http://localhost:8050/info | grep -q '\"service_name\"'"

run_test "TinyLlama info" \
    "curl -sf http://localhost:8051/info | grep -q '\"service_name\"'"

run_test "TLC Classifier info" \
    "curl -sf http://localhost:8052/info | grep -q '\"service_name\"'"

echo ""

# Test 3: Chat completions (OpenAI-compatible)
echo "=== Chat Completions API ==="

# Llama 3.1 8B chat completion
echo -n "Testing: Llama 3.1 8B chat completion ... "
CHAT_RESPONSE=$(curl -sf http://localhost:8050/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3.1:8b",
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "temperature": 0.0,
        "max_tokens": 10
    }')

if echo "$CHAT_RESPONSE" | grep -q '"choices"'; then
    echo "✅ PASS"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "❌ FAIL"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# TinyLlama chat completion
echo -n "Testing: TinyLlama chat completion ... "
CHAT_RESPONSE=$(curl -sf http://localhost:8051/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "tinyllama:1.1b",
        "messages": [{"role": "user", "content": "Say hello"}],
        "temperature": 0.7,
        "max_tokens": 20
    }')

if echo "$CHAT_RESPONSE" | grep -q '"choices"'; then
    echo "✅ PASS"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "❌ FAIL"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo ""

# Test 4: Generate API (Ollama-compatible)
echo "=== Generate API (Ollama-compatible) ==="

echo -n "Testing: Llama 3.1 8B generate ... "
GEN_RESPONSE=$(curl -sf http://localhost:8050/generate \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3.1:8b",
        "prompt": "The capital of France is",
        "stream": false,
        "options": {"temperature": 0.0, "num_predict": 5}
    }')

if echo "$GEN_RESPONSE" | grep -q '"response"'; then
    echo "✅ PASS"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "❌ FAIL"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo ""

# Test 5: TLC Domain Classification
echo "=== TLC Domain Classification ==="

echo -n "Testing: Domain classification ... "
DOMAIN_RESPONSE=$(curl -sf http://localhost:8052/classify_domain \
    -H "Content-Type: application/json" \
    -d '{
        "query": "What is the role of glucose in diabetes?",
        "top_k": 3
    }')

if echo "$DOMAIN_RESPONSE" | grep -q '"primary_domain"'; then
    echo "✅ PASS"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "❌ FAIL"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo ""

# Test 6: TLC TMD Extraction
echo "=== TLC TMD Extraction ==="

echo -n "Testing: TMD extraction ... "
TMD_RESPONSE=$(curl -sf http://localhost:8052/extract_tmd \
    -H "Content-Type: application/json" \
    -d '{
        "query": "What is machine learning?",
        "method_hint": "DENSE"
    }')

if echo "$TMD_RESPONSE" | grep -q '"tmd"'; then
    echo "✅ PASS"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo "❌ FAIL"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo ""

# Test 7: Registry Integration (if registry is running)
echo "=== Registry Integration (Optional) ==="

if curl -sf http://localhost:6121/health >/dev/null 2>&1; then
    echo "Registry is running - testing integration..."

    # Check if LLM agents are registered
    echo -n "Testing: Llama 3.1 8B registered ... "
    REGISTRY_RESPONSE=$(curl -sf "http://localhost:6121/discover?cap=reasoning")

    if echo "$REGISTRY_RESPONSE" | grep -q 'llm_llama31_8b'; then
        echo "✅ PASS"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "⚠️  SKIP (not registered)"
    fi

    echo -n "Testing: TinyLlama registered ... "
    REGISTRY_RESPONSE=$(curl -sf "http://localhost:6121/discover?cap=classification")

    if echo "$REGISTRY_RESPONSE" | grep -q 'llm_tinyllama_1b'; then
        echo "✅ PASS"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "⚠️  SKIP (not registered)"
    fi

    echo -n "Testing: TLC Classifier registered ... "
    REGISTRY_RESPONSE=$(curl -sf "http://localhost:6121/discover?cap=domain_classification")

    if echo "$REGISTRY_RESPONSE" | grep -q 'tlc_domain_classifier'; then
        echo "✅ PASS"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "⚠️  SKIP (not registered)"
    fi
else
    echo "⚠️  Registry not running - skipping registration tests"
    echo "   Start registry: ./scripts/start_phase1_services.sh"
fi

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo ""
echo "Passed: $PASS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "Total:  $((PASS_COUNT + FAIL_COUNT))"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "===================================================="
    echo "✅ All Tests Passed!"
    echo "===================================================="
    echo ""
    echo "Summary:"
    echo "  - All 3 LLM services are healthy"
    echo "  - Chat completions API working"
    echo "  - Generate API working"
    echo "  - TLC domain classification working"
    echo "  - TLC TMD extraction working"
    echo ""
    echo "Phase 5 integration tests complete!"
    exit 0
else
    echo "===================================================="
    echo "❌ Some Tests Failed"
    echo "===================================================="
    echo ""
    echo "Check service logs:"
    echo "  tail -f /tmp/llm_llama31_8b.log"
    echo "  tail -f /tmp/llm_tinyllama.log"
    echo "  tail -f /tmp/llm_tlc_classifier.log"
    exit 1
fi
