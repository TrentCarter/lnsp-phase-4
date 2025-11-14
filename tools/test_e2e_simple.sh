#!/bin/bash
# Simple End-to-End Agent Chat Test
# Tests delegation chain: Dir-Code → Mgr-Code-01 → Prog-Qwen-001

set -e

THREAD_ID="e2e-$(date +%s)"
RUN_ID="test-run-$(date +%s)"

echo "=== End-to-End Agent Chat Test ==="
echo "Thread ID: $THREAD_ID"
echo "Run ID: $RUN_ID"
echo ""

# Test 1: Send to Dir-Code
echo "Step 1: Send delegation to Dir-Code (6111)..."
curl -s -X POST http://localhost:6111/agent_chat/receive \
  -H "Content-Type: application/json" \
  -d "{
    \"message_id\": \"msg-001\",
    \"thread_id\": \"$THREAD_ID\",
    \"run_id\": \"$RUN_ID\",
    \"from_agent\": \"Architect\",
    \"to_agent\": \"Dir-Code\",
    \"message_type\": \"delegation\",
    \"content\": \"Create a simple Python function that adds two numbers. The function should be called add_numbers and take two parameters.\",
    \"created_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
    \"metadata\": {\"test\": true, \"file_target\": \"/tmp/test_add.py\"}
  }" | jq '.'

echo ""
echo "Step 2: Wait 2 seconds for processing..."
sleep 2

# Test 2: Check if Dir-Code sent message to Mgr-Code-01
echo ""
echo "Step 3: Send test delegation to Mgr-Code-01 (6141)..."
THREAD_ID_MGR="mgr-e2e-$(date +%s)"
curl -s -X POST http://localhost:6141/agent_chat/receive \
  -H "Content-Type: application/json" \
  -d "{
    \"message_id\": \"msg-002\",
    \"thread_id\": \"$THREAD_ID_MGR\",
    \"run_id\": \"$RUN_ID\",
    \"from_agent\": \"Dir-Code\",
    \"to_agent\": \"Mgr-Code-01\",
    \"message_type\": \"delegation\",
    \"content\": \"Implement a simple add function in Python.\",
    \"created_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
    \"metadata\": {\"test\": true}
  }" | jq '.'

echo ""
echo "Step 4: Wait 2 seconds for processing..."
sleep 2

# Test 3: Send to Prog-Qwen-001 directly
echo ""
echo "Step 5: Send test delegation to Prog-Qwen-001 (6130)..."
THREAD_ID_PROG="prog-e2e-$(date +%s)"
curl -s -X POST http://localhost:6130/agent_chat/receive \
  -H "Content-Type: application/json" \
  -d "{
    \"message_id\": \"msg-003\",
    \"thread_id\": \"$THREAD_ID_PROG\",
    \"run_id\": \"$RUN_ID\",
    \"from_agent\": \"Mgr-Code-01\",
    \"to_agent\": \"Prog-Qwen-001\",
    \"message_type\": \"delegation\",
    \"content\": \"Write a Python function to add two numbers.\",
    \"created_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
    \"metadata\": {
      \"test\": true,
      \"files\": [\"/tmp/test_simple_add.py\"],
      \"task_description\": \"Create add function\"
    }
  }" | jq '.'

echo ""
echo "=== Test Complete ==="
echo ""
echo "Check HMI Sequencer at http://localhost:6101 to see agent chat messages"
echo "Look for:"
echo "  - Thread: $THREAD_ID (Dir-Code)"
echo "  - Thread: $THREAD_ID_MGR (Mgr-Code-01)"
echo "  - Thread: $THREAD_ID_PROG (Prog-Qwen-001)"
