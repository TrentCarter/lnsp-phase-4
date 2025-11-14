#!/bin/bash
# Test script for Manager agent chat integration

echo "=== Testing Manager Agent Chat Integration ==="
echo ""

echo "1. Testing Mgr-Code-01 (6141)..."
RESPONSE_01=$(curl -s -X POST http://localhost:6141/agent_chat/receive \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "msg-test-mgr-01",
    "thread_id": "test-thread-mgr-01",
    "from_agent": "Dir-Code",
    "to_agent": "Mgr-Code-01",
    "message_type": "delegation",
    "content": "Add unit test for hello() function",
    "created_at": "2025-11-13T22:00:00Z",
    "metadata": {"files": ["tests/test_hello.py"], "test": true}
  }')
echo "$RESPONSE_01" | jq .
echo ""

echo "2. Testing Mgr-Code-02 (6142)..."
RESPONSE_02=$(curl -s -X POST http://localhost:6142/agent_chat/receive \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "msg-test-mgr-02",
    "thread_id": "test-thread-mgr-02",
    "from_agent": "Dir-Code",
    "to_agent": "Mgr-Code-02",
    "message_type": "delegation",
    "content": "Refactor authentication module",
    "created_at": "2025-11-13T22:00:00Z",
    "metadata": {"files": ["src/auth.py"], "test": true}
  }')
echo "$RESPONSE_02" | jq .
echo ""

echo "3. Testing Mgr-Code-03 (6143)..."
RESPONSE_03=$(curl -s -X POST http://localhost:6143/agent_chat/receive \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "msg-test-mgr-03",
    "thread_id": "test-thread-mgr-03",
    "from_agent": "Dir-Code",
    "to_agent": "Mgr-Code-03",
    "message_type": "delegation",
    "content": "Add error handling to API endpoints",
    "created_at": "2025-11-13T22:00:00Z",
    "metadata": {"files": ["src/api/*.py"], "test": true}
  }')
echo "$RESPONSE_03" | jq .
echo ""

echo "=== Test Complete ==="
echo ""
echo "Summary:"
echo "Mgr-Code-01: $(echo "$RESPONSE_01" | jq -r '.status // "ERROR"')"
echo "Mgr-Code-02: $(echo "$RESPONSE_02" | jq -r '.status // "ERROR"')"
echo "Mgr-Code-03: $(echo "$RESPONSE_03" | jq -r '.status // "ERROR"')"
