#!/bin/bash
# Test script for Director agent chat integration

echo "=== Testing Director Agent Chat Integration ==="
echo ""

echo "1. Testing Dir-Data (6113)..."
RESPONSE_DATA=$(curl -s -X POST http://localhost:6113/agent_chat/receive \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "msg-test-data-001",
    "thread_id": "test-coverage-dir-data",
    "run_id": "test-coverage-001",
    "from_agent": "Architect",
    "to_agent": "Dir-Data",
    "message_type": "delegation",
    "content": "Test agent chat integration for Dir-Data",
    "created_at": "2025-11-13T22:00:00Z",
    "metadata": {"test": true}
  }')
echo "$RESPONSE_DATA" | jq .
echo ""

echo "2. Testing Dir-Docs (6115)..."
RESPONSE_DOCS=$(curl -s -X POST http://localhost:6115/agent_chat/receive \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "msg-test-docs-001",
    "thread_id": "test-coverage-dir-docs",
    "run_id": "test-coverage-002",
    "from_agent": "Architect",
    "to_agent": "Dir-Docs",
    "message_type": "delegation",
    "content": "Test agent chat integration for Dir-Docs",
    "created_at": "2025-11-13T22:00:00Z",
    "metadata": {"test": true}
  }')
echo "$RESPONSE_DOCS" | jq .
echo ""

echo "3. Testing Dir-DevSecOps (6114)..."
RESPONSE_DEVSECOPS=$(curl -s -X POST http://localhost:6114/agent_chat/receive \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "msg-test-devsecops-001",
    "thread_id": "test-coverage-dir-devsecops",
    "run_id": "test-coverage-003",
    "from_agent": "Architect",
    "to_agent": "Dir-DevSecOps",
    "message_type": "delegation",
    "content": "Test agent chat integration for Dir-DevSecOps",
    "created_at": "2025-11-13T22:00:00Z",
    "metadata": {"test": true}
  }')
echo "$RESPONSE_DEVSECOPS" | jq .
echo ""

echo "=== Test Complete ==="
echo ""
echo "Summary:"
echo "Dir-Data:      $(echo "$RESPONSE_DATA" | jq -r '.status // "ERROR"')"
echo "Dir-Docs:      $(echo "$RESPONSE_DOCS" | jq -r '.status // "ERROR"')"
echo "Dir-DevSecOps: $(echo "$RESPONSE_DEVSECOPS" | jq -r '.status // "ERROR"')"
