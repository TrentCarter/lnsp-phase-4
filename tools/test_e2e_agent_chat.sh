#!/bin/bash
# End-to-End Agent Chat Test
# Tests full delegation chain: Architect → Dir-Code → Mgr-Code-01 → Prog-Qwen-001

set -e

THREAD_ID="e2e-test-$(date +%s)"
BASE_URL="http://localhost"

echo "================================"
echo "End-to-End Agent Chat Test"
echo "================================"
echo ""
echo "Thread ID: $THREAD_ID"
echo "Chain: Architect → Dir-Code → Mgr-Code-01 → Prog-Qwen-001"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Step 1: Check Service Health${NC}"
echo "-----------------------------------"

check_service() {
    local name=$1
    local port=$2
    if lsof -ti:$port > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $name (port $port) is running"
    else
        echo -e "${YELLOW}✗${NC} $name (port $port) is NOT running"
        exit 1
    fi
}

check_service "Architect" 6110
check_service "Dir-Code" 6111
check_service "Mgr-Code-01" 6141
check_service "Prog-Qwen-001" 6130

echo ""
echo -e "${BLUE}Step 2: Send Delegation to Architect${NC}"
echo "-----------------------------------"

# Create delegation message for Architect
PAYLOAD=$(cat <<EOF
{
  "thread_id": "$THREAD_ID",
  "from_agent": "test-client",
  "message_type": "delegation",
  "content": "Create a simple Python function called add_numbers that takes two integers and returns their sum. Save it to /tmp/test_add.py",
  "metadata": {
    "task_id": "e2e-test-001",
    "priority": "high",
    "expected_file": "/tmp/test_add.py"
  }
}
EOF
)

echo "Payload:"
echo "$PAYLOAD" | jq '.'

RESPONSE=$(echo "$PAYLOAD" | curl -s -X POST $BASE_URL:6110/agent_chat/receive \
  -H "Content-Type: application/json" \
  -d @-)

echo ""
echo "Response from Architect:"
echo "$RESPONSE" | jq '.'

echo ""
echo -e "${BLUE}Step 3: Monitor SSE Events${NC}"
echo "-----------------------------------"
echo "Monitoring SSE stream for 10 seconds..."
echo "(Press Ctrl+C to stop early)"
echo ""

# Monitor SSE events
timeout 10s curl -s -N $BASE_URL:6101/stream 2>/dev/null | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
        # Extract JSON from data: prefix
        json_data="${line#data: }"

        # Check if it's related to our thread
        if echo "$json_data" | jq -e ".thread_id == \"$THREAD_ID\"" > /dev/null 2>&1; then
            agent=$(echo "$json_data" | jq -r '.agent // "unknown"')
            msg_type=$(echo "$json_data" | jq -r '.message_type // "unknown"')
            content=$(echo "$json_data" | jq -r '.content // ""')

            echo -e "${GREEN}[$agent]${NC} $msg_type: $content"
        fi
    fi
done || true

echo ""
echo -e "${BLUE}Step 4: Check Result${NC}"
echo "-----------------------------------"

if [ -f "/tmp/test_add.py" ]; then
    echo -e "${GREEN}✓${NC} File /tmp/test_add.py was created!"
    echo ""
    echo "Contents:"
    cat /tmp/test_add.py
    echo ""

    # Try to execute the function
    echo "Testing the function:"
    python3 -c "exec(open('/tmp/test_add.py').read()); print('add_numbers(5, 3) =', add_numbers(5, 3))"
else
    echo -e "${YELLOW}✗${NC} File /tmp/test_add.py was not created"
fi

echo ""
echo "================================"
echo "End-to-End Test Complete"
echo "================================"
