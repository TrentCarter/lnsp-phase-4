#!/bin/bash
# Test Prime Directive Completion Signal

set -e

echo "🧪 Testing Prime Directive Completion Signal"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
REGISTRY_URL="http://localhost:6121"
PAS_URL="http://localhost:6200"
HMI_URL="http://localhost:6101"
RUN_ID="test-completion-$(date +%s)"
PROJECT_ID=9999

echo "📋 Test Configuration:"
echo "  Run ID: $RUN_ID"
echo "  Project ID: $PROJECT_ID"
echo ""

# Function to check service health
check_service() {
    local name=$1
    local url=$2
    echo -n "Checking $name... "
    if curl -s -f "$url/health" > /dev/null; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

# Check all services are running
echo "🔍 Step 1: Verify Services"
echo "--------------------------"
check_service "Registry" "$REGISTRY_URL" || { echo -e "${RED}Error: Registry not running${NC}"; exit 1; }
check_service "PAS Stub" "$PAS_URL" || { echo -e "${RED}Error: PAS Stub not running${NC}"; exit 1; }
check_service "HMI" "$HMI_URL" || { echo -e "${RED}Error: HMI not running${NC}"; exit 1; }
echo ""

# Start the run
echo "🚀 Step 2: Start Run"
echo "--------------------"
echo "Creating run: $RUN_ID"
START_RESPONSE=$(curl -s -X POST "$PAS_URL/pas/v1/runs/start" \
  -H "Content-Type: application/json" \
  -d "{
    \"project_id\": $PROJECT_ID,
    \"run_id\": \"$RUN_ID\",
    \"run_kind\": \"baseline\"
  }")

if echo "$START_RESPONSE" | grep -q "executing"; then
    echo -e "${GREEN}✓ Run started${NC}"
else
    echo -e "${RED}✗ Failed to start run${NC}"
    echo "Response: $START_RESPONSE"
    exit 1
fi
echo ""

# Submit tasks
echo "📝 Step 3: Submit Tasks"
echo "-----------------------"
for i in {1..3}; do
    echo -n "Submitting task $i... "
    TASK_RESPONSE=$(curl -s -X POST "$PAS_URL/pas/v1/jobcards" \
      -H "Content-Type: application/json" \
      -d "{
        \"project_id\": $PROJECT_ID,
        \"run_id\": \"$RUN_ID\",
        \"lane\": \"Code-Impl\",
        \"priority\": 0.5,
        \"payload\": {\"task_num\": $i}
      }")

    if echo "$TASK_RESPONSE" | grep -q "task_id"; then
        TASK_ID=$(echo "$TASK_RESPONSE" | grep -o '"task_id":"[^"]*"' | cut -d'"' -f4)
        echo -e "${GREEN}✓ $TASK_ID${NC}"
    else
        echo -e "${RED}✗ Failed${NC}"
        exit 1
    fi
done
echo ""

# Wait for execution to complete
echo "⏳ Step 4: Wait for Execution"
echo "-----------------------------"
echo "Tasks will take 15-45 seconds to complete (synthetic delays)..."
echo ""

MAX_WAIT=60
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATUS_RESPONSE=$(curl -s "$PAS_URL/pas/v1/runs/status?run_id=$RUN_ID")

    if echo "$STATUS_RESPONSE" | grep -q '"status":"completed"'; then
        echo -e "${GREEN}✓ Run completed!${NC}"
        echo ""
        break
    elif echo "$STATUS_RESPONSE" | grep -q '"status":"needs_review"'; then
        echo -e "${YELLOW}⚠ Run completed with failures${NC}"
        echo ""
        break
    else
        echo -ne "\r  Elapsed: ${ELAPSED}s / ${MAX_WAIT}s (Status: executing)"
        sleep 2
        ELAPSED=$((ELAPSED + 2))
    fi
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo -e "\n${RED}✗ Timeout waiting for run completion${NC}"
    exit 1
fi

# Wait a moment for PAS to send completion signal
echo "⏱️  Waiting 2s for completion signal to propagate..."
sleep 2
echo ""

# Check Registry for completion signal
echo "🔍 Step 5: Verify Completion Signal"
echo "-----------------------------------"
echo "Checking Registry action_logs for completion entry..."

DB_PATH="artifacts/registry/registry.db"
if [ ! -f "$DB_PATH" ]; then
    echo -e "${RED}✗ Registry database not found: $DB_PATH${NC}"
    exit 1
fi

COMPLETION_ENTRY=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM action_logs WHERE task_id='$RUN_ID' AND action_type='directive_complete' AND from_agent='PAS_ROOT'")

if [ "$COMPLETION_ENTRY" -eq "1" ]; then
    echo -e "${GREEN}✓ Completion signal found in Registry${NC}"
    echo ""

    echo "📊 Completion Data:"
    sqlite3 -header -column "$DB_PATH" "SELECT log_id, timestamp, from_agent, to_agent, action_type, action_name FROM action_logs WHERE task_id='$RUN_ID' AND action_type='directive_complete'"
    echo ""

    echo "📦 Action Data:"
    ACTION_DATA=$(sqlite3 "$DB_PATH" "SELECT action_data FROM action_logs WHERE task_id='$RUN_ID' AND action_type='directive_complete'")
    echo "$ACTION_DATA" | python3 -m json.tool
    echo ""
else
    echo -e "${RED}✗ Completion signal NOT found in Registry${NC}"
    echo ""
    echo "Debugging Info:"
    echo "Expected task_id: $RUN_ID"
    echo "All action logs for this run:"
    sqlite3 -header -column "$DB_PATH" "SELECT * FROM action_logs WHERE task_id='$RUN_ID'"
    exit 1
fi

# Final summary
echo "✅ Step 6: Test Summary"
echo "----------------------"
echo -e "${GREEN}All checks passed!${NC}"
echo ""
echo "Next Steps:"
echo "1. Open browser: http://localhost:6101/sequencer?task_id=$RUN_ID"
echo "2. You should see the 'END OF PROJECT' banner"
echo "3. Verify timeline has stopped auto-scrolling"
echo "4. Check browser console for: 🎯 [PRIME DIRECTIVE COMPLETE]"
echo ""
echo "To test again, run: bash scripts/test_completion_signal.sh"
