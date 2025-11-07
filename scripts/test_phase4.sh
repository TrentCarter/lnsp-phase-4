#!/bin/bash
#
# Integration Tests for Phase 4 - Claude Sub-Agents
#
# Tests:
# 1. Agent registration (50 agents)
# 2. Agent discovery by capability
# 3. Agent discovery by role/tier
# 4. Agent invocation (direct by name)
# 5. Agent invocation (capability-based routing)
# 6. Agent hierarchy traversal
#

set -e

REGISTRY_URL="http://localhost:6121"
ROUTER_URL="http://localhost:6119"

echo "=================================================="
echo "Phase 4 Integration Tests - Claude Sub-Agents"
echo "=================================================="
echo ""

# Check dependencies
echo "Checking dependencies..."
if ! curl -s $REGISTRY_URL/health > /dev/null 2>&1; then
    echo "❌ Registry service not running (required)"
    exit 1
fi
echo "✅ Registry service running"

if ! curl -s $ROUTER_URL/health > /dev/null 2>&1; then
    echo "❌ Agent Router not running (required)"
    exit 1
fi
echo "✅ Agent Router running"
echo ""

# Test 1: Agent Registration
echo "Test 1: Agent Registration"
echo "============================"

# Check current registration count
AGENT_COUNT=$(curl -s $REGISTRY_URL/services | jq '[.items[] | select(.type=="agent")] | length')
echo "Currently registered: $AGENT_COUNT agents"

# Only register if not all present
if [ "$AGENT_COUNT" -lt 50 ]; then
    echo "Registering agents..."
    ./.venv/bin/python tools/register_agents.py > /tmp/phase4_registration.log 2>&1
    AGENT_COUNT=$(curl -s $REGISTRY_URL/services | jq '[.items[] | select(.type=="agent")] | length')
else
    echo "All agents already registered, skipping registration"
fi
echo "Registered agents: $AGENT_COUNT"

if [ "$AGENT_COUNT" -eq 50 ]; then
    echo "✅ Test 1 PASSED: All 50 agents registered"
else
    echo "❌ Test 1 FAILED: Expected 50 agents, got $AGENT_COUNT"
    exit 1
fi
echo ""

# Test 2: Agent Discovery by Capability
echo "Test 2: Agent Discovery by Capability"
echo "======================================"

# Discover agents with planning capability
RESPONSE=$(curl -s -X POST $ROUTER_URL/discover \
    -H "Content-Type: application/json" \
    -d '{"capabilities": ["planning"], "limit": 10}')

COUNT=$(echo $RESPONSE | jq '.count')
AGENT_NAMES=$(echo $RESPONSE | jq -r '.agents[].agent_name' | tr '\n' ', ')

echo "Found $COUNT agent(s) with 'planning' capability: $AGENT_NAMES"

if [ "$COUNT" -gt 0 ]; then
    echo "✅ Test 2 PASSED: Capability-based discovery works"
else
    echo "❌ Test 2 FAILED: No agents found with 'planning' capability"
    exit 1
fi
echo ""

# Test 3: Agent Discovery by Role
echo "Test 3: Agent Discovery by Role"
echo "================================"

# Discover coordinator agents
COORD_RESPONSE=$(curl -s -X POST $ROUTER_URL/discover \
    -H "Content-Type: application/json" \
    -d '{"agent_role": "coord", "limit": 20}')

COORD_COUNT=$(echo $COORD_RESPONSE | jq '.count')
echo "Found $COORD_COUNT coordinator agents"

# Discover execution agents
EXEC_RESPONSE=$(curl -s -X POST $ROUTER_URL/discover \
    -H "Content-Type: application/json" \
    -d '{"agent_role": "exec", "limit": 30}')

EXEC_COUNT=$(echo $EXEC_RESPONSE | jq '.count')
echo "Found $EXEC_COUNT execution agents"

# Discover system agents
SYSTEM_RESPONSE=$(curl -s -X POST $ROUTER_URL/discover \
    -H "Content-Type: application/json" \
    -d '{"agent_role": "system", "limit": 30}')

SYSTEM_COUNT=$(echo $SYSTEM_RESPONSE | jq '.count')
echo "Found $SYSTEM_COUNT system agents"

TOTAL=$(($COORD_COUNT + $EXEC_COUNT + $SYSTEM_COUNT))
echo "Total: $TOTAL agents (expected 50)"

if [ "$TOTAL" -eq 50 ]; then
    echo "✅ Test 3 PASSED: Role-based discovery works"
else
    echo "❌ Test 3 FAILED: Total agents by role ($TOTAL) != 50"
    exit 1
fi
echo ""

# Test 4: Agent Discovery by Tier
echo "Test 4: Agent Discovery by Tier"
echo "================================"

# Discover Tier 1 agents (Claude Code sub-agents)
TIER1_RESPONSE=$(curl -s -X POST $ROUTER_URL/discover \
    -H "Content-Type: application/json" \
    -d '{"tier": 1, "limit": 50}')

TIER1_COUNT=$(echo $TIER1_RESPONSE | jq '.count')
echo "Found $TIER1_COUNT Tier 1 agents (Claude Code sub-agents)"

if [ "$TIER1_COUNT" -gt 0 ]; then
    echo "✅ Test 4 PASSED: Tier-based discovery works"
else
    echo "❌ Test 4 FAILED: No Tier 1 agents found"
    exit 1
fi
echo ""

# Test 5: Direct Agent Invocation
echo "Test 5: Direct Agent Invocation"
echo "================================"

INVOKE_RESPONSE=$(curl -s -X POST $ROUTER_URL/invoke \
    -H "Content-Type: application/json" \
    -d '{
        "request_id": "test-001",
        "agent_name": "architect",
        "payload": {
            "task": "Test invocation",
            "message": "Hello from integration test"
        },
        "timeout_s": 10
    }')

STATUS=$(echo $INVOKE_RESPONSE | jq -r '.status')
AGENT=$(echo $INVOKE_RESPONSE | jq -r '.agent_name')
LATENCY=$(echo $INVOKE_RESPONSE | jq -r '.latency_ms')

echo "Invoked agent: $AGENT"
echo "Status: $STATUS"
echo "Latency: ${LATENCY}ms"

if [ "$STATUS" = "success" ]; then
    echo "✅ Test 5 PASSED: Direct agent invocation works"
else
    echo "❌ Test 5 FAILED: Invocation status = $STATUS"
    exit 1
fi
echo ""

# Test 6: Capability-Based Invocation
echo "Test 6: Capability-Based Invocation"
echo "===================================="

CAPABILITY_INVOKE=$(curl -s -X POST $ROUTER_URL/invoke \
    -H "Content-Type: application/json" \
    -d '{
        "request_id": "test-002",
        "capabilities": ["planning"],
        "payload": {
            "task": "Route by capability",
            "message": "Test capability-based routing"
        },
        "timeout_s": 10
    }')

STATUS=$(echo $CAPABILITY_INVOKE | jq -r '.status')
AGENT=$(echo $CAPABILITY_INVOKE | jq -r '.agent_name')

echo "Routed to agent: $AGENT"
echo "Status: $STATUS"

if [ "$STATUS" = "success" ]; then
    echo "✅ Test 6 PASSED: Capability-based invocation works"
else
    echo "❌ Test 6 FAILED: Invocation status = $STATUS"
    exit 1
fi
echo ""

# Test 7: Router Stats
echo "Test 7: Router Stats"
echo "===================="

STATS=$(curl -s $ROUTER_URL/stats)
TOTAL_AGENTS=$(echo $STATS | jq '.total_agents')
BY_ROLE=$(echo $STATS | jq '.by_role')

echo "Total agents in cache: $TOTAL_AGENTS"
echo "Breakdown by role:"
echo $BY_ROLE | jq .

if [ "$TOTAL_AGENTS" -eq 50 ]; then
    echo "✅ Test 7 PASSED: Router stats accurate"
else
    echo "❌ Test 7 FAILED: Expected 50 agents, stats show $TOTAL_AGENTS"
    exit 1
fi
echo ""

# Summary
echo "=================================================="
echo "All Tests Passed! ✅"
echo "=================================================="
echo ""
echo "Summary:"
echo "  - 50 agents registered"
echo "  - Discovery by capability: ✅"
echo "  - Discovery by role: ✅"
echo "  - Discovery by tier: ✅"
echo "  - Direct invocation: ✅"
echo "  - Capability-based invocation: ✅"
echo "  - Router stats: ✅"
echo ""
echo "Phase 4 integration tests complete!"
