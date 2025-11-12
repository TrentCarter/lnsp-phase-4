#!/usr/bin/bash
#
# Fix heartbeat API calls across all PAS services
#
# Issues:
# 1. update_heartbeat() -> heartbeat()
# 2. update_state() -> heartbeat()
# 3. AgentState.BUSY -> AgentState.PLANNING
# 4. AgentState.ERROR -> AgentState.FAILED
# 5. MessageType.ERROR -> MessageType.RESPONSE or MessageType.STATUS

set -e

echo "=== Fixing heartbeat API calls across all services ===="

# Fix 1: update_heartbeat() -> heartbeat()
echo "[1/5] Fixing update_heartbeat() -> heartbeat()..."
find services/pas -name "*.py" -type f -exec sed -i '' 's/heartbeat_monitor\.update_heartbeat(/heartbeat_monitor.heartbeat(/g' {} +
find services/tools -name "*.py" -type f -exec sed -i '' 's/heartbeat_monitor\.update_heartbeat(/heartbeat_monitor.heartbeat(/g' {} +

# Fix 2: update_state() -> heartbeat()
echo "[2/5] Fixing update_state() -> heartbeat()..."
find services/pas -name "*.py" -type f -exec sed -i '' 's/heartbeat_monitor\.update_state(/heartbeat_monitor.heartbeat(/g' {} +
find services/tools -name "*.py" -type f -exec sed -i '' 's/heartbeat_monitor\.update_state(/heartbeat_monitor.heartbeat(/g' {} +

# Fix 3: AgentState.BUSY -> AgentState.PLANNING (or EXECUTING depending on context)
echo "[3/5] Fixing AgentState.BUSY -> AgentState.EXECUTING..."
find services/pas -name "*.py" -type f -exec sed -i '' 's/AgentState\.BUSY/AgentState.EXECUTING/g' {} +
find services/tools -name "*.py" -type f -exec sed -i '' 's/AgentState\.BUSY/AgentState.EXECUTING/g' {} +

# Fix 4: AgentState.ERROR -> AgentState.FAILED
echo "[4/5] Fixing AgentState.ERROR -> AgentState.FAILED..."
find services/pas -name "*.py" -type f -exec sed -i '' 's/AgentState\.ERROR/AgentState.FAILED/g' {} +
find services/tools -name "*.py" -type f -exec sed -i '' 's/AgentState\.ERROR/AgentState.FAILED/g' {} +

# Fix 5: MessageType.ERROR -> MessageType.STATUS (conservatively)
echo "[5/5] Fixing MessageType.ERROR -> MessageType.STATUS..."
find services/pas -name "*.py" -type f -exec sed -i '' 's/MessageType\.ERROR/MessageType.STATUS/g' {} +
find services/tools -name "*.py" -type f -exec sed -i '' 's/MessageType\.ERROR/MessageType.STATUS/g' {} +

echo ""
echo "=== Summary of changes ==="
echo "Files modified:"
find services/pas services/tools -name "*.py" -type f -newer /tmp/fix_heartbeat_marker 2>/dev/null | wc -l
echo ""
echo "✓ All heartbeat API calls fixed"
echo "✓ AgentState enum values fixed"
echo "✓ MessageType enum values fixed"
echo ""
echo "Next steps:"
echo "1. Restart all services: bash scripts/stop_all_fastapi_services.sh && bash scripts/start_all_fastapi_services.sh"
echo "2. Run parallel execution test: ./.venv/bin/python tests/test_parallel_execution.py"
