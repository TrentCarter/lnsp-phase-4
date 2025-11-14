#!/bin/bash
# Start Manager-Code services (Mgr-Code-01, Mgr-Code-02, Mgr-Code-03)

echo "=== Starting Manager-Code Services ==="

# Start Mgr-Code-01 (port 6141)
MANAGER_ID="Mgr-Code-01" MANAGER_PORT=6141 MANAGER_LLM="qwen2.5-coder:7b" \
  ./.venv/bin/uvicorn services.pas.manager_code.app:app \
  --host 127.0.0.1 --port 6141 --log-level info \
  > logs/manager_code_01.log 2>&1 &

# Start Mgr-Code-02 (port 6142)
MANAGER_ID="Mgr-Code-02" MANAGER_PORT=6142 MANAGER_LLM="qwen2.5-coder:7b" \
  ./.venv/bin/uvicorn services.pas.manager_code.app:app \
  --host 127.0.0.1 --port 6142 --log-level info \
  > logs/manager_code_02.log 2>&1 &

# Start Mgr-Code-03 (port 6143)
MANAGER_ID="Mgr-Code-03" MANAGER_PORT=6143 MANAGER_LLM="qwen2.5-coder:7b" \
  ./.venv/bin/uvicorn services.pas.manager_code.app:app \
  --host 127.0.0.1 --port 6143 --log-level info \
  > logs/manager_code_03.log 2>&1 &

sleep 3

echo ""
echo "=== Health Checks ==="
curl -s http://localhost:6141/health | jq -r '"Mgr-Code-01 (6141): ✓"' 2>/dev/null || echo "Mgr-Code-01 (6141): ✗"
curl -s http://localhost:6142/health | jq -r '"Mgr-Code-02 (6142): ✓"' 2>/dev/null || echo "Mgr-Code-02 (6142): ✗"
curl -s http://localhost:6143/health | jq -r '"Mgr-Code-03 (6143): ✓"' 2>/dev/null || echo "Mgr-Code-03 (6143): ✗"

echo ""
echo "=== Manager-Code Services Started ==="
