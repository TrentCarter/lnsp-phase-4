#!/bin/bash
# Restart all HMI backend services

echo "🔄 Restarting HMI services..."

# Stop all running services
echo "⏹️  Stopping existing services..."
pkill -f "registry_service.py"
pkill -f "heartbeat_monitor.py"
pkill -f "resource_manager.py"
pkill -f "token_governor.py"
pkill -f "hmi_app.py"

# Wait for processes to stop
sleep 2

# Start services in background
echo "▶️  Starting services..."

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PROJECT_ROOT}/.venv/bin/python"

# Start Registry (port 6121)
cd "${PROJECT_ROOT}/services/registry" && ${VENV} registry_service.py &
sleep 1

# Start Heartbeat Monitor (port 6109)
cd "${PROJECT_ROOT}/services/heartbeat_monitor" && ${VENV} heartbeat_monitor.py &
sleep 1

# Start Resource Manager (port 6104)
cd "${PROJECT_ROOT}/services/resource_manager" && ${VENV} resource_manager.py &
sleep 1

# Start Token Governor (port 6105)
cd "${PROJECT_ROOT}/services/token_governor" && ${VENV} token_governor.py &
sleep 1

# Start HMI App (port 6101)
cd "${PROJECT_ROOT}/services/webui" && ${VENV} hmi_app.py &
sleep 1

echo "✅ All services restarted!"
echo ""
echo "Service endpoints:"
echo "  Registry:          http://localhost:6121"
echo "  Heartbeat Monitor: http://localhost:6109"
echo "  Resource Manager:  http://localhost:6104"
echo "  Token Governor:    http://localhost:6105"
echo "  HMI Dashboard:     http://localhost:6101"
