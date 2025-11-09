#!/bin/bash
#
# Start HMI Server (Flask Web Dashboard)
# Port 6101
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Starting HMI Server (Flask Web Dashboard)..."
echo "Port: 6101"
echo ""
echo "Access the dashboard at:"
echo "  Main Dashboard:  http://localhost:6101/"
echo "  Tree View:       http://localhost:6101/tree"
echo "  Sequencer:       http://localhost:6101/sequencer"
echo "  Actions:         http://localhost:6101/actions"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$PROJECT_ROOT"
exec ./.venv/bin/python services/webui/hmi_app.py
