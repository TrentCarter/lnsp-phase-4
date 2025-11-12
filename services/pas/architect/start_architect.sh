#!/usr/bin/env bash
#
# Start Architect Service
#
# Port: 6110
# LLM: Claude Sonnet 4.5 (primary), Gemini 2.5 Pro (fallback)
#

set -euo pipefail

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Change to repo root
cd "$REPO_ROOT"

# Set LLM provider (default: anthropic)
export ARCHITECT_LLM_PROVIDER="${ARCHITECT_LLM_PROVIDER:-anthropic}"
export ARCHITECT_LLM="${ARCHITECT_LLM:-claude-sonnet-4-5-20250929}"

# Director endpoints (default: localhost)
export DIR_CODE_URL="${DIR_CODE_URL:-http://127.0.0.1:6111}"
export DIR_MODELS_URL="${DIR_MODELS_URL:-http://127.0.0.1:6112}"
export DIR_DATA_URL="${DIR_DATA_URL:-http://127.0.0.1:6113}"
export DIR_DEVSECOPS_URL="${DIR_DEVSECOPS_URL:-http://127.0.0.1:6114}"
export DIR_DOCS_URL="${DIR_DOCS_URL:-http://127.0.0.1:6115}"

echo "Starting Architect Service..."
echo "  Port: 6110"
echo "  LLM Provider: $ARCHITECT_LLM_PROVIDER"
echo "  LLM Model: $ARCHITECT_LLM"
echo ""

# Run service
./.venv/bin/uvicorn services.pas.architect.app:app \
    --host 127.0.0.1 \
    --port 6110 \
    --log-level info
