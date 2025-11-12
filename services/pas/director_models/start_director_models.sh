#!/usr/bin/env bash
#
# Start Director-Models Service
#
# Port: 6112
# LLM: Claude 2.5 Flash (primary), Claude Sonnet 4.5 (fallback)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: No venv found. Run 'python -m venv .venv' first."
    exit 1
fi

# Set LLM configuration
export DIR_MODELS_LLM_PROVIDER="${DIR_MODELS_LLM_PROVIDER:-anthropic}"
export DIR_MODELS_LLM="${DIR_MODELS_LLM:-claude-sonnet-4-5-20250929}"

# Ensure API keys are set
if [ "$DIR_MODELS_LLM_PROVIDER" = "anthropic" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "Warning: ANTHROPIC_API_KEY not set. Claude will not work."
fi

if [ "$DIR_MODELS_LLM_PROVIDER" = "anthropic" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "Warning: ANTHROPIC_API_KEY not set. Claude will not work."
fi

# Set Architect URL (for reporting)
export ARCHITECT_URL="${ARCHITECT_URL:-http://127.0.0.1:6110}"

echo "=== Starting Director-Models Service ==="
echo "Port: 6112"
echo "LLM Provider: $DIR_MODELS_LLM_PROVIDER"
echo "LLM Model: $DIR_MODELS_LLM"
echo "Architect URL: $ARCHITECT_URL"
echo ""

# Start service
exec python -m uvicorn services.pas.director_models.app:app \
    --host 127.0.0.1 \
    --port 6112 \
    --reload
