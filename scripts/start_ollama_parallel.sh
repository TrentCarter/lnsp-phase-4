#!/bin/bash
# Start Ollama with parallel request support
#
# This enables concurrent LLM requests for the ingestion pipeline.
# Without this, Ollama processes requests sequentially (slow!).
#
# Usage:
#   ./scripts/start_ollama_parallel.sh
#
# For debugging:
#   OLLAMA_DEBUG=1 ./scripts/start_ollama_parallel.sh

set -e

echo "🚀 Starting Ollama with parallel request support..."
echo ""

# Enable parallel processing
export OLLAMA_NUM_PARALLEL=10
echo "   OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL (10 concurrent requests)"

# Optional: Increase context size for longer prompts
export OLLAMA_MAX_LOADED_MODELS=3
echo "   OLLAMA_MAX_LOADED_MODELS=$OLLAMA_MAX_LOADED_MODELS"

# Optional: Debug mode
if [ "${OLLAMA_DEBUG:-0}" = "1" ]; then
    export OLLAMA_DEBUG=1
    echo "   OLLAMA_DEBUG=1 (verbose logging)"
fi

echo ""
echo "Starting Ollama server..."
echo "   Endpoint: http://localhost:11434"
echo "   Models: tinyllama:1.1b, llama3.1:8b"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start Ollama (foreground so logs are visible)
ollama serve
