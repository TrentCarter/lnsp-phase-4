#!/bin/bash
# Start the LNSP Chunking API Server
#
# Usage:
#   ./start_chunking_api.sh          # Start on port 8001
#   ./start_chunking_api.sh 8080     # Start on custom port

PORT="${1:-8001}"

echo "========================================"
echo "🚀 Starting LNSP Chunking API"
echo "========================================"
echo ""
echo "Port: $PORT"
echo "Web UI: http://127.0.0.1:$PORT/web"
echo "API Docs: http://127.0.0.1:$PORT/docs"
echo ""
echo "Press Ctrl+C to stop"
echo "========================================"
echo ""

# Check if venv exists
if [ ! -d "./.venv" ]; then
    echo "❌ Error: .venv directory not found"
    echo "   Please create virtual environment first:"
    echo "   python3 -m venv .venv"
    echo "   ./.venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Check if llama-index is installed
if ! ./.venv/bin/python -c "import llama_index" 2>/dev/null; then
    echo "⚠️  Warning: llama-index not installed"
    echo "   Installing dependencies..."
    ./.venv/bin/pip install llama-index llama-index-embeddings-huggingface
fi

# Start the server
./.venv/bin/uvicorn app.api.chunking:app --host 127.0.0.1 --port $PORT --reload
