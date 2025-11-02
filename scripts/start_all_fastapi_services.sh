#!/bin/bash
# Start All FastAPI Services for LNSP Pipeline
# Created: 2025-10-18
#
# Required services for Wikipedia ingestion and LVM pipeline:
# - Episode Chunker (8900): Doc → Episodes
# - Semantic Chunker (8001): Text → Chunks
# - GTR-T5 Embeddings (8767): Text → 768D vectors (vec2text-compatible)
# - Ingest API (8004): Chunks → PostgreSQL + FAISS
# - Vec2Text Decoder (8766): 768D → Text (optional, for testing)
#
# Optional/Future services:
# - TMD Router (8002): Not yet implemented
# - LVM Inference (8003): Not yet implemented

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Log directory
LOG_DIR="/tmp/lnsp_api_logs"
mkdir -p "$LOG_DIR"

# Project root (assuming script is in scripts/ directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}=== LNSP FastAPI Services Startup ===${NC}"
echo "Project root: $PROJECT_ROOT"
echo "Logs: $LOG_DIR"
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Virtual environment not found at .venv${NC}"
    echo "Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Function to check if port is in use
port_in_use() {
    lsof -i ":$1" >/dev/null 2>&1
}

# Function to start a service
start_service() {
    local name="$1"
    local port="$2"
    local module="$3"
    local log_file="$LOG_DIR/${name}.log"

    if port_in_use "$port"; then
        echo -e "${YELLOW}⚠️  ${name} (port ${port}) already running${NC}"
    else
        echo -e "${GREEN}🚀 Starting ${name} on port ${port}...${NC}"
        ./.venv/bin/uvicorn "$module" --host 127.0.0.1 --port "$port" \
            > "$log_file" 2>&1 &
        local pid=$!
        echo "$pid" > "$LOG_DIR/${name}.pid"
        echo "   PID: $pid, Log: $log_file"
    fi
}

# Start services in order of dependency
echo -e "${YELLOW}Starting core services...${NC}"

# 1. Episode Chunker (needed for Wikipedia pipeline)
start_service "episode_chunker" 8900 "app.api.episode_chunker:app"

# 2. Semantic Chunker (needed for Wikipedia pipeline)
start_service "semantic_chunker" 8001 "app.api.chunking:app"

# 3. GTR-T5 Embeddings (vec2text-compatible, needed for training)
start_service "gtr_t5_embeddings" 8767 "app.api.vec2text_embedding_server:app"

# 4. Ingest API (handles PostgreSQL + FAISS + TMD extraction)
start_service "ingest_api" 8004 "app.api.ingest_chunks:app"

# 6. LVM Inference (for text-to-text generation)
start_service "lvm_inference" 9001 "app.api.lvm_inference:app"

echo ""
echo -e "${YELLOW}Optional services (for testing):${NC}"

# 5. Vec2Text Decoder (optional, for round-trip testing)
if [ "$START_VEC2TEXT_DECODER" = "1" ]; then
    start_service "vec2text_decoder" 8766 "app.api.vec2text_server:app"
else
    echo -e "${YELLOW}⏭️  Vec2Text Decoder (8766) skipped (set START_VEC2TEXT_DECODER=1 to enable)${NC}"
fi

# Wait for services to start
echo ""
echo -e "${YELLOW}Waiting 5 seconds for services to initialize...${NC}"
sleep 5

# Health checks
echo ""
echo -e "${YELLOW}=== Health Check ===${NC}"

check_health() {
    local name="$1"
    local url="$2"
    if curl -s "$url" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ ${name}: ${url}${NC}"
    else
        echo -e "${RED}❌ ${name}: ${url} (not responding)${NC}"
    fi
}

check_health "Episode Chunker" "http://localhost:8900/health"
check_health "Semantic Chunker" "http://localhost:8001/health"
check_health "GTR-T5 Embeddings" "http://localhost:8767/health"
check_health "Ingest API" "http://localhost:8004/health"
check_health "LVM Inference" "http://localhost:9001/health"

if [ "$START_VEC2TEXT_DECODER" = "1" ]; then
    check_health "Vec2Text Decoder" "http://localhost:8766/health"
fi

echo ""
echo -e "${GREEN}=== Services Started ===${NC}"
echo ""
echo "To stop all services:"
echo "  ./scripts/stop_all_fastapi_services.sh"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "Individual logs:"
ls -1 "$LOG_DIR"/*.log 2>/dev/null | while read log; do
    echo "  tail -f $log"
done
