#!/bin/bash
#
# Wikipedia Pipeline Quick Start
#
# Downloads Wikipedia data locally and ingests through FastAPI pipeline.
# All data stored in data/datasets/wikipedia/
#
# Usage:
#   # Pilot (10 articles)
#   bash scripts/wikipedia_quick_start.sh pilot
#
#   # Full (3000 articles)
#   bash scripts/wikipedia_quick_start.sh full
#

set -e  # Exit on error

MODE=${1:-pilot}

if [ "$MODE" = "pilot" ]; then
    LIMIT=10
    echo "🧪 PILOT MODE: Processing $LIMIT articles"
elif [ "$MODE" = "full" ]; then
    LIMIT=3000
    echo "🚀 FULL MODE: Processing $LIMIT articles"
else
    echo "Usage: $0 {pilot|full}"
    exit 1
fi

echo "="
echo "Wikipedia Pipeline Quick Start"
echo "="
echo ""

# Step 1: Download Wikipedia
echo "📥 Step 1: Downloading Simple English Wikipedia..."
if [ ! -f "data/datasets/wikipedia/wikipedia_simple_articles.jsonl" ]; then
    ./.venv/bin/python tools/download_wikipedia.py --limit $LIMIT
else
    echo "   ✓ Already downloaded"
fi

# Step 2: Start Episode Chunker API
echo ""
echo "🔧 Step 2: Starting Episode Chunker API (port 8900)..."
pkill -f "uvicorn app.api.episode_chunker" 2>/dev/null || true
sleep 1
./.venv/bin/uvicorn app.api.episode_chunker:app --host 127.0.0.1 --port 8900 > /dev/null 2>&1 &
sleep 2
echo "   ✓ Episode Chunker started"

# Step 3: Check all APIs
echo ""
echo "🔍 Step 3: Verifying all APIs..."
echo "   • Episode Chunker: http://localhost:8900"
echo "   • Semantic Chunker: http://localhost:8001"
echo "   • TMD Router: http://localhost:8002"
echo "   • Vec2Text GTR-T5 Embeddings: http://localhost:8767"
echo "   • Ingest: http://localhost:8004"

# Step 4: Run pipeline
echo ""
echo "⚙️  Step 4: Running ingestion pipeline..."
./.venv/bin/python tools/ingest_wikipedia_pipeline.py --limit $LIMIT

echo ""
echo "✅ Complete!"
echo ""
echo "📊 Next steps:"
echo "   1. Verify data: psql lnsp -c \"SELECT COUNT(*) FROM cpe_entry WHERE dataset_source LIKE 'wikipedia%'\""
echo "   2. Check coherence: ./.venv/bin/python tools/test_sequential_coherence.py"
echo "   3. Export training data: ./.venv/bin/python tools/export_training_sequences.py"
