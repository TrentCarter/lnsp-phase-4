#!/bin/bash
#
# Build Faiss IVF index from Postgres - Sprint 1 Task 1.2
#
# Usage:
#   ./scripts/build_faiss_index.sh
#
# Requirements:
#   - Postgres running with lnsp database
#   - cpe_vectors table populated with ~9,477 fused vectors
#   - Python environment with faiss-cpu, psycopg2-binary, numpy

set -e  # Exit on error

echo "============================================================"
echo "Faiss IVF Index Builder - Sprint 1 Task 1.2"
echo "============================================================"

# Change to project root
cd "$(dirname "$0")/.."

# Check if virtual environment exists
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
    echo "⚠️  No virtual environment found (.venv or venv)"
    echo "   Using system Python"
fi

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Check Postgres connection
echo ""
echo "Checking Postgres connection..."
if command -v psql &> /dev/null; then
    VECTOR_COUNT=$(psql lnsp -t -c "SELECT COUNT(*) FROM cpe_vectors;" 2>/dev/null || echo "0")
    VECTOR_COUNT=$(echo "$VECTOR_COUNT" | xargs)  # Trim whitespace
    
    if [ "$VECTOR_COUNT" = "0" ]; then
        echo "❌ No vectors found in cpe_vectors table"
        echo "   Run ingestion pipeline first"
        exit 1
    fi
    
    echo "✅ Found $VECTOR_COUNT vectors in cpe_vectors table"
else
    echo "⚠️  psql command not found, skipping database check"
fi

# Build Faiss index
echo ""
echo "Building Faiss IVF index..."
python src/faiss_builder.py \
    --output-dir artifacts \
    --nlist 128 \
    --nprobe 16

# Check if build succeeded
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Build failed"
    exit 1
fi

# Run verification tests
echo ""
echo "Running verification tests..."
python tests/test_faiss_retrieval.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ TASK 1.2 COMPLETE"
    echo "============================================================"
    echo "Deliverables:"
    echo "  - src/faiss_builder.py ✅"
    echo "  - artifacts/fw9k_ivf_flat_ip.index ✅"
    echo "  - artifacts/fw9k_cpe_ids.npy ✅"
    echo "  - tests/test_faiss_retrieval.py ✅"
    echo ""
    echo "Next steps:"
    echo "  1. Run full test suite: pytest tests/test_faiss_retrieval.py -v"
    echo "  2. Verify exit criteria (see sprint document)"
    echo "  3. Update sprint file with completion status"
    echo "============================================================"
else
    echo ""
    echo "⚠️  Verification tests failed"
    echo "   Index built but some tests did not pass"
    echo "   Run: pytest tests/test_faiss_retrieval.py -v"
fi
