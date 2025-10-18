#!/bin/bash
# Continue Wikipedia 500k Ingestion from Batch 2
#
# Run this ONLY after Batch 1 validation passes
#
# Usage:
#   ./tools/continue_wikipedia_from_batch2.sh

set -e

echo "🚀 Continuing Wikipedia Ingestion from Batch 2"
echo "=============================================="
echo ""
echo "⚠️  This assumes Batch 1 validation passed!"
echo "   Press Ctrl+C within 5 seconds to abort..."
sleep 5

# Set checkpoint to batch 1 (so it starts at batch 2)
mkdir -p artifacts/ingestion_metrics
echo "1" > artifacts/ingestion_metrics/checkpoint.txt
echo "✓ Checkpoint set to Batch 1 (will resume from Batch 2)"

# Run batched ingestion (will skip batch 1, start at batch 2)
echo ""
echo "Starting batches 2-50..."
LNSP_TMD_MODE=hybrid bash tools/ingest_wikipedia_batched.sh 10000 500000

echo ""
echo "✅ Full ingestion complete!"
