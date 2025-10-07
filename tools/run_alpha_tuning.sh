#!/bin/bash
# Complete α-tuning workflow for Tiny Bite #2
#
# This script:
# 1. Creates validation queries from ontology data
# 2. Runs α parameter sweep
# 3. Reports optimal α value
#
# Usage:
#   bash tools/run_alpha_tuning.sh [npz_path] [index_path]
#
# Example:
#   bash tools/run_alpha_tuning.sh artifacts/ontology_13k.npz artifacts/ontology_13k_ivf_flat_ip.index

set -e  # Exit on error

# Default paths (use ontology data, NOT fw10k!)
NPZ=${1:-artifacts/ontology_13k.npz}
INDEX=${2:-artifacts/ontology_13k_ivf_flat_ip.index}
QUERIES="eval/validation_queries.jsonl"
OUTPUT="artifacts/alpha_tuning_results.json"

# α values to test (log-search)
ALPHAS="0.0 0.1 0.2 0.3 0.5"

echo "================================================================================"
echo "  α-WEIGHTED FUSION TUNING (Tiny Bite #2)"
echo "================================================================================"
echo ""
echo "NPZ:       $NPZ"
echo "Index:     $INDEX"
echo "Queries:   $QUERIES"
echo "α values:  $ALPHAS"
echo ""

# Step 1: Create validation queries (if not exists)
if [ ! -f "$QUERIES" ]; then
    echo "Step 1: Creating validation queries..."
    echo ""
    PYTHONPATH=. ./.venv/bin/python tools/create_validation_queries.py \
        --npz "$NPZ" \
        --output "$QUERIES" \
        --n 200 \
        --seed 42
    echo ""
else
    echo "Step 1: Validation queries already exist at $QUERIES (skipping creation)"
    echo ""
fi

# Step 2: Run α parameter sweep
echo "Step 2: Running α parameter sweep..."
echo ""
PYTHONPATH=. ./.venv/bin/python tools/tune_alpha_fusion.py \
    --npz "$NPZ" \
    --index "$INDEX" \
    --queries "$QUERIES" \
    --output "$OUTPUT" \
    --alphas $ALPHAS \
    --k 10

echo ""
echo "================================================================================"
echo "  α-TUNING COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to: $OUTPUT"
echo ""
echo "Next steps:"
echo "  1. Review optimal α in $OUTPUT"
echo "  2. Update calibrator config with optimal α"
echo "  3. Re-train calibrators with optimal α"
echo ""
