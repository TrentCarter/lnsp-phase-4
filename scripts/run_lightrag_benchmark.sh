#!/usr/bin/env bash
set -euo pipefail

################################################################################
# LightRAG Benchmark Runner
#
# Compares three RAG approaches:
# 1. vecRAG (baseline FAISS vector search)
# 2. LightRAG (query→concept matching + graph traversal)
# 3. TMD re-ranking (vecRAG + token-matching density boost)
#
# Usage:
#   ./scripts/run_lightrag_benchmark.sh [N_QUERIES]
#
# Examples:
#   ./scripts/run_lightrag_benchmark.sh 50     # Quick test
#   ./scripts/run_lightrag_benchmark.sh 200    # Full benchmark
#   ./scripts/run_lightrag_benchmark.sh        # Default 200
################################################################################

N_QUERIES="${1:-200}"
TMD_ALPHA="${TMD_ALPHA:-0.3}"
OUTPUT_FILE="RAG/results/lightrag_vs_tmd_$(date +%Y%m%d_%H%M%S).jsonl"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║            LightRAG vs TMD Benchmark                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Queries:           $N_QUERIES"
echo "  Top-k:             10"
echo "  TMD Alpha:         $TMD_ALPHA (weight for token-matching)"
echo "  TMD Normalization: ${TMD_NORM:-softmax}"
echo "  TMD Search Pool:   ${TMD_SEARCH_MULT:-10}x (max ${TMD_SEARCH_MAX:-200})"
echo "  TMD Diagnostics:   ${TMD_DIAG:-1} (1=enabled, 0=disabled)"
echo "  Output:            $OUTPUT_FILE"
echo ""

# Check prerequisites
echo "=== Checking Prerequisites ==="

if [[ ! -f "artifacts/ontology_4k_full.npz" ]]; then
    echo "❌ FAISS vectors not found: artifacts/ontology_4k_full.npz"
    echo "   Run: ./scripts/reingest_full_4k.sh"
    exit 1
fi

if [[ ! -f "artifacts/ontology_4k_full.index" ]]; then
    echo "❌ FAISS index not found: artifacts/ontology_4k_full.index"
    echo "   Run: ./scripts/reingest_full_4k.sh"
    exit 1
fi

if ! cypher-shell -u neo4j -p password "RETURN 1" >/dev/null 2>&1; then
    echo "❌ Neo4j not accessible"
    exit 1
fi

# Check if graph has concepts with vectors
CONCEPT_COUNT=$(cypher-shell -u neo4j -p password --format plain \
    "MATCH (c:Concept) WHERE c.vector IS NOT NULL RETURN count(c)" 2>/dev/null | tail -1)

if [[ "$CONCEPT_COUNT" -lt 1000 ]]; then
    echo "❌ Neo4j graph has insufficient concepts with vectors ($CONCEPT_COUNT)"
    echo "   Expected: ~4,484 concepts"
    echo "   Run: ./scripts/reingest_full_4k.sh"
    exit 1
fi

echo "✓ FAISS vectors: artifacts/ontology_4k_full.npz"
echo "✓ FAISS index: artifacts/ontology_4k_full.index"
echo "✓ Neo4j concepts: $CONCEPT_COUNT"
echo ""

# Set environment
export FAISS_NPZ_PATH="artifacts/ontology_4k_full.npz"
export PYTHONPATH="."
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export FAISS_NUM_THREADS=1

# TMD re-ranking configuration (robust normalization + diagnostics)
export TMD_ALPHA="$TMD_ALPHA"
export TMD_NORM="${TMD_NORM:-softmax}"           # softmax normalization for stability
export TMD_TEMP="${TMD_TEMP:-1.0}"               # Temperature for softmax
export TMD_SEARCH_MULT="${TMD_SEARCH_MULT:-10}"  # Search pool multiplier (10x topk)
export TMD_SEARCH_MAX="${TMD_SEARCH_MAX:-200}"   # Max search pool size
export TMD_DIAG="${TMD_DIAG:-1}"                 # Enable diagnostics by default
export TMD_USE_LLM="${TMD_USE_LLM:-1}"           # Use LLM for TMD extraction

# LLM configuration (for TMD extraction)
export LNSP_LLM_ENDPOINT="${LNSP_LLM_ENDPOINT:-http://localhost:11434}"
export LNSP_LLM_MODEL="${LNSP_LLM_MODEL:-llama3.1:8b}"

echo "=== Running Benchmark ==="
echo "Start time: $(date)"
echo ""

# Run benchmark
./.venv/bin/python RAG/bench.py \
    --dataset self \
    --n "$N_QUERIES" \
    --topk 10 \
    --backends vec,lightrag,vec_tmd_rerank \
    --out "$OUTPUT_FILE"

BENCH_EXIT=$?

echo ""
echo "=== Benchmark Complete ==="
echo "End time: $(date)"
echo ""

if [[ $BENCH_EXIT -ne 0 ]]; then
    echo "❌ Benchmark failed with exit code $BENCH_EXIT"
    exit $BENCH_EXIT
fi

# Extract summary from JSONL
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                      RESULTS SUMMARY                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Find the summary markdown file
SUMMARY_FILE=$(ls -t RAG/results/summary_*.md 2>/dev/null | head -1)

if [[ -f "$SUMMARY_FILE" ]]; then
    cat "$SUMMARY_FILE"
    echo ""
    echo "Full summary: $SUMMARY_FILE"
else
    echo "⚠️  Summary file not found"
    echo "Results written to: $OUTPUT_FILE"
fi

echo ""
echo "Per-query results: $OUTPUT_FILE"
echo ""

# Display JSON summary lines (for downstream tools)
echo "=== JSON Summary Lines ==="
echo "(For programmatic consumption by analysis tools)"
echo ""
grep '"summary": true' "$OUTPUT_FILE" || echo "No JSON summaries found"
echo ""

# Quick analysis
echo "=== Quick Analysis ==="
echo ""

python3 <<EOF
import json
import sys

results = {"vec": [], "lightrag": [], "vec_tmd_rerank": []}

with open("$OUTPUT_FILE") as f:
    for line in f:
        try:
            obj = json.loads(line)
            if "summary" in obj:
                continue  # Skip summary lines
            backend = obj.get("backend")
            rank = obj.get("gold_rank")
            if backend and rank:
                results[backend].append(rank)
        except:
            pass

for backend, ranks in results.items():
    if not ranks:
        continue
    p1 = sum(1 for r in ranks if r == 1) / len(ranks)
    p5 = sum(1 for r in ranks if r and r <= 5) / len(ranks)
    mrr = sum(1/r for r in ranks if r) / len(ranks)

    print(f"{backend:20s} | P@1: {p1:.3f} | P@5: {p5:.3f} | MRR: {mrr:.3f}")

print("")
print("Legend:")
print("  vec             = Pure vector search (baseline)")
print("  lightrag        = Query→concept matching + graph traversal")
print("  vec_tmd_rerank  = Vector search + token-matching density boost")
EOF

echo ""
echo "✅ Benchmark complete!"
