#!/bin/bash
# Comprehensive RAG Benchmark Script

set -e

echo "=== Comprehensive RAG Benchmark ==="
echo "Backends: vecRAG, BM25, Lexical, GraphRAG"
echo "Dataset: Self-retrieval (ontology data)"
echo "Queries: 200 samples"
echo ""

export FAISS_NPZ_PATH=artifacts/ontology_4k_full.npz
export PYTHONPATH=.

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUTFILE="RAG/results/comprehensive_benchmark_${TIMESTAMP}.jsonl"

echo "Running benchmark..."
./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 200 \
  --topk 10 \
  --backends vec,bm25,lex \
  --out "$OUTFILE"

echo ""
echo "Results written to: $OUTFILE"
echo ""

# Display summary if markdown was generated
SUMMARY=$(ls -t RAG/results/summary_*.md 2>/dev/null | head -1)
if [ -f "$SUMMARY" ]; then
    echo "=== Summary ==="
    cat "$SUMMARY"
fi
