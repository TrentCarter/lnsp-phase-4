#!/bin/bash
set -e

export FAISS_NPZ_PATH=artifacts/ontology_4k_full.npz
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export FAISS_NUM_THREADS=1

echo "Running benchmark with vecRAG, BM25, Lexical, LightVec..."

./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 200 \
  --topk 10 \
  --backends vec,bm25,lex,lightvec \
  --out RAG/results/with_lightvec.jsonl

# Display summary
SUMMARY=$(ls -t RAG/results/summary_*.md 2>/dev/null | head -1)
if [ -f "$SUMMARY" ]; then
    echo ""
    echo "=== Results ==="
    cat "$SUMMARY"
fi
