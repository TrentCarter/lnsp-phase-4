#!/bin/bash
# Run LLM-based TMD Re-ranking on same 200 queries as comprehensive test

echo "=== LLM-based TMD Benchmark (200 queries) ==="
echo "This will compare against the comprehensive test baseline"
echo "Estimated time: ~5 minutes (200 queries × 1.5s LLM call)"
echo ""

# Prevent FAISS threading issues
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export FAISS_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

# Set paths and LLM config
export FAISS_NPZ_PATH=artifacts/ontology_4k_full.npz
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"
export PYTHONPATH=.

./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 200 \
  --topk 10 \
  --backends vec_tmd_rerank \
  --out RAG/results/tmd_200_oct4.jsonl

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: RAG/results/tmd_200_oct4.jsonl"
echo ""
echo "Compare with baseline:"
echo "  Baseline (vec): RAG/results/comprehensive_200.jsonl"
echo "  TMD rerank:     RAG/results/tmd_200_oct4.jsonl"
