#!/bin/bash
# Run LLM-based TMD Re-ranking Benchmark

echo "=== Starting LLM-based TMD Benchmark ==="
echo "Queries: 50"
echo "Backends: vec (baseline) vs vec_tmd_rerank (LLM query TMD)"
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
  --n 50 \
  --topk 10 \
  --backends vec,vec_tmd_rerank \
  --out RAG/results/llm_tmd_oct4.jsonl

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: RAG/results/llm_tmd_oct4.jsonl"
