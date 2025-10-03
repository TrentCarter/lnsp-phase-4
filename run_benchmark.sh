#!/bin/bash
set -e

# Prevent FAISS threading issues on macOS
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export FAISS_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

export FAISS_NPZ_PATH=artifacts/fw9k_vectors_tmd_fixed.npz
export PYTHONPATH=.

TIMESTAMP=$(date +%s)

echo "🚀 Running RAG Benchmark: vecRAG vs BM25 vs Lexical"
echo "📊 Dataset: 500 queries from 9K ontology corpus"
echo "🎯 Metrics: P@1, P@5, MRR@10, nDCG@10, Latency"
echo ""

./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 500 \
  --topk 10 \
  --backends vec,bm25,lex \
  --npz artifacts/fw9k_vectors_tmd_fixed.npz \
  --index artifacts/fw9k_ivf_flat_ip_tmd_fixed.index \
  --out RAG/results/vecrag_vs_baselines_${TIMESTAMP}.jsonl

echo ""
echo "✅ Benchmark complete!"
echo "📁 Results: RAG/results/vecrag_vs_baselines_${TIMESTAMP}.jsonl"
echo "📝 Summary: RAG/results/summary_*.md (latest)"
