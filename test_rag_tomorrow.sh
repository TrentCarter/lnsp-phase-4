#!/usr/bin/env bash
# Quick RAG test script for 4,484 ontology vectors
# Run this tomorrow morning before building the LVM

set -e

echo "🧪 Testing RAG on Ontology Data (4,484 vectors)"
echo "================================================"
echo ""

# Setup
export FAISS_NPZ_PATH=artifacts/ontology_4k_vectors.npz
mkdir -p RAG/results

# Test 1: BM25 only (no FAISS, fast sanity check)
echo "Test 1: BM25 Baseline (100 queries, ~10s)"
./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 100 \
  --topk 5 \
  --backends bm25 \
  --out RAG/results/ont_bm25.jsonl

echo ""
echo "Test 2: FAISS Dense Retrieval (100 queries, ~15s)"
./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 100 \
  --topk 5 \
  --backends vec \
  --out RAG/results/ont_vec.jsonl

echo ""
echo "Test 3: Full Comparison - vec vs BM25 (200 queries)"
./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 200 \
  --topk 10 \
  --backends vec,bm25 \
  --out RAG/results/ont_full.jsonl

echo ""
echo "✅ All tests complete!"
echo ""
echo "Results:"
ls -lh RAG/results/ont_*.jsonl
echo ""
cat RAG/results/summary_*.md | tail -20

echo ""
echo "📊 Expected Results (Self-Retrieval):"
echo "  - P@1 > 0.95 (should find exact match)"
echo "  - P@5 > 0.98 (nearly perfect)"
echo "  - vec should beat BM25 significantly"
echo ""
echo "Ready to build LVM tomorrow! 🚀"
