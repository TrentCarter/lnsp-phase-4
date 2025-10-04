#!/bin/bash
# Tune alpha parameter for TMD re-ranking
# Tests alpha values: 0.2, 0.3 (current), 0.4, 0.5, 0.6

echo "=== TMD Alpha Parameter Tuning ==="
echo "Testing alpha values: 0.2, 0.3, 0.4, 0.5, 0.6"
echo "Dataset: 200 queries (same as comprehensive benchmark)"
echo "Estimated time: ~25 minutes (5 min × 5 alpha values)"
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

# Test each alpha value
for alpha in 0.2 0.3 0.4 0.5 0.6; do
  echo ""
  echo "=== Testing alpha=$alpha ==="
  export TMD_ALPHA=$alpha

  ./.venv/bin/python RAG/bench.py \
    --dataset self \
    --n 200 \
    --topk 10 \
    --backends vec_tmd_rerank \
    --out RAG/results/tmd_alpha_${alpha}_oct4.jsonl

  echo "✓ Alpha $alpha complete"
done

echo ""
echo "=== Alpha Tuning Complete ==="
echo "Results saved to: RAG/results/tmd_alpha_*.jsonl"
echo ""
echo "To compare results:"
echo "  ./.venv/bin/python compare_alpha_results.py"
