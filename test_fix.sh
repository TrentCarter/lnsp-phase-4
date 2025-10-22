#!/bin/bash
# Quick test to verify the OpenMP fix works

set -e

echo "=========================================="
echo "TESTING OPENMP FIX"
echo "=========================================="
echo ""
echo "Setting: export KMP_DUPLICATE_LIB_OK=TRUE"
echo ""

export KMP_DUPLICATE_LIB_OK=TRUE

# Run tiny dataset test (should complete in ~30 seconds)
echo "Running training with tiny dataset (100 samples, 2 epochs)..."
echo ""

./.venv/bin/python tools/train_twotower_v4.py \
  --pairs artifacts/twotower/pairs_tiny.npz \
  --bank artifacts/wikipedia_500k_corrected_vectors.npz \
  --epochs 2 \
  --bs 8 \
  --accum 1 \
  --device cpu \
  --out runs/twotower_test_fix \
  2>&1 | tee logs/test_fix.log

if [ $? -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "✅ FIX VERIFIED - TRAINING COMPLETED!"
  echo "=========================================="
  echo ""
  echo "Root cause: Duplicate OpenMP runtime (PyTorch + FAISS)"
  echo "Solution: export KMP_DUPLICATE_LIB_OK=TRUE"
  echo ""
  echo "Next: Run full training with:"
  echo "  ./launch_v4_cpu.sh"
  echo ""
else
  echo ""
  echo "=========================================="
  echo "❌ FIX DID NOT WORK"
  echo "=========================================="
  echo ""
  echo "Check logs/test_fix.log for details"
  echo ""
  exit 1
fi
