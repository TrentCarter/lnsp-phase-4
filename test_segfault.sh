#!/bin/bash
# Quick tests to isolate the segfault cause
# Run these in order and note which one crashes

set -e

echo "=========================================="
echo "SEGFAULT ISOLATION TESTS"
echo "=========================================="
echo ""
echo "System: 128 GB RAM (OOM ruled out)"
echo "Likely: PyTorch/FAISS segfault on macOS ARM"
echo ""

# Test 1: Tiny dataset (should NOT crash)
echo "TEST 1: Tiny dataset (100 samples)"
echo "Expected: Should complete without crash"
echo ""

python3 << 'EOF'
import numpy as np
data = np.load('artifacts/twotower/pairs_v4_synth.npz')
np.savez('artifacts/twotower/pairs_tiny.npz',
         X_train=data['X_train'][:100],
         Y_train=data['Y_train'][:100],
         X_val=data['X_val'][:50],
         Y_val=data['Y_val'][:50])
print("✓ Created tiny dataset (100 train, 50 val)")
EOF

./.venv/bin/python tools/train_twotower_v4.py \
  --pairs artifacts/twotower/pairs_tiny.npz \
  --bank artifacts/wikipedia_500k_corrected_vectors.npz \
  --epochs 2 \
  --bs 8 \
  --accum 1 \
  --device cpu \
  --out runs/twotower_test1_tiny \
  > logs/test1_tiny.log 2>&1

if [ $? -eq 0 ]; then
  echo "✓ TEST 1 PASSED (tiny dataset completed)"
else
  echo "✗ TEST 1 FAILED (crashed even on tiny dataset!)"
  echo "   See logs/test1_tiny.log"
  exit 1
fi

echo ""
echo "=========================================="
echo "TEST 2: Small batch size (bs=4)"
echo "Expected: Should complete without crash"
echo ""

./.venv/bin/python tools/train_twotower_v4.py \
  --pairs artifacts/twotower/pairs_v4_synth.npz \
  --bank artifacts/wikipedia_500k_corrected_vectors.npz \
  --epochs 1 \
  --bs 4 \
  --accum 1 \
  --device cpu \
  --out runs/twotower_test2_bs4 \
  > logs/test2_bs4.log 2>&1

if [ $? -eq 0 ]; then
  echo "✓ TEST 2 PASSED (small batch size completed)"
else
  echo "✗ TEST 2 FAILED (crashed with bs=4)"
  echo "   See logs/test2_bs4.log"
  echo "   Last 20 lines:"
  tail -20 logs/test2_bs4.log
  exit 1
fi

echo ""
echo "=========================================="
echo "TEST 3: Single-threaded CPU (avoid MKL races)"
echo "Expected: Should complete without crash"
echo ""

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

./.venv/bin/python tools/train_twotower_v4.py \
  --pairs artifacts/twotower/pairs_v4_synth.npz \
  --bank artifacts/wikipedia_500k_corrected_vectors.npz \
  --epochs 1 \
  --bs 8 \
  --accum 4 \
  --device cpu \
  --out runs/twotower_test3_singlethread \
  > logs/test3_singlethread.log 2>&1

if [ $? -eq 0 ]; then
  echo "✓ TEST 3 PASSED (single-threaded completed)"
else
  echo "✗ TEST 3 FAILED (crashed even single-threaded)"
  echo "   See logs/test3_singlethread.log"
  tail -20 logs/test3_singlethread.log
  exit 1
fi

echo ""
echo "=========================================="
echo "ALL TESTS PASSED!"
echo "=========================================="
echo ""
echo "Next step: Try full training with successful config"
echo "Recommended: Use single-threaded CPU + small batch size"
