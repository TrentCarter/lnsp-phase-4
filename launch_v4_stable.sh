#!/bin/bash
# STABLE training launcher: macOS ARM64 safety rails
# Use this to avoid crashes while we bisect the root cause

set -e

# FIX: macOS OpenMP duplicate runtime issue (ROOT CAUSE of crashes)
export KMP_DUPLICATE_LIB_OK=TRUE

# Safety: Single-threaded execution
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTORCH_NO_FORK=1

# Timestamp for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/twotower_stable_${TIMESTAMP}.log"

echo "=========================================="
echo "STABLE TRAINING (macOS ARM64)"
echo "=========================================="
echo ""
echo "Config:"
echo "  - Python 3.11 + PyTorch 2.5.0 (recommended)"
echo "  - No FAISS mining (stability first)"
echo "  - No gradient accumulation"
echo "  - No DataLoader workers"
echo "  - AdamW foreach=False"
echo "  - GRU (or use --use-lstm for more stability)"
echo ""
echo "Log: $LOG_FILE"
echo ""

# Check which Python to use
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" = "twotower311" ]; then
    PYTHON="python"
    echo "✓ Using conda environment: twotower311"
elif [ -f ".venv311/bin/python" ]; then
    PYTHON=".venv311/bin/python"
    echo "✓ Using venv: .venv311"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    echo "⚠️  Using current venv (not Python 3.11 + PyTorch 2.5.0)"
    echo "   For best stability, run: bash setup_stable_env.sh"
else
    PYTHON="python3"
    echo "⚠️  Using system Python (not recommended)"
fi

echo ""
echo "Python: $($PYTHON --version)"
echo "PyTorch: $($PYTHON -c 'import torch; print(torch.__version__)')"
echo ""

# Run training
$PYTHON tools/train_twotower_v4_stable.py \
  --pairs artifacts/twotower/pairs_v4_synth.npz \
  --bank artifacts/wikipedia_500k_corrected_vectors.npz \
  --epochs 10 \
  --bs 32 \
  --lr 5e-5 \
  --wd 0.01 \
  --tau 0.05 \
  --device cpu \
  --hidden-dim 512 \
  --out runs/twotower_stable \
  --seed 42 \
  2>&1 | tee "$LOG_FILE"

# Check exit code
if [ $? -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "✓ TRAINING COMPLETED SUCCESSFULLY"
  echo "=========================================="
  echo ""
  echo "Next steps:"
  echo "  1. Training worked! Now bisect to find crash cause"
  echo "  2. Run: bash bisect_crash_cause.sh"
  echo ""
else
  echo ""
  echo "=========================================="
  echo "✗ TRAINING FAILED"
  echo "=========================================="
  echo ""
  echo "If crash persists even in stable mode:"
  echo "  1. Try --use-lstm flag (replace GRU with LSTM)"
  echo "  2. Check log: $LOG_FILE"
  echo "  3. Verify environment: $PYTHON --version"
  echo ""
fi
