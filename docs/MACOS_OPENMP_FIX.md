# macOS OpenMP Crash Fix

**Critical Rule #8**: Export `KMP_DUPLICATE_LIB_OK=TRUE` for ALL CPU training on macOS

This document explains the macOS-specific OpenMP crash issue and how to fix it.

---

## The Problem: Duplicate OpenMP Libraries

### What Happens

When running CPU training on macOS, your process crashes with:

```
OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.
[1]    12345 abort      python train_model.py
```

### Why It Happens

**Root Cause**: PyTorch and FAISS both load OpenMP libraries, but they load DIFFERENT versions:

```
PyTorch → libiomp5.dylib (Intel OpenMP)
FAISS   → libomp.dylib   (LLVM OpenMP)
```

macOS's dynamic linker (dyld) detects this conflict and **kills the process** to prevent memory corruption.

### When It Happens

✅ **Affects**: CPU training on macOS (M1/M2/M3 chips or Intel)
❌ **Does NOT affect**: MPS/GPU training (doesn't use OpenMP)
❌ **Does NOT affect**: Linux (different linker behavior)

---

## The Solution: KMP_DUPLICATE_LIB_OK=TRUE

### Quick Fix

Add this to **every training script** on macOS:

```bash
#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
python tools/train_model.py ...
```

### What It Does

`KMP_DUPLICATE_LIB_OK=TRUE` tells the OpenMP runtime:
- "Yes, I know there are duplicate libraries"
- "Don't crash, just use the first one loaded"

This is safe because:
1. Both libraries implement the same OpenMP standard
2. We're not mixing OpenMP calls (PyTorch and FAISS use them internally)
3. The crash protection is overly conservative for our use case

---

## Implementation Guide

### For Shell Scripts

```bash
#!/bin/bash
# 🚨 CRITICAL: Add this line at the top of every training script on macOS
export KMP_DUPLICATE_LIB_OK=TRUE

# Training command
python tools/train_model.py \
  --model transformer \
  --batch-size 32 \
  --epochs 10 \
  --device cpu
```

### For Python Scripts

```python
import os

# 🚨 CRITICAL: Add this at the top of training scripts on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import faiss
# ... rest of imports
```

### For Jupyter Notebooks

```python
# Cell 1: Set environment variable FIRST
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Cell 2: Now safe to import PyTorch/FAISS
import torch
import faiss
```

### For Command Line (One-Time)

```bash
# Run before training (lasts for current terminal session)
export KMP_DUPLICATE_LIB_OK=TRUE

# Now run training
python tools/train_model.py ...
```

### For Persistent Setup (All Sessions)

Add to your shell profile:

```bash
# Add to ~/.zshrc (macOS default) or ~/.bashrc (if using bash)
echo 'export KMP_DUPLICATE_LIB_OK=TRUE' >> ~/.zshrc

# Reload shell
source ~/.zshrc

# Verify
echo $KMP_DUPLICATE_LIB_OK  # Should print: TRUE
```

---

## When You Need This Fix

### ✅ YES - Apply Fix

| Scenario | Device | Platform | Fix Needed? |
|----------|--------|----------|-------------|
| LVM training | CPU | macOS | ✅ YES |
| FAISS index building | CPU | macOS | ✅ YES |
| Training with CPU fallback | CPU | macOS | ✅ YES |
| Data preprocessing (if using FAISS) | CPU | macOS | ✅ YES |

### ❌ NO - Fix Not Needed

| Scenario | Device | Platform | Fix Needed? |
|----------|--------|----------|-------------|
| MPS training | MPS | macOS | ❌ NO (MPS doesn't use OpenMP) |
| GPU training | CUDA | macOS | ❌ NO (CUDA doesn't use OpenMP) |
| CPU training | CPU | Linux | ❌ NO (Linux linker allows duplicates) |
| Inference only | Any | Any | ❌ NO (usually doesn't trigger) |
| Web server (FastAPI) | Any | macOS | ⚠️ MAYBE (if using FAISS) |

---

## Real-World Examples

### Example 1: Training Script

**File**: `scripts/train_transformer_p6b_v23.sh`

```bash
#!/bin/bash

# 🚨 macOS OpenMP fix
export KMP_DUPLICATE_LIB_OK=TRUE

# Training command
./.venv/bin/python app/lvm/train_unified.py \
  --model transformer \
  --version p6b \
  --config v23 \
  --epochs 12 \
  --batch-size 32 \
  --device cpu \
  --npz artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz
```

### Example 2: FAISS Index Building

**File**: `tools/build_faiss_index.py`

```python
#!/usr/bin/env python3
import os

# 🚨 macOS OpenMP fix - MUST be before faiss import
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import faiss

def build_index(vectors_npz: str, output_index: str):
    # Load vectors
    data = np.load(vectors_npz)
    vectors = data["vectors"]

    # Build FAISS index (uses OpenMP for parallel operations)
    index = faiss.IndexFlatIP(768)
    index.add(vectors)

    # Save
    faiss.write_index(index, output_index)
    print(f"✓ Built index: {output_index}")

if __name__ == "__main__":
    build_index("artifacts/fw9k_vectors.npz", "artifacts/fw9k_flat_ip.index")
```

### Example 3: Data Preparation Pipeline

**File**: `tools/create_training_sequences.py`

```bash
#!/bin/bash

# 🚨 macOS OpenMP fix
export KMP_DUPLICATE_LIB_OK=TRUE

# Generate training sequences (uses FAISS for nearest neighbor search)
./.venv/bin/python tools/create_training_sequences.py \
  --input artifacts/wikipedia_584k_fresh.npz \
  --output artifacts/lvm/training_sequences_ctx5.npz \
  --context-size 5 \
  --device cpu
```

---

## Verification

### Check if Fix is Applied

```bash
# Method 1: Check environment variable
echo $KMP_DUPLICATE_LIB_OK
# Expected: TRUE

# Method 2: Test with Python
python -c "import os; print('KMP_DUPLICATE_LIB_OK:', os.environ.get('KMP_DUPLICATE_LIB_OK', 'NOT SET'))"
# Expected: KMP_DUPLICATE_LIB_OK: TRUE
```

### Test Training Without Crash

```bash
# Set fix
export KMP_DUPLICATE_LIB_OK=TRUE

# Run minimal training test (should NOT crash)
python -c "
import torch
import faiss
import numpy as np

# Create dummy data
data = np.random.randn(100, 768).astype('float32')
index = faiss.IndexFlatIP(768)
index.add(data)

# Create dummy model
model = torch.nn.Linear(768, 768)

# Dummy training loop (5 iterations)
optimizer = torch.optim.Adam(model.parameters())
for i in range(5):
    x = torch.randn(8, 768)
    y = model(x)
    loss = y.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f'Step {i+1}/5: loss={loss.item():.4f}')

print('✓ Training test passed (no crash)')
"
```

### Expected Output

```
Step 1/5: loss=0.0123
Step 2/5: loss=-0.0045
Step 3/5: loss=0.0067
Step 4/5: loss=-0.0012
Step 5/5: loss=0.0034
✓ Training test passed (no crash)
```

---

## Troubleshooting

### Still Crashing After Setting KMP_DUPLICATE_LIB_OK?

**Problem 1**: Environment variable not propagated to subprocess

```bash
# ❌ WRONG (env var doesn't propagate)
export KMP_DUPLICATE_LIB_OK=TRUE
./scripts/train.sh  # Runs in new shell, doesn't inherit

# ✅ CORRECT (set inside script)
# Edit train.sh:
export KMP_DUPLICATE_LIB_OK=TRUE
python train_model.py
```

**Problem 2**: Set AFTER imports (Python)

```python
# ❌ WRONG (too late, libraries already loaded)
import torch
import faiss
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ✅ CORRECT (set BEFORE imports)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import faiss
```

**Problem 3**: Using venv without activation

```bash
# ❌ WRONG (venv not activated)
export KMP_DUPLICATE_LIB_OK=TRUE
python train.py  # Uses system Python, different OpenMP libs

# ✅ CORRECT (activate venv first)
source .venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE
python train.py
```

### How to Debug OpenMP Issues

```bash
# Check which OpenMP libraries are loaded
# Run this WHILE training is running (in another terminal)
ps aux | grep python  # Get PID
lsof -p <PID> | grep -i omp

# Expected output (shows both libraries):
# python  12345  user  txt  /opt/homebrew/.../libiomp5.dylib
# python  12345  user  txt  /opt/homebrew/.../libomp.dylib
```

---

## Alternative Solutions (Not Recommended)

### Option 1: Rebuild PyTorch Without OpenMP

⚠️ **Not recommended** - Complex, breaks other functionality

```bash
# Rebuild PyTorch from source without OpenMP support
pip uninstall torch
git clone https://github.com/pytorch/pytorch
cd pytorch
USE_OPENMP=0 python setup.py install
```

**Downsides**:
- Loses parallelization in PyTorch operations
- Much slower CPU training
- Hard to maintain

### Option 2: Rebuild FAISS with Intel OpenMP

⚠️ **Not recommended** - Complex, may break on Apple Silicon

```bash
# Rebuild FAISS to use Intel OpenMP (same as PyTorch)
git clone https://github.com/facebookresearch/faiss
cd faiss
cmake -DFAISS_OPT_LEVEL=avx2 -DBLA_VENDOR=OpenBLAS .
make
```

**Downsides**:
- Intel OpenMP may not be optimized for Apple Silicon
- Complex build process
- Hard to maintain

### Option 3: Use MPS Instead of CPU

✅ **Best alternative** if you have Apple Silicon (M1/M2/M3)

```bash
# Train on MPS (Metal Performance Shaders) instead of CPU
python train_model.py --device mps

# MPS doesn't use OpenMP, so no conflict!
```

**Downsides**:
- Only works on Apple Silicon Macs (M1/M2/M3)
- Not available on Intel Macs
- Some operations may fall back to CPU

---

## Summary

**Problem**: PyTorch + FAISS load duplicate OpenMP libraries → macOS kills process

**Solution**: `export KMP_DUPLICATE_LIB_OK=TRUE` before training

**When**: ✅ CPU training on macOS | ❌ NOT needed for MPS/GPU or Linux

**How**: Add to **every training script** on macOS:

```bash
#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
python train_model.py ...
```

**Verification**: Run minimal training test (5 steps) - should NOT crash

**See Also**:
- `CRASH_ROOT_CAUSE.md` - Full diagnostic analysis
- `scripts/train_*.sh` - Example training scripts with fix applied
- [Intel OpenMP Documentation](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/optimization-and-programming/openmp-support.html)
