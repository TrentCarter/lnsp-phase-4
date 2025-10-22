# GPU (MPS) vs CPU Training: Performance & Stability

**Date:** 2025-10-21
**System:** macOS M-series (ARM64) with 40 GPU cores, 16 CPU cores

---

## 🎯 Quick Answer

**For production training: Use CPU (16 cores)**

**Why:**
- ✅ **Stable** - No crashes with OpenMP fix
- ✅ **Fast enough** - 1-1.5 hours for 30 epochs (35k pairs)
- ✅ **Predictable** - No memory issues, no backend bugs
- ✅ **Well-tested** - PyTorch CPU backend is mature

**GPU (MPS) is theoretically faster, but:**
- ⚠️ **Less stable** - PyTorch MPS backend is experimental (as of 2.7.1)
- ⚠️ **May still crash** - Different issues than CPU OpenMP problem
- ⚠️ **Limited debugging** - Harder to diagnose MPS crashes

---

## 📊 Performance Comparison

### Your Hardware
- **CPU**: 16 performance cores (M-series)
- **GPU**: 40 GPU cores (Metal Performance Shaders)
- **RAM**: 128 GB unified memory

### Expected Training Times (35,901 pairs, 30 epochs)

| Device | Speed | Total Time | Stability |
|--------|-------|------------|-----------|
| **CPU (16 cores)** | ~7.8 it/s | **1-1.5 hours** | ✅ Stable (with OpenMP fix) |
| **MPS (40 cores)** | ~15-20 it/s (est.) | **0.5-1 hour** (est.) | ⚠️ May crash |
| **CPU (4 cores)** | ~4 it/s | 4-5 hours | ✅ Stable (slower) |

**Verdict:** GPU *could* be 2x faster, but stability risk isn't worth 30-45 min savings.

---

## 🔍 Why GPU Might Crash Too

The OpenMP fix (`KMP_DUPLICATE_LIB_OK=TRUE`) only fixes **CPU-specific** crashes.

**MPS has different issues:**

1. **Immature backend** - PyTorch MPS support is newer (added in 2.x)
2. **Memory pressure** - Unified memory architecture can cause issues with large tensors
3. **GRU/LSTM bugs** - Recurrent layers have known issues on MPS
4. **FAISS on MPS** - FAISS uses CPU fallback on MPS, causing mixed-device issues

**Example MPS crashes:**
```
RuntimeError: MPS backend out of memory
RuntimeError: Placeholder storage has not been allocated on MPS device
SIGABRT: Metal command buffer execution failed
```

These are **different** from the CPU OpenMP crash we just fixed.

---

## 🧪 Should You Try MPS?

**Yes, if you want to experiment** - but CPU is safer for production.

### How to Test MPS

1. **Create MPS launch script:**
```bash
#!/bin/bash
# launch_v4_mps.sh

# NOTE: KMP_DUPLICATE_LIB_OK not needed for MPS (doesn't use OpenMP)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/twotower_v4_mps_${TIMESTAMP}.log"

./.venv/bin/python tools/train_twotower_v4.py \
  --pairs artifacts/twotower/pairs_v4_synth.npz \
  --bank artifacts/wikipedia_500k_corrected_vectors.npz \
  --epochs 30 \
  --bs 32 \
  --accum 16 \
  --device mps \
  --out runs/twotower_v4_mps \
  2>&1 | tee "$LOG_FILE"
```

2. **Run test:**
```bash
chmod +x launch_v4_mps.sh
./launch_v4_mps.sh
```

3. **Monitor for crashes:**
```bash
tail -f logs/twotower_v4_mps_*.log
```

**If MPS works:** You get 2x faster training (~30-45 min)
**If MPS crashes:** Fall back to CPU (known stable)

---

## 🔧 MPS Stability Tricks (If You Try It)

### 1. Reduce batch size (lower memory pressure)
```bash
--bs 16 --accum 32  # Same effective batch (16×32=512)
```

### 2. Disable gradient accumulation initially
```bash
--bs 32 --accum 1  # Simpler optimizer state
```

### 3. Clear MPS cache periodically
```python
# In train_twotower_v4.py, add after each epoch:
if device == 'mps':
    torch.mps.empty_cache()
    torch.mps.synchronize()
```

### 4. Switch to LSTM (more stable on MPS than GRU)
```bash
# Use train_twotower_v4_stable.py with --use-lstm
python tools/train_twotower_v4_stable.py --device mps --use-lstm ...
```

### 5. Disable FAISS mining (avoid CPU↔MPS transfers)
```bash
--mine-schedule "0-30:none"  # No mining at all
```

---

## 📈 When to Use Each Device

### Use CPU (16 cores) when:
- ✅ Stability is critical (production training)
- ✅ You need reproducible results
- ✅ Training time is acceptable (1-1.5 hours)
- ✅ You're using FAISS hard negative mining

### Use MPS (40 cores) when:
- ✅ You're experimenting and can tolerate crashes
- ✅ You need faster iteration (development/prototyping)
- ✅ You disable FAISS mining (CPU↔GPU transfer issues)
- ✅ You're training without gradient accumulation

### Use CPU (4 cores) when:
- ✅ You're running other tasks in parallel
- ✅ You want to conserve power (laptop mode)

---

## 🎓 Lessons from Our Debugging

1. **CPU crashes were NOT your code** - OpenMP library conflict
2. **MPS crashes (if any) are backend issues** - Not fixable with user code
3. **PyTorch 2.7.1 is bleeding edge** - Expect rough edges on MPS
4. **CPU is the safe choice** - Mature backend, predictable performance

---

## 🚀 Recommendation

**Start with CPU (what you're doing now):**
```bash
./launch_v4_cpu.sh  # 1-1.5 hours, stable
```

**If you're feeling adventurous, try MPS in parallel:**
```bash
# Terminal 1: CPU training (safe)
./launch_v4_cpu.sh

# Terminal 2: MPS experiment (faster but risky)
./launch_v4_mps.sh

# See which finishes first or which crashes!
```

If MPS completes successfully, you can use it for future runs. If it crashes, you already have CPU as backup.

---

## 📚 References

- PyTorch MPS backend: https://pytorch.org/docs/stable/notes/mps.html
- Known MPS issues: https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+mps
- Apple Metal docs: https://developer.apple.com/metal/pytorch/
- OpenMP on macOS: https://github.com/pytorch/pytorch/issues/78490

---

**Current status:** CPU training running successfully with OpenMP fix ✅
