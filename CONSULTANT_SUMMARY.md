# Quick Summary for Consultant

## The Problem
Training crashes silently (no error message) at random batches: 117, 464, 472, 927, 951.

**Evidence it's NOT Python:**
- No exception, no traceback
- My error handling never triggered
- Happens on both MPS (GPU) and CPU
- Log file stops mid-line

**This means:** System-level kill (OOM, segfault in C/C++ layer, or external signal)

---

## System Specs
- **OS:** macOS 15.0 (Darwin 25.0.0) ARM64
- **RAM:** **128 GB** (so OOM is unlikely!)
- **Python:** 3.13.7
- **PyTorch:** 2.7.1
- **NumPy:** 2.0.2
- **FAISS:** 1.11.0.post1

---

## Memory Footprint
- **Training data NPZ:** 11 GB (mmap'd, not all loaded)
- **Bank vectors:** 2.37 GB (771k x 768 x float32)
- **Model + optimizer:** ~76 MB (4.7M params x 4 bytes x 2 for AdamW)
- **Memory bank:** 154 MB (50k vectors FIFO queue)
- **Batch:** 0.4 MB (32 samples)
- **FAISS index (during mining):** 2.37 GB (temporary copy)

**Peak estimate:** ~5-7 GB << 128 GB available

**Conclusion:** NOT an OOM issue!

---

## Most Likely Cause
**Segfault in PyTorch GRU on macOS ARM with large tensors**

**Why:**
1. ✅ 128 GB RAM rules out OOM
2. ✅ Crashes on both MPS and CPU (different backends, same result)
3. ✅ Silent crashes = C/C++ level bug (not Python)
4. ✅ GRU operations with 771k-vector bank could hit edge case
5. ✅ macOS ARM (M-series) is newer platform (less tested than x86)

**Known issues:**
- PyTorch 2.7.x is very recent (released ~Oct 2024)
- GRU on ARM64 with MPS has had bugs: https://github.com/pytorch/pytorch/issues
- FAISS on ARM64 can have alignment issues

---

## What I Need From You

### 1. Is this a known PyTorch bug?
- Search PyTorch issues for: "GRU macOS ARM crash"
- Should I downgrade PyTorch? (e.g., 2.6.x or 2.5.x?)

### 2. Is FAISS stable on ARM64?
- FAISS builds its index during hard negative mining
- Could this trigger memory corruption?
- Should I use a different FAISS build? (faiss-cpu vs faiss-gpu-arm)

### 3. How to debug segfaults on macOS?
- Valgrind doesn't work well on ARM64 macOS
- Is there a better profiler for C/C++ crashes?
- Should I enable core dumps? (`ulimit -c unlimited`)

### 4. Could Python 3.13 be the issue?
- Python 3.13 is very new (Oct 2024)
- Could there be ABI incompatibilities with PyTorch C extensions?
- Should I downgrade to Python 3.11 or 3.12?

---

## What I'll Try Next

### Test 1: Downgrade PyTorch
```bash
pip install torch==2.5.0  # Last stable before 2.7.x
```

### Test 2: Disable FAISS (skip hard negative mining)
```python
# In train_twotower_v4.py:
hard_neg_indices = None  # Force skip mining
```

### Test 3: Replace GRU with simpler model
```python
# Use mean pooling instead of GRU
class SimpleMeanQuery(torch.nn.Module):
    def forward(self, x):
        return F.normalize(x.mean(dim=1), dim=-1)
```

### Test 4: Enable crash dumps
```bash
ulimit -c unlimited
./launch_v4_cpu.sh
# If it crashes, check for core dump file
```

---

## Full Details
See `CONSULTANT_REPORT.md` for complete diagnostic report with:
- Memory analysis
- Crash pattern analysis
- All diagnostic commands
- Code changes I made
