# Training Crash Diagnostic Report
**Date:** 2025-10-21
**Model:** Two-Tower GRU (Query Tower + Identity Doc Tower)
**Issue:** Silent process crashes during training (no Python traceback)

---

## 🚨 CRITICAL: This is NOT a Python-level bug

**Evidence:**
- Process dies with NO exception, NO traceback
- Crashes happen on BOTH MPS (GPU) and CPU
- Error handling in code NEVER triggers (lines 242-319 of train_twotower_v4.py)
- Log file stops mid-progress bar output (see example below)

**This indicates:** System-level kill (OOM, segfault in PyTorch/MKL, or external signal)

---

## 📊 CRASH PATTERN

### Recent Crashes
| Date | Device | Batch # | Log File |
|------|--------|---------|----------|
| 2025-10-21 16:14 | CPU | 464 | `logs/twotower_v4_cpu_20251021_161447.log` (86KB) |
| 2025-10-21 12:21 | CPU | 472 | `logs/twotower_v4_cpu_20251021_122123.log` |
| Earlier | CPU | 927 | (mentioned by user) |
| Earlier | CPU | 951 | (mentioned by user) |
| Earlier | MPS | 117 | (mentioned by user) |

### Pattern Analysis
- **NO consistent batch number** (crashes at 117, 464, 472, 927, 951)
- **Gradient accumulation boundary?** Batches 464, 472 are close (within 8 batches)
  - Optimizer step happens every 16 batches
  - Batch 464 ÷ 16 = 29 (step), Batch 472 ÷ 16 = 29.5 (not a step)
  - **Conclusion:** NOT strictly tied to optimizer steps
- **Crash happens mid-epoch** (never at epoch boundary)
- **No warnings before crash** (process just vanishes)

### Last Lines Before Crash (Batch 472)
```
Training:  42%|████▏     | 472/1122 [00:59<01:24,  7.67it/s]
[Process died - no output after this line]
```
**Note:** Line was cut off mid-progress bar, suggesting instant kill (not graceful exit)

---

## 💾 MEMORY FOOTPRINT ANALYSIS

### Data Files
```
artifacts/twotower/pairs_v4_synth.npz: 11 GB (on disk)
  - X_train: Query vectors (training)
  - Y_train: Target vectors (training)
  - X_val: Query vectors (validation)
  - Y_val: Target vectors (validation)

artifacts/wikipedia_500k_corrected_vectors.npz: 2.1 GB (on disk)
  - vectors: (771,115, 768) float32 = 2.37 GB in memory
```

### Training Configuration (runs/twotower_v4/config.yaml)
```yaml
device: cpu
batch_size: 32
accum_steps: 16                # Gradient accumulation
epochs: 30
memory_bank_size: 50,000       # FIFO queue of recent doc vectors
hidden_dim: 512                # GRU hidden size (bidirectional)
num_layers: 1

# Model
Query Tower: GRU (bidirectional) + Linear projection
  - Input: (batch, seq_len=1, 768)
  - GRU: 768 → 512 hidden (bidirectional → 1024 output)
  - Linear: 1024 → 768
  - Output: L2-normalized 768D vectors
  - Parameters: 4,725,504 (~19 MB for weights + gradients + optimizer state)
```

### Estimated Memory Usage (CPU)
| Component | Size | Notes |
|-----------|------|-------|
| **Bank vectors** | **2.37 GB** | 771,115 x 768 x float32 (loaded once, kept in RAM) |
| **Memory bank** | 154 MB | 50,000 x 768 x float32 (FIFO queue) |
| **Model weights** | 19 MB | 4.7M params x 4 bytes |
| **Optimizer state** | 38 MB | AdamW: 2x model params (momentum + variance) |
| **Gradients** | 19 MB | Same shape as weights |
| **Batch data** | 0.4 MB | 32 x 2 x 768 x float32 (query + target) |
| **Accumulated grads** | 19 MB | Gradients for 16 steps (reused, not 16x) |
| **FAISS index (mining)** | 2.37 GB | Temporary copy of bank vectors for search |
| **Training data NPZ** | 11 GB | **Loaded at once? Or mmap?** (QUESTION FOR CONSULTANT) |
| **Python overhead** | ~500 MB | PyTorch runtime, libraries, etc. |
| **TOTAL (peak)** | **~16-27 GB** | Depends on NPZ loading strategy |

**🔴 CRITICAL QUESTION:** Does NumPy load the 11GB NPZ file entirely into RAM, or does it mmap() for lazy loading?

---

## 🧪 DIAGNOSTIC COMMANDS

### 1. Check System Memory During Training
```bash
# Start training in background
./launch_v4_cpu.sh &
TRAIN_PID=$!

# Monitor memory every 5 seconds
while kill -0 $TRAIN_PID 2>/dev/null; do
  ps -p $TRAIN_PID -o pid,vsz,rss,%mem,command
  sleep 5
done | tee memory_profile.log
```

**What to look for:**
- `RSS` (Resident Set Size) growing unbounded → **memory leak**
- `RSS` approaching system RAM limit → **OOM kill**
- Sudden drop in RSS → **process was killed**

### 2. Check for OOM Killer (macOS)
```bash
# Check system logs for memory pressure events
log show --predicate 'eventMessage contains "killed"' --last 2h | grep -i memory

# Check if process was killed by kernel
sudo dmesg | grep -i 'out of memory'
```

### 3. Profile Memory with Scalene (Python profiler)
```bash
# Install Scalene
pip install scalene

# Run training with memory profiling (will slow down training)
scalene --profile-all --reduced-profile --outfile scalene_report.html \
  ./venv/bin/python tools/train_twotower_v4.py \
    --pairs artifacts/twotower/pairs_v4_synth.npz \
    --bank artifacts/wikipedia_500k_corrected_vectors.npz \
    --epochs 1 --bs 32 --accum 16 --device cpu
```

**Opens HTML report showing:**
- Line-by-line memory usage
- Peak memory allocations
- GPU memory (if applicable)

### 4. Check PyTorch Memory Allocator
```python
# Add to training script after crash (if it prints)
import torch
print(torch.cuda.memory_summary())  # For GPU (won't help on CPU)
print(torch._C._get_allocator_backend())  # CPU allocator backend
```

### 5. Check NPZ Loading Strategy
```python
# Test if NPZ is mmap'd or loaded into RAM
import numpy as np
import psutil
import os

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1e9  # GB

# Load NPZ file
data = np.load('artifacts/twotower/pairs_v4_synth.npz')
mem_after_open = process.memory_info().rss / 1e9

# Access array (trigger loading if mmap'd)
_ = data['X_train'][0]
mem_after_access = process.memory_info().rss / 1e9

print(f"Before NPZ open: {mem_before:.2f} GB")
print(f"After NPZ open: {mem_after_open:.2f} GB")
print(f"After accessing array: {mem_after_access:.2f} GB")
print(f"Delta (open): {mem_after_open - mem_before:.2f} GB")
print(f"Delta (access): {mem_after_access - mem_after_open:.2f} GB")
```

**Expected:**
- If **mmap'd**: Delta (open) ≈ 0 MB, Delta (access) ≈ small
- If **loaded**: Delta (open) ≈ 11 GB

---

## ❓ QUESTIONS FOR CONSULTANT

### 1. Is this OOM (Out of Memory)?
- **How to verify?** See "Diagnostic Commands" above
- **System RAM:** How much RAM does the machine have?
- **Peak usage:** Is 16-27 GB estimate reasonable for this training?

### 2. Should we reduce memory footprint?
**Option A: Reduce batch size**
```bash
# Try batch_size=8 (4x reduction in batch memory)
--bs 8 --accum 64  # Keep effective batch size = 512
```

**Option B: Disable gradient accumulation**
```bash
# Simplify optimizer state (no multi-step accumulation)
--bs 32 --accum 1  # Effective batch size = 32 (much smaller)
```

**Option C: Reduce memory bank**
```bash
# Use smaller FIFO queue (50k → 10k)
--memory-bank-size 10000  # 154 MB → 31 MB (saves 123 MB)
```

### 3. Is there a PyTorch/MKL bug?
- **GRU on CPU with large banks:** Known issues with GRU + large tensor operations on CPU?
- **MKL threading:** Could multi-threaded BLAS cause memory corruption?
  ```bash
  # Try single-threaded execution
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  ./launch_v4_cpu.sh
  ```

### 4. Is FAISS causing segfaults?
- **Pattern:** Crashes happen during hard negative mining (which uses FAISS)
- **Test:** Disable hard negative mining for 1 epoch
  ```python
  # In train_twotower_v4.py, skip mining:
  hard_neg_indices = None  # Force skip
  ```

### 5. Could shuffle=False have made it worse?
- **User changed:** DataLoader shuffle=True → shuffle=False (line 528)
- **Impact:** Changed data access pattern (now sequential)
- **Could this trigger different memory pattern?** (e.g., sequential access to large NPZ file)

---

## 🔧 RECOMMENDED TROUBLESHOOTING STEPS

### Step 1: Confirm OOM vs. Segfault
```bash
# Run with memory monitoring
./launch_v4_cpu.sh &
TRAIN_PID=$!
watch -n 1 "ps -p $TRAIN_PID -o pid,vsz,rss,%mem,command"
```

**If RSS grows to system limit → OOM**
**If RSS stable when crash happens → Segfault or signal**

### Step 2: Simplify Training (Eliminate Variables)
```bash
# Test 1: No gradient accumulation
python tools/train_twotower_v4.py \
  --pairs artifacts/twotower/pairs_v4_synth.npz \
  --bank artifacts/wikipedia_500k_corrected_vectors.npz \
  --epochs 1 --bs 8 --accum 1 --device cpu \
  --out runs/twotower_debug_noaccum

# Test 2: Small batch size
python tools/train_twotower_v4.py \
  --pairs artifacts/twotower/pairs_v4_synth.npz \
  --bank artifacts/wikipedia_500k_corrected_vectors.npz \
  --epochs 1 --bs 4 --accum 1 --device cpu \
  --out runs/twotower_debug_bs4

# Test 3: Tiny subset (no crash expected)
python -c "
import numpy as np
data = np.load('artifacts/twotower/pairs_v4_synth.npz')
np.savez('artifacts/twotower/pairs_tiny.npz',
         X_train=data['X_train'][:100],
         Y_train=data['Y_train'][:100],
         X_val=data['X_val'][:100],
         Y_val=data['Y_val'][:100])
"

python tools/train_twotower_v4.py \
  --pairs artifacts/twotower/pairs_tiny.npz \
  --bank artifacts/wikipedia_500k_corrected_vectors.npz \
  --epochs 5 --bs 8 --accum 1 --device cpu \
  --out runs/twotower_debug_tiny
```

### Step 3: Single-Threaded Execution (Avoid MKL bugs)
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
./launch_v4_cpu.sh
```

### Step 4: Profile with Valgrind (Detect memory corruption)
```bash
# Install Valgrind (if not present)
brew install valgrind  # macOS (might need x86 Rosetta)

# Run training under Valgrind (VERY SLOW)
valgrind --leak-check=full --track-origins=yes \
  python tools/train_twotower_v4.py \
    --pairs artifacts/twotower/pairs_tiny.npz \
    --bank artifacts/wikipedia_500k_corrected_vectors.npz \
    --epochs 1 --bs 4 --accum 1 --device cpu
```

**Note:** Valgrind may not work on macOS ARM (M1/M2/M3). Use Scalene instead.

---

## 📝 CODE CHANGES USER MADE (Potentially Irrelevant)

### File: tools/train_twotower_v4.py

#### Change 1: shuffle=False (Line 528)
```python
# BEFORE
train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True)

# AFTER
train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=False)
```

**User's reasoning:** Thought shuffle was causing index mismatches
**Impact:** Changed data access pattern (sequential instead of random)
**Likely relevance:** **LOW** (crashes happened before this change)

#### Change 2: Error Handling (Lines 242-316)
Added try-catch around training loop with detailed logging.

**Impact:** Should help debug, but **NEVER triggered** (meaning Python exceptions aren't happening)
**Likely relevance:** **NONE** (proves it's not a Python-level error)

#### Change 3: NaN/Inf Checks (Lines 286-292)
Added checks for invalid loss values.

**Impact:** **NEVER triggered** (no NaN/Inf detected before crash)
**Likely relevance:** **NONE**

---

## 🎯 MOST LIKELY ROOT CAUSES (Ranked)

### 1. Out of Memory (OOM) - **80% probability**
**Why:**
- Silent kills are classic OOM behavior on macOS/Linux
- Estimated peak usage: 16-27 GB
- Large bank vectors (2.37 GB) + 11 GB training data

**Test:** Run with `watch ps` (see Step 1 above)

### 2. Segfault in PyTorch/MKL - **15% probability**
**Why:**
- Happens on both MPS and CPU (different backends)
- GRU operations with large tensors could trigger edge case bugs
- Multi-threaded BLAS could cause race conditions

**Test:** Single-threaded execution (see Step 3 above)

### 3. FAISS Memory Corruption - **5% probability**
**Why:**
- FAISS creates temporary index (2.37 GB copy of bank vectors)
- Could trigger OOM or memory fragmentation

**Test:** Disable hard negative mining

---

## 📤 WHAT TO SEND TO CONSULTANT

1. **This entire report**
2. **Latest crash log:** `logs/twotower_v4_cpu_20251021_161447.log`
3. **Training config:** `runs/twotower_v4/config.yaml`
4. **System info:**
   ```bash
   uname -a
   sysctl hw.memsize  # macOS total RAM
   python --version
   pip show torch | grep Version
   pip show numpy | grep Version
   pip show faiss-cpu | grep Version
   ```
5. **Memory profile:** (run diagnostic command #1 above)

---

## ⚙️ QUICK FIXES TO TRY NOW

### Try 1: Reduce Batch Size (Fastest test)
```bash
# Edit launch_v4_cpu.sh
--bs 8 --accum 64  # Instead of --bs 32 --accum 16
```

### Try 2: Disable Memory Bank
```bash
# Edit launch_v4_cpu.sh
--memory-bank-size 0  # Disable FIFO queue (saves 154 MB)
```

### Try 3: Single-Threaded CPU
```bash
# Add to launch_v4_cpu.sh (top of file)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

### Try 4: Monitor Memory Live
```bash
./launch_v4_cpu.sh &
TRAIN_PID=$!
while kill -0 $TRAIN_PID 2>/dev/null; do
  ps -p $TRAIN_PID -o rss= | awk '{printf "Memory: %.2f GB\n", $1/1024/1024}'
  sleep 5
done
```

---

## 📚 REFERENCES

- PyTorch Memory Management: https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
- NumPy NPZ Format: https://numpy.org/doc/stable/reference/generated/numpy.savez.html
- macOS Memory Pressure: `man memory_pressure`
- Scalene Profiler: https://github.com/plasma-umass/scalene
