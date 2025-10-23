# Training Script Patches for Stability
**Target:** `tools/train_twotower_v4.py`
**Purpose:** Apply sync miner + stability fixes to existing training code
**Date:** October 22, 2025

---

## Option 1: Use New Stable Trainer (Recommended)

**Skip patches entirely** - use the new standalone trainer:

```bash
./scripts/run_sync_cpu.sh
```

This uses `src/training/train_twotower_sync.py` with all fixes pre-applied.

---

## Option 2: Patch Existing Trainer

If you need to modify `tools/train_twotower_v4.py` instead:

### Patch 1: Import Sync Miner

**Location:** Top of file, imports section

**Replace:**
```python
from tools.async_miner import AsyncFaissMiner
```

**With:**
```python
# Stable alternative: synchronous miner (no multiprocessing)
import sys
sys.path.insert(0, 'src')
from retrieval.miner_sync import SyncFaissMiner
from utils.memprof import log_mem
```

---

### Patch 2: DataLoader Configuration

**Location:** DataLoader initialization (~line 200)

**Replace:**
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,        # ❌ Causes issues on macOS
    pin_memory=True,      # ❌ Incompatible with MPS
)
```

**With:**
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,        # ✅ No worker processes (stable)
    pin_memory=False,     # ✅ MPS-friendly
)
```

---

### Patch 3: Device Selection

**Location:** Device initialization (~line 50)

**Replace:**
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

**With:**
```python
# Force CPU until sync mode proven stable (re-enable MPS after validation)
device = torch.device("cpu")
print(f"Using device: {device} (CPU-only mode for stability)")
```

**Note:** Re-enable MPS after 1 clean epoch on CPU.

---

### Patch 4: Miner Initialization

**Location:** Miner creation (~line 150)

**Replace:**
```python
miner = AsyncFaissMiner(
    faiss_index=index,
    k=128,
    qbatch=2048,
    ttl=5,
    use_multiprocessing=True,  # ❌ Causes crashes
)
```

**With:**
```python
# Synchronous miner (no multiprocessing) for stability
miner = SyncFaissMiner(
    index=index,
    nprobe=8,  # Adjustable: 4-32 (higher = better recall, slower)
)
print("✓ Using synchronous FAISS miner (stable mode)")
```

---

### Patch 5: Training Loop (Miner Usage)

**Location:** Training loop, mining call (~line 300)

**Replace:**
```python
# Async receive
try:
    qid, indices, distances = miner.async_receive(timeout=1.0)
except queue.Empty:
    # Fallback to empty
    indices = np.zeros((batch_size, k), dtype=np.int64)
    distances = np.zeros((batch_size, k), dtype=np.float32)
```

**With:**
```python
# Synchronous search (main thread)
q_np = query_vectors.detach().cpu().numpy().astype(np.float32)
indices, distances = miner.search(q_np, k=500)
# indices: (B, K) int64, distances: (B, K) float32
```

---

### Patch 6: Memory Profiling

**Location:** Training loop, inside step iteration (~line 350)

**Add:**
```python
# Memory monitoring (every 500 steps)
if (step + 1) % 500 == 0:
    log_mem(f"epoch_{epoch}_step_{step+1}")
    # Optional: Alert if RSS grows >500 MB from baseline
    import psutil, os
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    if rss_mb > baseline_rss_mb + 500:
        print(f"⚠️  Memory leak detected: {rss_mb:.1f} MB (baseline: {baseline_rss_mb:.1f} MB)")
```

**Note:** Set `baseline_rss_mb` before training loop starts.

---

### Patch 7: Remove Async Cleanup

**Location:** End of training script (~line 800)

**Remove:**
```python
# Cleanup async miner
miner.stop()
miner.join()
```

**Reason:** Sync miner has no background workers to stop.

---

## Patch Application Script

**Automated patching** (use with caution - verify diffs first):

```bash
# Backup original
cp tools/train_twotower_v4.py tools/train_twotower_v4.py.backup

# Apply patches
patch tools/train_twotower_v4.py << 'EOF'
--- a/tools/train_twotower_v4.py
+++ b/tools/train_twotower_v4.py
@@ -10,7 +10,9 @@
 import torch.nn.functional as F
 from torch.utils.data import DataLoader
-from tools.async_miner import AsyncFaissMiner
+import sys
+sys.path.insert(0, 'src')
+from retrieval.miner_sync import SyncFaissMiner
+from utils.memprof import log_mem

@@ -50,1 +52,2 @@
-device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
+# Force CPU for stability (re-enable MPS after validation)
+device = torch.device("cpu")
+print(f"Using device: {device} (CPU-only mode for stability)")

@@ -150,6 +153,4 @@
-miner = AsyncFaissMiner(
-    faiss_index=index, k=128, qbatch=2048, ttl=5, use_multiprocessing=True
-)
+miner = SyncFaissMiner(index=index, nprobe=8)
+print("✓ Using synchronous FAISS miner (stable mode)")

@@ -200,2 +201,2 @@
-    num_workers=4, pin_memory=True,
+    num_workers=0, pin_memory=False,  # MPS-friendly, stable

@@ -300,8 +301,3 @@
-try:
-    qid, indices, distances = miner.async_receive(timeout=1.0)
-except queue.Empty:
-    indices = np.zeros((batch_size, k), dtype=np.int64)
-    distances = np.zeros((batch_size, k), dtype=np.float32)
+q_np = query_vectors.detach().cpu().numpy().astype(np.float32)
+indices, distances = miner.search(q_np, k=500)

@@ -350,0 +346,7 @@
+# Memory monitoring
+if (step + 1) % 500 == 0:
+    log_mem(f"epoch_{epoch}_step_{step+1}")
+    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
+    if rss_mb > baseline_rss_mb + 500:
+        print(f"⚠️  Memory leak: {rss_mb:.1f} MB")
+
@@ -800,3 +807,1 @@
-# Cleanup async miner
-miner.stop()
-miner.join()

EOF

# Verify patches applied correctly
diff tools/train_twotower_v4.py.backup tools/train_twotower_v4.py
```

**Note:** Line numbers are approximate. Adjust based on your actual file structure.

---

## Validation After Patching

### 1. Import Check
```bash
PYTHONPATH=. python3 -c "
import sys
sys.path.insert(0, 'src')
from retrieval.miner_sync import SyncFaissMiner
from utils.memprof import log_mem
print('✓ Imports successful')
"
```

### 2. Syntax Check
```bash
python3 -m py_compile tools/train_twotower_v4.py
```

### 3. Dry Run (1 step)
```bash
PYTHONPATH=. python3 tools/train_twotower_v4.py \
  --batch-size 4 \
  --epochs 1 \
  --max-steps 1 \
  --device cpu \
  --output runs/patch_test
```

Expected output:
```
Using device: cpu (CPU-only mode for stability)
✓ Using synchronous FAISS miner (stable mode)
Epoch 1/1
  Training:   1/1 [00:00<00:00, ?it/s]
[epoch_1_step_1] rss_mb=1234.5
✓ Training step completed
```

---

## Rollback

If patches cause issues:

```bash
# Restore original
mv tools/train_twotower_v4.py.backup tools/train_twotower_v4.py

# Use standalone trainer instead
./scripts/run_sync_cpu.sh
```

---

## Summary

**Recommended Path:**
1. Use `./scripts/run_sync_cpu.sh` (no patches needed)
2. OR patch existing trainer with above changes
3. Validate: 1 full epoch without crashes
4. Graduate to threaded miner (optional)

**Key Changes:**
- ❌ AsyncFaissMiner → ✅ SyncFaissMiner
- ❌ num_workers=4 → ✅ num_workers=0
- ❌ MPS device → ✅ CPU device (initially)
- ✅ Memory profiling added

**Expected Result:**
- 100% training completion rate (vs. 0% with async)
- 2-5 ms miner latency (acceptable)
- <500 MB memory growth
- Clean checkpoints every epoch

---

**Created:** October 22, 2025
**Maintainer:** Autonomous Training System
**See Also:** `docs/ops/triage_playbook.md`
