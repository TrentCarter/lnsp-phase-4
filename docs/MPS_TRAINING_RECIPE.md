# MPS Training Recipe for macOS

**Safe MPS configuration for 3-5x faster training on Apple Silicon**

## Environment Variables

```bash
# OpenMP thread control (prevent duplicate library crashes)
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1

# MPS memory management (prevent memory fragmentation)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Unbuffered output for log visibility
export PYTHONUNBUFFERED=1
```

## Device Selection

```python
import torch

# Safe device detection
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
```

## DataLoader Configuration

```python
from torch.utils.data import DataLoader

# Safe DataLoader settings for MPS
train_loader = DataLoader(
    train_dataset,
    batch_size=256,          # Same as CPU
    shuffle=True,
    num_workers=0,           # CRITICAL: MPS requires 0 workers
    pin_memory=False,        # CRITICAL: Must be False for MPS
    persistent_workers=False, # CRITICAL: Must be False (requires num_workers=0)
    drop_last=True           # Prevent incomplete batches
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,           # CRITICAL: MPS requires 0 workers
    pin_memory=False,        # CRITICAL: Must be False for MPS
    persistent_workers=False  # CRITICAL: Must be False
)
```

## Complete Training Script Template

```bash
#!/bin/bash
# Safe MPS training launch script

# Environment setup
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTHONUNBUFFERED=1

# Run training
python -u app/lvm/train_twotower_fast.py \
    --resume artifacts/lvm/models/twotower_samearticle/epoch3.pt \
    --train-npz artifacts/lvm/train_clean_disjoint.npz \
    --same-article-k 3 \
    --nearmiss-jsonl artifacts/mined/nearmiss_train_ep3.jsonl \
    --p-cache-npy artifacts/eval/p_train_ep3.npy \
    --epochs 1 \
    --batch-size 256 \
    --lr 1e-4 \
    --device mps \
    --save-dir artifacts/lvm/models/twotower_fast_mps \
    > /tmp/epoch4_mps.log 2>&1 &

# Save PID for monitoring
echo $! > /tmp/training_mps.pid
echo "Training started on MPS (PID: $!)"
echo "Monitor: tail -f /tmp/epoch4_mps.log"
```

## Expected Performance

- **CPU**: 30-60 minutes per epoch (316% CPU, 4 cores)
- **MPS**: 10-20 minutes per epoch (3-5x speedup on Apple Silicon)
- **GPU usage**: Monitor with `sudo powermetrics --samplers gpu_power`

## Monitoring

```bash
# Watch training progress
tail -f /tmp/epoch4_mps.log

# Check heartbeat
watch -n 5 'cat artifacts/lvm/train_heartbeat.json | jq'

# Monitor GPU power
sudo powermetrics --samplers gpu_power -i 5000
```

## Known Issues and Solutions

### Issue 1: "Abort trap: 6" Crash
**Cause**: Duplicate OpenMP libraries (PyTorch + FAISS)
**Solution**: Always set `KMP_DUPLICATE_LIB_OK=TRUE`

### Issue 2: DataLoader Multiprocessing Hang
**Cause**: MPS doesn't support multiprocessing workers
**Solution**: Set `num_workers=0`

### Issue 3: Memory Fragmentation
**Cause**: MPS memory allocator issues
**Solution**: Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`

### Issue 4: Empty Log Files
**Cause**: Python stdout buffering
**Solution**: Use `python -u` and `PYTHONUNBUFFERED=1`

## When to Use MPS vs CPU

**Use CPU when:**
- Maximum stability required (production runs)
- Long training runs (>1 hour) where crashes are costly
- Debugging new training logic

**Use MPS when:**
- Fast iteration required
- Proven stable training script
- Time is critical (deadlines, experiments)
- Acceptable to restart if crash occurs

## Fallback to CPU

```python
# Automatic fallback if MPS crashes
try:
    device = torch.device("mps")
    # ... training code ...
except RuntimeError as e:
    if "MPS" in str(e):
        print("MPS failed, falling back to CPU")
        device = torch.device("cpu")
        # Reload models and restart
```

## References

- PyTorch MPS backend docs: https://pytorch.org/docs/stable/notes/mps.html
- OpenMP duplicate library issue: See `CRASH_ROOT_CAUSE.md`
- Training performance benchmarks: See `docs/LVM_DATA_MAP.md`
