# Training Quick Start Guide
**Two-Tower Stable Training with Background Hang Fixes**

## 🚀 Quick Start (Choose One)

### Option 1: TMux Launcher (Recommended)

**Best for**: Interactive development, can reattach to see live output

```bash
# Install tmux (one-time only)
brew install tmux

# Launch training
./launch_tmux_stable.sh

# Inside tmux:
#   - Detach: Ctrl-B then D (training keeps running)
#   - Stop: Ctrl-C

# From another terminal:
tmux attach -t tt_sync           # Reattach to session
tail -f runs/stable_sync_*/training.log  # Monitor progress
```

**Pros**: No App Nap, can reattach, clean terminal management

---

### Option 2: Nohup Launcher (No tmux required)

**Best for**: Simple background jobs, automation scripts

```bash
# Launch training
./launch_nohup_stable.sh

# Monitor progress
tail -f logs/train_sync_*.out

# Check if running
ps -p $(cat /tmp/lnsp_training.pid)

# On-demand stack dump (if it hangs)
kill -USR1 $(cat /tmp/lnsp_training.pid)

# Stop training
kill $(cat /tmp/lnsp_training.pid)
```

**Pros**: No dependencies, simple, good for automation

---

## 🧪 Test First (Recommended)

Before starting a full training run, validate with the probe:

```bash
# Test single batch (should complete in <1s)
python -u tools/probe_simple.py --batch 4 --K 100

# Expected output:
# ✅ Single batch completed successfully!
# Total time: 0.030s
```

If the probe hangs or fails, **do not start full training** - debug first.

---

## 📊 Monitoring

### Check Progress
```bash
# View recent log lines
tail -20 runs/stable_sync_*/training.log

# Watch live (Ctrl-C to exit)
tail -f runs/stable_sync_*/training.log

# Search for errors
grep -i error runs/stable_sync_*/training.log
```

### Expected Output
```
[11:30:15] Epoch 1/5 | Step 50 | Loss 2.3456 | 2.5 it/s
[11:31:20] Epoch 1/5 | Step 100 | Loss 2.2134 | 2.6 it/s
[11:32:25] Epoch 1/5 | Step 150 | Loss 2.1892 | 2.5 it/s
```

### Check Status
```bash
# Is training running?
ps -p $(cat /tmp/lnsp_training.pid)

# CPU usage
top -pid $(cat /tmp/lnsp_training.pid)

# Memory usage
ps -o rss,vsz,comm -p $(cat /tmp/lnsp_training.pid)
```

---

## 🐛 Debugging

### If Training Hangs

**1. Check watchdog output**:
```bash
grep "WATCHDOG" logs/train_sync_*.out
```
If you see "No progress for 120s", the watchdog detected a hang and dumped stacks.

**2. On-demand stack dump**:
```bash
kill -USR1 $(cat /tmp/lnsp_training.pid)
tail -50 logs/train_sync_*.out  # View stack traces
```

**3. Check last log timestamp**:
```bash
tail -1 runs/stable_sync_*/training.log
```
If timestamp is >5 minutes old, training likely hung.

**4. Run probe to isolate issue**:
```bash
python -u tools/probe_simple.py --batch 8 --K 200
```

**5. Compare foreground vs background**:
```bash
# Try running in foreground
PYTHONPATH=. python -u tools/train_stable_sync.py
```

---

## 🛑 Stopping Training

### TMux Session
```bash
tmux attach -t tt_sync
# Press Ctrl-C to stop
```

### Nohup Background
```bash
kill $(cat /tmp/lnsp_training.pid)

# If not responding, force kill:
kill -9 $(cat /tmp/lnsp_training.pid)
```

---

## 📦 Output Files

### Training Run
```
runs/stable_sync_YYYYMMDD_HHMMSS/
├── training.log          # Main log file
├── epoch_001.pt          # Checkpoint after epoch 1
├── epoch_002.pt          # Checkpoint after epoch 2
└── ...
```

### Logs (nohup)
```
logs/
└── train_sync_YYYY-MM-DD_HHMM.out
```

### Monitoring Files
```
/tmp/
├── lnsp_training.pid          # Process ID
├── lnsp_training_outdir.txt   # Output directory
└── lnsp_training_log.txt      # Log file path
```

---

## ⚙️ Configuration

### Edit Training Parameters

Edit `tools/train_stable_sync.py`:

```python
CONFIG = {
    'device': 'cpu',           # or 'mps' for Apple Silicon
    'batch_size': 8,           # batch size
    'accum_steps': 2,          # gradient accumulation
    'epochs': 5,               # number of epochs
    'lr': 3e-4,                # learning rate
    'temperature': 0.07,       # InfoNCE temperature
    'K': 500,                  # number of candidates
    'nprobe': 8,               # FAISS nprobe
}
```

### Recommended Settings

**Initial Validation (1 hour)**:
- `epochs`: 1
- `batch_size`: 8
- `K`: 200

**Full Training (2-3 hours)**:
- `epochs`: 5
- `batch_size`: 8
- `K`: 500

**Fast Iteration (30 min)**:
- `epochs`: 1
- `batch_size`: 16
- `K`: 100

---

## 🔧 Environment Variables

All set automatically by launchers:

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTHONUNBUFFERED` | 1 | Show logs immediately |
| `PYTHONFAULTHANDLER` | 1 | Crash dumps |
| `OMP_NUM_THREADS` | 1 | Prevent thread contention |
| `KMP_DUPLICATE_LIB_OK` | TRUE | Allow duplicate OpenMP |

---

## 📈 Performance

### Expected Times (CPU, Single-Threaded)
- **Single batch**: ~0.03-0.05s
- **50 steps**: ~2-3 minutes
- **1 epoch**: ~10-15 minutes (10k samples)
- **5 epochs**: ~1-1.5 hours

### Optimization Later
- Enable multi-threading (2-3x faster)
- Use MPS device on Apple Silicon (5-10x faster)
- Increase batch size for better throughput

---

## ✅ Success Checklist

Before starting full training:

- [ ] Probe test passed (`python -u tools/probe_simple.py`)
- [ ] Integration tests passed (`pytest tests/test_twotower_integration.py`)
- [ ] Data files exist (NPZ + FAISS index)
- [ ] Launcher made executable (`chmod +x launch_*.sh`)
- [ ] Sufficient disk space (>5GB free)
- [ ] Tmux installed (if using tmux launcher) or skip this

After starting:

- [ ] Log file is updating (`tail -f ...`)
- [ ] Process is alive (`ps -p ...`)
- [ ] No watchdog warnings
- [ ] CPU usage is reasonable (30-80%)
- [ ] Loss is decreasing

---

## 🆘 Common Issues

### "tmux not installed"
```bash
brew install tmux
# or use nohup launcher instead
./launch_nohup_stable.sh
```

### "No such file: artifacts/..."
```bash
# Verify data files exist
ls -lh artifacts/wikipedia_500k_corrected_vectors.npz
ls -lh artifacts/wikipedia_500k_corrected_ivf_flat_ip.index
```

### "Process not found"
```bash
# Check PID file
cat /tmp/lnsp_training.pid

# Search for process
ps aux | grep train_stable_sync
```

### "No progress in logs"
```bash
# Check watchdog output
grep "WATCHDOG" logs/train_sync_*.out

# Try stack dump
kill -USR1 $(cat /tmp/lnsp_training.pid)
```

### "Permission denied"
```bash
chmod +x launch_tmux_stable.sh
chmod +x launch_nohup_stable.sh
chmod +x tools/probe_simple.py
```

---

## 📚 Additional Documentation

- **Full Fix Report**: `docs/reports/Training_Hang_Fix_Report_2025-10-22.md`
- **API Reference**: `docs/API_REFERENCE.md`
- **Integration Tests**: `tests/test_twotower_integration.py`
- **Task Summary**: `docs/reports/Task_Completion_Summary_2025-10-22.md`

---

**Last Updated**: 2025-10-22
**Version**: 1.0
**Status**: Ready for use
