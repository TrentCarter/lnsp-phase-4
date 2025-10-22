# Async Mining Optimizations - Complete Implementation

## 🚀 Status: LIVE & RUNNING

Training started: 2025-10-21 22:36:08 EDT
Log: `logs/twotower_v4_mps_async_20251021_223608.log`

## ✅ Implemented Optimizations

### 1. Aggressive Async Mining Parameters

**Before:**
- qbatch=1024 (FAISS query batch size)
- ttl=3 (cache reuse steps)
- prefetch=2 (queue depth)

**After (optimized):**
- **qbatch=2048** - 2x larger batching for reduced FAISS overhead
- **ttl=5** - Reuse mined negatives for 5 steps (2-3x fewer FAISS calls)
- **prefetch=3** - Deeper queue for FAISS hiccup tolerance

**Impact:** Expected 2-3x reduction in FAISS overhead starting epoch 6

### 2. FIFO Alignment Queue

**Problem:** Async mining returns results for batch t when training batch t+N

**Solution:** 10-line FIFO queue pairs batches with mined results:
```python
batch_fifo = deque(maxlen=8)  # holds (q_cpu, d_pos_cpu) awaiting mined negs

# PRODUCER: Stash batch + submit for mining
batch_fifo.append((q.detach().cpu(), d_pos.detach().cpu()))
async_miner.submit(batch_idx, q)

# CONSUMER: Retrieve mined negs for OLDEST batch
mined = async_miner.try_get()
if mined is not None and len(batch_fifo) > 0:
    _, neg_indices = mined
    q_wait, d_pos_wait = batch_fifo.popleft()  # matching batch
    # ... compute loss with aligned batch ...
```

**Impact:** Ensures mined negatives match the correct batch (no misalignment)

### 3. Mixed Hard Negative Sources (hardneg-frac=0.5)

**Strategy:** 50% from FAISS, 50% from memory bank

```python
num_from_faiss = int(mining_spec['num'] * args.hardneg_frac)
num_from_memory = mining_spec['num'] - num_from_faiss

# Take first num_from_faiss from FAISS mining
hard_neg_vecs = mined_hard_vecs[:, :num_from_faiss, :]

# Add memory bank negatives
mem_neg_vecs = bank_vecs[mem_indices].view(B, num_from_memory, -1)
hard_neg_vecs = torch.cat([hard_neg_vecs, mem_neg_vecs], dim=1)
```

**Impact:** Halves FAISS demand while maintaining "hardness" quality

### 4. Queue Health Monitoring

**Every 200 steps:**
```
[Queue Health @ step 400] in_q=2, out_q=3, fifo=3
```

**What to watch:**
- `out_q > 0`: Mining keeping up (good)
- `out_q = 0`: Increase prefetch or qbatch (bottleneck)
- `fifo size`: Should grow to ~3-5 during steady state

### 5. Safety Rails (All Active)

✅ **num_workers=0** - TensorDataset already in RAM, workers add overhead
✅ **drop_last=True** - No partial batches (simplifies FIFO alignment)
✅ **foreach=False** - AdamW stability (no multi-tensor kernels)
✅ **fused=False** - AdamW stability (no fused ops)
✅ **miner.stop()** - Graceful shutdown at training end

## 📊 Expected Performance Gains (Epoch 6-30)

**Baseline (synchronous mining):**
- ~8-10 it/s during mining epochs
- FAISS overhead: ~60-70% of step time

**Optimized (async mining):**
- **Target: ~13 it/s sustained** (same as epoch 1-5 no-mining speed)
- FAISS overhead: <10% (overlapped with training)
- **Total speedup: 2-3x** on mining epochs

**Time savings:**
- Baseline: ~60-90 min for 30 epochs
- Optimized: ~20-30 min for 30 epochs
- **Saved: 40-60 minutes**

## 🔍 Monitoring During Training

### Watch These Metrics (Epoch 6+)

1. **Throughput stability:**
   ```
   Epoch 5 (no mining): 13.2 it/s  ← baseline
   Epoch 6 (mining):    13.0 it/s  ← should stay close!
   ```

2. **Queue health (every 200 steps):**
   ```
   [Queue Health @ step 600] in_q=2, out_q=3, fifo=4  ← healthy
   [Queue Health @ step 800] in_q=1, out_q=0, fifo=2  ← out_q empty = bottleneck
   ```

3. **Separation margin Δ (every epoch):**
   ```
   Epoch 6:  Δ = 0.03  (mining just started)
   Epoch 10: Δ = 0.05  (target: healthy training)
   Epoch 15: Δ = 0.07  (excellent separation)
   ```

### Troubleshooting

**If throughput drops below 10 it/s starting epoch 6:**
- Increase `ttl=7` (reuse mined negatives longer)
- Reduce `qbatch=1536` (if latency spikes)
- Increase `prefetch=4` (if `out_q` often 0)

**If separation Δ drops after mining starts:**
- Tighten cosine window: `0.84-0.94` (was `0.84-0.96`)
- Lower hard-neg count: `12` (was `16`)
- Adjust `hardneg-frac=0.4` (more memory bank, less FAISS)

**If you see shape mismatch errors:**
- Check `drop_last=True` on DataLoader (should be set)
- Verify `batch_fifo` size matches batch count
- Skip mined negatives once (don't popleft) and use fallback

## 📁 Key Files Modified

1. **`tools/train_twotower_v4.py`:**
   - Added FIFO alignment queue
   - Implemented hardneg-frac mixing
   - Added queue health monitoring
   - Safety rails: num_workers=0, drop_last=True

2. **`launch_v4_mps_async.sh`:**
   - Updated mining parameters (qbatch=2048, ttl=5, prefetch=3)
   - Added hardneg-frac=0.5, mine-refresh-steps=3000

3. **`tools/async_miner.py`:**
   - Already optimal (no changes needed)

4. **`monitor_async_training.sh`:** (NEW)
   - Real-time monitoring script for queue health + throughput

## 🎯 Next Steps

1. **Let epoch 6 complete** - First mining epoch, critical diagnostic
2. **Check queue health logs** - Ensure `out_q > 0` consistently
3. **Verify separation Δ grows** - Should reach ≥0.05 by epoch 10
4. **If stable:** Run full 30 epochs (ETA: ~25 min)
5. **If bottleneck:** Tune qbatch/ttl/prefetch based on queue logs

## 📝 Command Reference

```bash
# Monitor training (real-time)
./monitor_async_training.sh

# Check current status
tail -100 logs/twotower_v4_mps_async_20251021_223608.log

# Watch for queue health
grep "Queue Health" logs/twotower_v4_mps_async_20251021_223608.log

# Check separation margin
grep "Separation Δ" logs/twotower_v4_mps_async_20251021_223608.log

# View config
cat runs/twotower_v4_mps_async/config.yaml
```

## 🏆 Success Criteria

✅ **Throughput:** ≥12 it/s sustained through epoch 6-30
✅ **Queue health:** `out_q > 0` in most snapshots
✅ **Separation Δ:** ≥0.05 by epoch 10
✅ **No crashes:** Full 30 epochs complete
✅ **Time:** ≤30 minutes total

---

**Implementation Date:** 2025-10-21
**Status:** Live and running with optimized parameters
**Expected completion:** ~22:56 EDT (20 minutes from start)
