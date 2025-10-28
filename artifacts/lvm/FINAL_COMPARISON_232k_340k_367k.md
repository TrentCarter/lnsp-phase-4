# Final Fair Comparison: 232k vs 340k vs 367k

**Date:** October 18, 2025
**Test Set:** Same fixed 10,000 samples for all comparisons
**Status:** ✅ COMPLETE

---

## 📊 Complete Results

### Cosine Similarity (Higher is Better)

| Model | 232k Baseline | 340k Model | 367k Model | Total Improvement |
|-------|---------------|------------|------------|-------------------|
| **Transformer** | 0.5402 | 0.5517 | **0.5614** | **+3.92%** ⬆️ |
| **GRU** | 0.5314 | 0.5384 | **0.5625** | **+5.84%** ⬆️ |
| **AMN** | 0.5228 | 0.5254 | **0.5275** | **+0.89%** ⬆️ |
| **LSTM** | 0.3921 | 0.3140 | **0.1102** | **-71.90%** ⬇️ |

---

## ✅ SUCCESS: 3/4 Models Improving Consistently

### Transformer (Best Overall)
- **232k → 340k:** +2.13% improvement
- **340k → 367k:** +1.75% improvement
- **Total gain:** +3.92% (0.5402 → 0.5614)
- **Trend:** Consistent improvement with more data ✅

### GRU (Strongest Improvement)
- **232k → 340k:** +1.31% improvement
- **340k → 367k:** +4.47% improvement
- **Total gain:** +5.84% (0.5314 → 0.5625)
- **Trend:** Accelerating improvement! ⭐

### AMN (Modest but Stable)
- **232k → 340k:** +0.50% improvement
- **340k → 367k:** +0.40% improvement
- **Total gain:** +0.89% (0.5228 → 0.5275)
- **Trend:** Small but consistent gains ✅

---

## 🚨 CRITICAL ISSUE: LSTM Catastrophic Failure

### LSTM Performance Collapse

| Dataset | Cosine Score | Change from Baseline |
|---------|--------------|---------------------|
| 232k | 0.3921 | — |
| 340k | 0.3140 | **-19.93%** ⬇️ |
| 367k | 0.1102 | **-71.90%** ⬇️ |

### Symptoms
- Progressive degradation across all dataset sizes
- Final score of 0.1102 is essentially **random predictions**
- Other models improving while LSTM failing suggests architecture-specific issue

### Possible Causes

1. **Random Seed Issue** (Most Likely)
   - `torch.utils.data.random_split()` without seed creates different validation sets
   - LSTM may be particularly sensitive to which data ends up in validation
   - Different validation sets = different early stopping points = poor generalization

2. **Learning Rate Too High**
   - Larger datasets may need lower learning rate for LSTM
   - Current: 0.0005 (same for all models)
   - LSTM may need: 0.0001-0.0002 for 340k+ data

3. **Gradient Instability**
   - LSTM sensitive to gradient exploding/vanishing
   - Larger datasets = more batches = more gradient steps
   - May need gradient clipping or different optimizer

4. **Model Loading Bug**
   - Comparison script may not be loading LSTM checkpoints correctly
   - Need to verify checkpoint integrity

---

## 🎯 Production Recommendation

### Use GRU for Production ⭐

**Reasons:**
1. **Best performance:** 0.5625 cosine (highest of all models)
2. **Most improvement:** +5.84% total gain (strongest scaling)
3. **Accelerating gains:** +4.47% from 340k→367k (getting better!)
4. **Stable training:** No issues across all dataset sizes

### Ranking (367k models)

1. **GRU:** 0.5625 (⭐ RECOMMENDED)
2. **Transformer:** 0.5614 (close second)
3. **AMN:** 0.5275 (acceptable)
4. **LSTM:** 0.1102 (❌ DO NOT USE)

---

## 📈 Data Scaling Analysis

### Does More Data Help?

**YES!** For 3 out of 4 architectures:

| Model | 232k→340k | 340k→367k | Trend |
|-------|-----------|-----------|-------|
| GRU | +1.31% | +4.47% | Accelerating ⬆️ |
| Transformer | +2.13% | +1.75% | Consistent ✅ |
| AMN | +0.50% | +0.40% | Consistent ✅ |

**Conclusion:** Continue scaling data! GRU benefits most from larger datasets.

---

## 🔬 Recommended Next Steps

### 1. Investigate LSTM (Priority: HIGH)

```bash
# Re-run with fixed random seed
python tools/retrain_lstm_with_seed.py \
  --data artifacts/lvm/data/training_sequences_ctx5.npz \
  --epochs 20 \
  --lr 0.0002 \
  --seed 42

# Verify checkpoint loading
python tools/verify_lstm_checkpoints.py
```

### 2. Optimize GRU Hyperparameters

Since GRU is the best performer, optimize it further:
- Test learning rates: 0.0003, 0.0005, 0.0007
- Test batch sizes: 32, 64, 128
- Test epochs: 20, 30, 40

### 3. Continue Data Scaling

GRU shows accelerating improvements with more data:
- Target: 500k concepts (current: 367k)
- Next ingestion: ~1,000 Wikipedia articles
- Expected GRU gain: +2-3% based on trend

### 4. Update Documentation

```bash
# Update leaderboard
vim artifacts/lvm/COMPREHENSIVE_LEADERBOARD.md

# Update data map
vim docs/LVM_DATA_MAP.md
```

---

## 📊 Training Efficiency

### Time Breakdown (367k training)

| Model | Training Time | Samples/sec | Efficiency |
|-------|--------------|-------------|------------|
| AMN | ~20 min | ~306 | Fastest |
| LSTM | ~20 min | ~306 | Fast |
| GRU | ~20 min | ~306 | Fast |
| Transformer | ~22 min | ~278 | Moderate |

**Total:** 1h 22m for all 4 models (MPS)

---

## 💾 Model Locations

### Production Models (367k - Best Performance)

```
artifacts/lvm/models_367k/
├── gru/               ⭐ RECOMMENDED (0.5625)
│   ├── best_model.pt
│   └── final_model.pt
├── transformer/       (0.5614)
│   ├── best_model.pt
│   └── final_model.pt
├── amn/               (0.5275)
│   ├── best_model.pt
│   └── final_model.pt
└── lstm/              ❌ DO NOT USE (0.1102)
    ├── best_model.pt
    └── final_model.pt
```

### Historical Models

- **232k:** `artifacts/lvm/models/*_232k_20251017_090129/`
- **340k:** `artifacts/lvm/models_340k/*/`

---

## 🎓 Lessons Learned

### 1. More Data Generally Helps
- 3/4 models improved with larger datasets
- GRU benefits most from scaling
- Diminishing returns expected beyond 500k

### 2. Architecture Matters
- GRU: Best scaling characteristics
- Transformer: Consistent but slower gains
- AMN: Modest improvements
- LSTM: Requires special handling

### 3. Fixed Test Sets Are Critical
- Random validation splits create invalid comparisons
- Always use same test set across experiments
- Document evaluation methodology clearly

### 4. Monitor Individual Model Behavior
- LSTM's progressive failure was caught early
- Architecture-specific issues need investigation
- One size doesn't fit all!

---

## ✅ Final Checklist

- [x] Export 367k training data
- [x] Train all 4 models on 367k dataset
- [x] Run fair comparison across 232k/340k/367k
- [x] Document results and findings
- [ ] Investigate LSTM degradation
- [ ] Optimize GRU hyperparameters
- [ ] Plan next Wikipedia ingestion (500k target)
- [ ] Update production deployment to use GRU

---

**Next Action:** Use GRU model (0.5625 cosine) for production inference!

**Model Path:** `artifacts/lvm/models_367k/gru/best_model.pt`
