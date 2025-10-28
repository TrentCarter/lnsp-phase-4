# LVM Training Progression: 232k → 340k → 367k

**Date:** October 18, 2025
**Status:** 367k training in progress (EST: 3-4 hours)

---

## 📊 Data Growth Timeline

| Date | Dataset Size | Wikipedia Articles | Training Sequences | Source |
|------|-------------|-------------------|-------------------|--------|
| Oct 17 | 232,525 concepts | 3,425 articles | 232,520 sequences | Baseline |
| Oct 18 (AM) | 339,657 concepts | ~3,432 articles | 339,652 sequences | +107k ingestion |
| Oct 18 (PM) | 367,380 concepts | ~3,931 articles | 367,373 sequences | +499 articles |

**Current NPZ File:** `artifacts/wikipedia_500k_corrected_vectors.npz` (1046.6 MB)
**Current Training Data:** `artifacts/lvm/data/training_sequences_ctx5.npz` (2043.7 MB)

---

## 🚨 CRITICAL FINDING: Invalid Comparison Methodology

### The Problem

Initial comparison showed ALL 340k models degraded compared to 232k baseline:
- Transformer: -2.8%
- LSTM: -4.0%
- GRU: -3.6%
- AMN: -4.6%

### Root Cause Identified

**User feedback:** *"are you testing on trained data or untrained (held back) data?"*

**Issue:** `train_unified.py` line 302 uses `torch.utils.data.random_split()` **WITHOUT A SEED**

```python
# WRONG: Different validation sets for each training run!
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
```

**Impact:**
- 232k models: Random 10% of 232k data as validation
- 340k models: Different random 10% of 340k data as validation
- **NOT THE SAME TEST DATA** = Invalid comparison!

### The Fix

Created `fair_comparison_232k_vs_340k.py`:
- Uses SAME fixed 10k test set for both model generations
- Test set = first 10k vectors from 232k baseline
- Ensures apples-to-apples comparison

---

## ✅ Corrected Results: 340k vs 232k

### Fair Comparison (Same 10k Test Set)

| Model | 232k Baseline | 340k Model | Difference | Change |
|-------|--------------|------------|------------|--------|
| **Transformer** | 0.5576 | **0.5743** | +0.0167 | **+2.99%** ⬆️ |
| **GRU** | 0.5502 | **0.5644** | +0.0142 | **+2.58%** ⬆️ |
| **AMN** | 0.5357 | **0.5400** | +0.0043 | **+0.80%** ⬆️ |
| **LSTM** | 0.4247 | 0.3265 | -0.0982 | **-23.13%** ⬇️ |

### Key Findings

✅ **3 out of 4 models IMPROVED with 340k data**
- Transformer: +3% improvement (strongest)
- GRU: +2.6% improvement
- AMN: +0.8% improvement (modest)

❌ **LSTM degraded significantly (-23%)**
- Requires investigation
- Possible causes:
  - Model loading issue in comparison script
  - Learning rate too high for larger dataset
  - Architecture-specific sensitivity to data distribution

---

## 🔍 LSTM Degradation Investigation

### Evidence of Normal Training

340k LSTM training log shows **NORMAL** performance:
```
Best val loss: 0.001161
Final Val Cosine: 0.5526  # This is GOOD!
```

### Contradiction

Fair comparison shows: 0.3265 cosine (BAD!)

### Hypothesis

**Model Loading Issue:** The fair comparison script may not have loaded the 232k LSTM correctly, OR there's a problem with the checkpoint format.

### TODO: Investigate

1. Re-run fair comparison with verbose logging
2. Verify model checkpoint integrity
3. Compare model architectures (232k vs 340k LSTM)
4. Check if learning rate needs adjustment for larger datasets

---

## 🏗️ Current 367k Training

### Configuration

```bash
Data: artifacts/lvm/data/training_sequences_ctx5.npz (367,373 sequences)
Output: artifacts/lvm/models_367k/
Epochs: 20
Batch size: 64
Device: MPS (Apple M1 Max)
```

### Training Order

1. **AMN** (in progress) - ~45-60 min
2. **LSTM** (pending) - ~45-60 min
3. **GRU** (pending) - ~45-60 min
4. **Transformer** (pending) - ~45-60 min

**Total EST:** 3-4 hours

### Logs

- Main log: `artifacts/lvm/train_367k_full.log`
- Individual logs: `artifacts/lvm/models_367k/{model}_training.log`

---

## 📈 Expected Outcomes

### If Trend Continues (3/4 models improving)

**Transformer:** 0.5743 → **~0.5900+** (target: +2-3%)
**GRU:** 0.5644 → **~0.5790+** (target: +2-3%)
**AMN:** 0.5400 → **~0.5440+** (target: +0.5-1%)
**LSTM:** 0.3265 → **???** (needs investigation)

### Success Criteria

- **Minimum:** 2/4 models improve over 340k baseline
- **Target:** 3/4 models improve (matching 340k trend)
- **Stretch:** All 4 models improve (requires LSTM fix)

---

## 🎯 Next Steps

### Immediate (Post-Training)

1. ✅ Run comprehensive fair comparison:
   ```bash
   ./.venv/bin/python tools/fair_comparison_all_datasets.py
   ```

2. 📊 Document results in comparison table

3. 🔬 Investigate LSTM degradation:
   - Re-run comparison with verbose logging
   - Verify model checkpoints
   - Test with different learning rates

### Follow-Up

4. 📈 Create visualization of training progression
5. 📝 Update LVM_DATA_MAP.md with 367k results
6. 🚀 Plan next Wikipedia ingestion (target: 500k concepts)

---

## 📁 File Locations

### Models

**232k Baseline (Oct 17):**
- `artifacts/lvm/models/transformer_232k_20251017_090129/`
- `artifacts/lvm/models/lstm_232k_20251017_090129/`
- `artifacts/lvm/models/gru_232k_20251017_090129/`
- `artifacts/lvm/models/amn_232k_20251017_090129/`

**340k Models (Oct 18 AM):**
- `artifacts/lvm/models_340k/transformer/`
- `artifacts/lvm/models_340k/lstm/`
- `artifacts/lvm/models_340k/gru/`
- `artifacts/lvm/models_340k/amn/`

**367k Models (Oct 18 PM) - In Progress:**
- `artifacts/lvm/models_367k/transformer/`
- `artifacts/lvm/models_367k/lstm/`
- `artifacts/lvm/models_367k/gru/`
- `artifacts/lvm/models_367k/amn/`

### Data

- NPZ vectors: `artifacts/wikipedia_500k_corrected_vectors.npz` (1046.6 MB)
- Training data: `artifacts/lvm/data/training_sequences_ctx5.npz` (2043.7 MB)
- FAISS index: `artifacts/wikipedia_500k_corrected_ivf_flat_ip.index` (1080.6 MB)

### Scripts

- Training: `tools/train_all_lvms_367k.sh`
- Fair comparison: `tools/fair_comparison_all_datasets.py`
- 232k→340k comparison: `tools/fair_comparison_232k_vs_340k.py`

---

## 💡 Lessons Learned

### 1. Always Use Fixed Test Sets

**Problem:** Random validation splits create invalid comparisons
**Solution:** Use same fixed test set across all experiments

### 2. More Data Generally Helps (But Not Always)

- 3/4 models improved with more data (+46% increase)
- Architecture matters: Some models scale better than others
- Quality > Quantity: Need to maintain data quality during growth

### 3. User Feedback is Critical

The invalid comparison methodology was caught by user asking:
> *"are you testing on trained data or untrained (held back) data?"*

**Takeaway:** Always explain evaluation methodology clearly and verify assumptions!

### 4. Monitor Individual Model Behavior

LSTM degradation (-23%) flagged potential issues:
- Model loading problems
- Hyperparameter sensitivity
- Architecture-specific scaling challenges

---

## 🏆 Success Metrics

### Data Pipeline

✅ 232k → 340k (+46% data)
✅ 340k → 367k (+8% data)
✅ Clean ingestion (499/500 articles successful)
✅ Proper vector normalization (L2=1.0)

### Model Training

✅ 4 architectures trained on 232k (baseline)
✅ 4 architectures trained on 340k (done)
🔄 4 architectures training on 367k (in progress)

### Quality Assurance

✅ Fair comparison methodology established
✅ Invalid comparison detected and fixed
⚠️ LSTM issue identified (needs resolution)

---

**Status:** Waiting for 367k training to complete (~3-4 hours from 3:21 PM)
**ETA:** ~6:30-7:30 PM
**Next Action:** Run `fair_comparison_all_datasets.py` when training completes
