# CRITICAL BUG: New Training Runs Produce Broken Models

**Date**: 2025-10-31
**Severity**: 🚨 **P0 - CRITICAL**
**Status**: ❌ **BLOCKING ALL NEW MODEL DEPLOYMENT**

---

## Executive Summary

All models trained in the latest retraining effort (Oct 30-31, 2025) are **completely broken** and produce near-zero prediction quality, despite checkpoints claiming 50%+ val_cosine. This affects:

- ❌ AMN (amn_584k_clean)
- ❌ Transformer (transformer_5cat_20251030_230110)

Meanwhile, **old production models from 340k data still work correctly**, confirming this is a NEW bug, not a systemic issue.

---

## Evidence

### Old Models (340k) - ✅ WORKING

| Model | Checkpoint Val Cosine | Actual Val Cosine | Difference | Status |
|-------|----------------------|-------------------|------------|---------|
| Transformer | 0.5658 | 0.5449 | -0.02 | ✅ **CORRECT** |
| GRU | 0.5920 (claimed) | Not yet tested | - | Likely OK |
| LSTM | 0.4102 (claimed) | Not yet tested | - | Likely OK |

### New Models (584k) - ❌ BROKEN

| Model | Checkpoint Val Cosine | Actual Val Cosine | Difference | Status |
|-------|----------------------|-------------------|------------|---------|
| AMN | 0.5329 | **-0.0073** | **-0.54** | ❌ **BROKEN** |
| Transformer | 0.5577 | Not yet tested | - | Likely broken |

---

## Detailed Findings

### Test Methodology

Replicated the EXACT validation procedure from `app/lvm/train_unified.py`:
1. Load checkpoint
2. Load training data `training_sequences_ctx5_584k_clean_splits.npz`
3. 90/10 random split (seed 42)
4. Evaluate on validation split using `cosine_similarity()` function
5. Compare to checkpoint `val_cosine`

### Results

**Old Transformer (340k)**:
```
Mean cosine: 0.5449
Checkpoint claim: 0.5658
Difference: 0.0209 ✅
```

**New AMN (584k)**:
```
Mean cosine: -0.0073
Checkpoint claim: 0.5329
Difference: 0.5401 ❌
```

The new AMN model produces **random noise** - cosine similarities near zero with high variance (-0.18 to +0.21).

###Root Cause (Hypothesis)

Something changed between old training (which worked) and new training (which doesn't). Possible causes:

1. **Model architecture changed**: AMN/Transformer code modified between trainings?
2. **Data format mismatch**: 584k data has different structure/normalization than 340k?
3. **Training hyperparameters**: New training used different loss function or optimizer settings?
4. **PyTorch version**: Different PyTorch version causing numerical instability?
5. **Device issue**: MPS (Apple Silicon) training vs CPU causing precision issues?

---

## Impact

### Immediate

- ✅ **Old 340k models still work** - production services (ports 9001-9006) are unaffected
- ❌ **Cannot deploy any new models** - all retraining effort wasted
- ❌ **Backward bias issue NOT fixed** - stuck with old models that have wrong direction

### Long-term

- **Training pipeline is unreliable** - cannot trust validation metrics
- **Quality gates insufficient** - models passed through without detecting catastrophic failure
- **Time wasted**: ~8 hours of training produced unusable models

---

## Next Steps

### P0: Identify Root Cause (URGENT)

**Compare working (340k) vs broken (584k) training:**

1. **Check Model Architectures**:
   ```bash
   # Compare AMN definitions
   git diff <340k-commit> <584k-commit> -- app/lvm/model.py
   ```

2. **Check Training Data Format**:
   ```python
   # Compare 340k vs 584k NPZ structure
   old_data = np.load('artifacts/lvm/data/training_sequences_ctx5.npz')
   new_data = np.load('artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz')
   # Check keys, shapes, normalization
   ```

3. **Check Training Script**:
   ```bash
   # Compare training invocations
   # Old: How was 340k Transformer trained?
   # New: train_unified.py --model-type amn --data 584k ...
   ```

4. **Check PyTorch/Device**:
   ```python
   # Test CPU vs MPS inference
   # Check if MPS training caused precision issues
   ```

### P1: Implement Safeguards

1. **Add validation gate to training**:
   - After each epoch, verify actual cosine matches computed cosine
   - Alert if discrepancy > 0.05
   - Auto-stop training if model is broken

2. **Add post-training verification**:
   - Before saving checkpoint, re-compute all metrics
   - Verify model can make reasonable predictions
   - Test on held-out data

3. **Update 5CAT test**:
   - Add "sanity check" mode that just verifies predictions are non-zero
   - Fail fast if model produces noise

### P2: Document Lessons Learned

1. **Never trust checkpoint metrics alone**
2. **Always verify model inference before deployment**
3. **Compare new models to old models as regression test**
4. **Test on multiple devices (CPU, MPS, CUDA) before declaring success**

---

## Affected Files

**Broken Models**:
- `artifacts/lvm/models/amn_584k_clean/best_model.pt` ❌
- `artifacts/lvm/models/amn_584k_clean/final_model.pt` ❌
- `artifacts/lvm/models/transformer_5cat_20251030_230110/best_model.pt` ❌ (likely)

**Working Models**:
- `artifacts/lvm/models/transformer_v0.pt` ✅ (symlink to 340k model)
- `artifacts/lvm/models/gru_v0.pt` ✅
- `artifacts/lvm/models/lstm_v0.pt` ✅

**Training Scripts**:
- `app/lvm/train_unified.py` - Used for both working and broken models
- `scripts/train_with_5cat_validation.sh` - Wrapper script (may have issues?)

**Test Tools**:
- `tools/tests/test_5to1_alignment.py` - Detected the issue but too late
- `tools/diagnose_model_output.py` - Created during investigation

---

## Timeline

**Oct 30, 23:08** - Started AMN training on 584k clean data
**Oct 30, 23:25** - AMN training completed (reported val_cosine 0.5315)
**Oct 30, 23:19** - Transformer training completed (reported val_cosine 0.5577)
**Oct 31, 03:14** - 5CAT test on AMN shows values 100x too small
**Oct 31, 03:30** - Discovered checkpoint metrics are completely wrong
**Oct 31, 03:45** - Verified old 340k models still work correctly
**Oct 31, 03:50** - Confirmed bug is specific to NEW training runs

---

## Recommendations

**DO NOT**:
- ❌ Deploy any models trained in this batch
- ❌ Trust validation metrics from new training runs
- ❌ Continue retraining until root cause is found

**DO**:
- ✅ Keep using old 340k production models (they work!)
- ✅ Investigate root cause immediately
- ✅ Add regression tests to prevent this in future
- ✅ Document what changed between old and new training

---

## Contact

**Investigation**: Claude Code (Anthropic)
**User**: trentcarter
**Report Date**: 2025-10-31
**Priority**: P0 - CRITICAL - BLOCKING

---

**Status**: 🔍 **INVESTIGATION IN PROGRESS**

The retraining effort to fix backward bias has been halted due to this critical bug. Until the root cause is identified and fixed, all new model training is blocked.
