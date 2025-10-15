# Task 1.2 & 1.3: Diagnostic Test Findings

**Date**: October 13, 2025
**Status**: Both tasks completed with surprising findings

---

## Executive Summary

1. **Task 1.3 (L2 normalization test)**: Hypothesis **REJECTED** - GTR-T5 always outputs normalized vectors
2. **Task 1.2 (Mean vector baseline)**: **Models are untrained**, not collapsed
3. **New discovery**: Current "best" models only trained for 1 epoch (essentially random)

---

## Task 1.3: Decoder A/B/C Test Results

### Hypothesis Tested
> "L2 normalization breaks vec2text decoder which expects raw encoder distribution"

###Result: ❌ **REJECTED**

### Evidence

1. **GTR-T5 encoder ALWAYS normalizes**
   ```python
   # SentenceTransformer GTR-T5-base
   embedding = model.encode(["test"], normalize_embeddings=False)[0]
   norm = np.linalg.norm(embedding)
   # Result: 1.000000 (ALWAYS L2-normalized)
   ```

2. **Our custom encoder also normalizes**
   - File: `app/vect_text_vect/vec_text_vect_isolated.py:282`
   - Explicitly calls `F.normalize(s, p=2, dim=1)`

3. **Vec2text decoder preprocessing**
   - File: `app/vect_text_vect/vec2text_processor.py:159`
   - Also normalizes inputs: `F.normalize(embedding, dim=-1)`

### Conclusion

Vec2text decoder was **trained on and expects L2-normalized embeddings**. The "raw distribution" concept doesn't apply because there is no unnormalized mode in GTR-T5.

**L2 normalization is NOT the root cause** of poor LVM performance.

---

## Task 1.2: Mean Vector Baseline Results

### Hypothesis Tested
> "Model has collapsed to outputting the global mean vector"

### Result: ⚠️ **MODELS ARE UNTRAINED**

### Measurements

**Baseline (using global mean vector)**:
- Global mean norm: 0.4856
- Average cosine (target vs global mean): **48.16%**

**"Best" GRU Model**:
- Training: 1 epoch only
- Validation cosine: **0.05%** (essentially random)
- Status: **Untrained**

**LSTM Model** (from Oct 13 training history):
- Epoch 1: train_cosine = -0.99%, val_cosine = -0.89%
- Status: **Failed to train** (negative cosine!)

**Transformer Model** (from Oct 13 training history):
- Epoch 1: train_cosine = 0.12%, val_cosine = -0.26%
- Status: **Barely initialized**

### Conclusion

The models are **not mode-collapsed** - they're simply **untrained or failed to train**. Mode collapse would show 48% cosine (matching the baseline). Instead, we see ~0% cosine (random outputs).

---

## Root Cause Analysis

### What We Thought Was Happening

❌ L2 normalization breaks decoder
❌ Model collapsed to mean vector
❌ Distribution mismatch

### What's Actually Happening

1. **Training failures**: Recent training runs (Oct 12-13) only completed 1 epoch or failed
2. **Old training histories**: The 76-78% cosine we saw were from older model versions
3. **Architecture mismatch**: The models being tested don't match the training code

### Evidence

**GRU/Mamba2 model checkpoint**:
```python
'val_cosine': 0.0005  # 0.05% - essentially random
'epoch': 1             # Only 1 epoch trained
```

**LSTM training history** (Oct 13):
```json
{"epoch": 1, "train_cosine": -0.0099, "val_cosine": -0.0088}
```

**Transformer training history** (Oct 13):
```json
{"epoch": 1, "train_cosine": 0.0012, "val_cosine": -0.0026}
```

---

## Implications

### Task List Updates

The original `LVM_FIX_TASKLIST.md` was based on incorrect assumptions:
- ❌ Task 1.3: L2 norm doesn't break decoder
- ❌ Task 2.1: Dual output heads not needed
- ❌ Task 2.2: Moment matching for "raw" dist N/A
- ✅ Task 2.3: InfoNCE loss still good idea
- ✅ Task 2.4: Hard negatives still useful

### Real Issues to Fix

1. **Training didn't complete**: All 3 models stopped after 1 epoch
   - Need to check why training failed/stopped
   - Background training processes may have crashed

2. **Negative cosine values**: LSTM showing -0.99% cosine
   - Suggests initialization or loss function issues
   - Model outputting vectors opposite to targets

3. **Training data**: Using `training_sequences_ctx5_sentence.npz` (8,106 sequences)
   - Need to verify this data is good
   - Check if it's sequential Wikipedia (correct) vs ontological (wrong)

---

## Next Steps

### Immediate Actions

1. **Check training process status**
   ```bash
   # Check if background training is still running
   ps aux | grep train_
   ```

2. **Verify training data quality**
   ```python
   data = np.load('artifacts/lvm/training_sequences_ctx5_sentence.npz')
   # Check metadata, source, statistics
   ```

3. **Restart training properly**
   - Use correct data
   - Monitor full 20 epochs
   - Watch for failure modes

### Training Configuration

Based on the baseline measurement (48.16%), a good model should achieve:
- **Minimum target**: 55-60% cosine (10+ points above baseline)
- **Good performance**: 70-80% cosine
- **Excellent**: 85%+ cosine

### Fixes to Apply Before Retraining

1. **Loss function**: InfoNCE contrastive loss (not MSE)
2. **Initialization**: Check weight initialization isn't broken
3. **Learning rate**: Current 0.0005 may be too high/low
4. **Batch construction**: Ensure diverse samples in each batch

---

## Corrected Understanding

### GTR-T5 Encoder Behavior
- **Output**: L2-normalized 768D vectors (norm=1.0)
- **Distribution**: Mean≈0, Std≈0.036 per dimension
- **No "raw" mode**: Always normalizes, by design

### Vec2text Decoder Expectations
- **Input**: L2-normalized vectors (same as encoder output)
- **Training**: Decoder was trained on these normalized vectors
- **Key requirement**: Input diversity (different vectors for different texts)

### LVM Training Requirements
- **Architecture**: Single output head is fine (already normalized)
- **Loss**: Contrastive (InfoNCE) better than MSE
- **Data**: Sequential sources (Wikipedia), not ontological
- **Training time**: Full 20 epochs minimum
- **Success metric**: >55% cosine (beating 48% baseline)

---

## Files Created

- `TASK_1_3_FINDINGS.md` - Detailed L2 normalization analysis
- `tools/test_decoder_distributions.py` - A/B/C test script
- `tools/test_mean_vector_baseline_simple.py` - Baseline measurement script

---

## Key Insight

The real problem isn't mode collapse or normalization - **the models just haven't been trained properly yet**. The background training from Oct 12-13 appears to have failed or stopped after 1 epoch.

We need to:
1. Verify training data is sequential (not ontological)
2. Fix any training script issues
3. Complete full 20-epoch training runs
4. Monitor training curves to catch failures

The 48.16% baseline is a useful target: any model scoring below this is worse than just returning the mean vector.
