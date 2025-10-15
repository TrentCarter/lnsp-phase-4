# Task 1.3: Decoder A/B/C Test - Findings

**Date**: October 13, 2025
**Test**: Whether L2 normalization breaks vec2text decoder
**Result**: ❌ **HYPOTHESIS REJECTED**

---

## Executive Summary

The hypothesis that "L2 normalization breaks vec2text decoder" is **fundamentally incorrect**. Testing reveals that:

1. **GTR-T5 encoder ALWAYS outputs L2-normalized vectors** (norm=1.0)
2. **Vec2text decoder was trained on these normalized vectors**
3. **L2 normalization cannot be the root cause** of LVM decode failures

---

## Evidence

### 1. GTR-T5 Encoder Behavior

Test with SentenceTransformer (the official encoder):

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/gtr-t5-base')
embedding = model.encode(["test"], normalize_embeddings=False)[0]
norm = np.linalg.norm(embedding)
# Result: norm = 1.000000 (ALWAYS normalized)
```

**Finding**: Even with `normalize_embeddings=False`, GTR-T5 returns unit-norm vectors.

### 2. Our Custom Encoder

File: `app/vect_text_vect/vec_text_vect_isolated.py:282`

```python
# Normalize
embedding = F.normalize(s, p=2, dim=1)
```

**Finding**: Our encoder explicitly normalizes (matching official behavior).

### 3. Vec2text Decoder Preprocessing

File: `app/vect_text_vect/vec2text_processor.py:159`

```python
def _prepare_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
    embedding = F.normalize(embedding, dim=-1)
    return embedding
```

**Finding**: Decoder wrapper ALSO normalizes inputs before decoding.

### 4. Test Results

Test A (raw encoder output):
- Vector norm: 1.0000 (already normalized)
- Mean: -0.001701, Std: 0.036044

Test B (L2 normalized):
- Vector norm: 1.0000 (identical to Test A)
- Mean: -0.001701, Std: 0.036044 (IDENTICAL)

**Finding**: Raw and "normalized" are the same - no unnormalized baseline exists.

---

## Implications

### What This Means

1. **Vec2text expects normalized inputs** - it was trained with GTR-T5's normalized outputs
2. **The "dual output heads" approach may be unnecessary** - both heads would need to be normalized
3. **The real problem is likely mode collapse**, not distribution mismatch

### What Was Wrong With the Hypothesis

The Manager stated:
> "Vec2text decoder expects RAW encoder distribution (not L2-normalized)"

But:
- GTR-T5 encoder has NO "raw" mode - it always normalizes
- Vec2text paper/implementation never mentions unnormalized embeddings
- All our code (3 separate places) normalizes before decoding

---

## Actual Root Cause (Hypothesis)

Based on training history analysis:

### Evidence of Mode Collapse

**GRU Model (Mamba2VectorPredictor)**:
- Epoch 1: train_cosine=0.5093, val_cosine=0.6963
- Epoch 20: train_cosine=0.8644, val_cosine=0.7642

**But inference tests show**:
- Sample 1 vs 2 cosine: 0.9732 (97.3% similar!)
- Sample 1 vs 3 cosine: 0.9421 (94.2% similar!)
- Different inputs → nearly identical outputs

**LSTM Model (NEW - Oct 13)**:
- Epoch 1: train_loss=6.954, train_cosine=-0.0099, val_cosine=-0.0088

**Transformer Model (NEW - Oct 13)**:
- Epoch 1: train_loss=6.950, train_cosine=0.0012, val_cosine=-0.0026

### Root Causes

1. **MSE loss encourages mean vector** → model learns to output average of training set
2. **Batch construction** → near-duplicate samples in batches make average look good
3. **Single projection head** → easy to collapse to constant output
4. **Weak regularization** → nothing prevents outputting same vector for everything

---

## Recommended Next Steps

### Abandon Task 1.3 Approach

- L2 normalization testing is moot
- "Raw distribution" concept doesn't apply
- Dual output heads may not help

### Focus on Mode Collapse (Task 1.2)

File: `LVM_FIX_TASKLIST.md` - Task 1.2

Test whether model outputs are just the global mean vector:

```python
# Compute mean of all training targets
global_mean = training_targets.mean(axis=0)

# Compare model outputs to global mean
for sample in test_samples:
    model_output = model(sample)
    cosine_to_mean = cosine_similarity(model_output, global_mean)
    # If cosine_to_mean > 0.95, model collapsed to mean
```

### Architecture Fixes (Without Dual Heads)

Keep normalized outputs, but fix mode collapse:

1. **InfoNCE contrastive loss** (not MSE) - encourages diversity
2. **Variance regularizer** - penalize if all outputs similar
3. **Diverse batch sampling** - ensure batch has dissimilar samples
4. **Stronger projection head** - 2-layer with LayerNorm

---

## Files to Update

### Remove/Update Misleading Content

- `LVM_FIX_TASKLIST.md` - Tasks 2.1, 2.2, 4.1, 4.2 assume dual heads
  - Keep: Task 1.2 (mean vector baseline), Task 2.3 (InfoNCE), Task 2.4 (hard negatives)
  - Remove: Task 2.1 (split heads), Task 2.2 (moment matching for raw dist)

### Keep These Fixes

- **Task 2.3**: InfoNCE loss (contrastive, not MSE)
- **Task 2.4**: Hard negatives in batches
- **Task 3.2**: Diverse batch sampler
- **Task 6.1-6.2**: Retraining and validation

---

## Corrected Understanding

### What Vec2text Actually Expects

- **Input**: L2-normalized 768D vectors (norm=1.0)
- **Distribution**: Mean≈0, Std≈0.036 per dimension
- **Diversity**: Different texts → different embeddings (high variance across samples)

### What LVMs Are Producing

- **Input**: L2-normalized 768D vectors (norm=1.0) ✅
- **Distribution**: Mean≈0, Std≈0.036 per dimension ✅
- **Diversity**: Different contexts → SAME embedding ❌ **ROOT CAUSE**

---

## Conclusion

Task 1.3 revealed that the original hypothesis was based on a misunderstanding of how GTR-T5 and vec2text work. The real issue is **mode collapse**: the LVM outputs nearly identical vectors for different inputs, making semantic decoding impossible.

The fix is NOT dual output heads, but rather:
1. Contrastive loss (InfoNCE) instead of MSE
2. Variance regularization
3. Diverse batch sampling
4. Better training data (sequential, not ontological)

**Next Action**: Run Task 1.2 (Mean Vector Baseline) to quantify mode collapse, then proceed with architecture fixes focused on diversity, not normalization.
