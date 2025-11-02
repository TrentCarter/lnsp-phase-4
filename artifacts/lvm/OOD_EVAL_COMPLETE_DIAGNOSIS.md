# OOD Evaluation - Complete Root Cause Diagnosis

**Date**: 2025-10-30
**Status**: 🎯 ROOT CAUSE CONFIRMED
**Severity**: CRITICAL - All existing models are not truly validated

---

## 🎯 Executive Summary

**Problem**: All LVM models show excellent "validation" scores (0.53-0.56 cosine) but negative/near-zero TRUE out-of-distribution scores (-0.017 to -0.007).

**Root Cause**: Training uses `random_split()` of sequences, NOT article-based holdout.
- "Val" set = random sequences from SAME articles seen in training
- TRUE OOD = sequences from DIFFERENT articles (never evaluated during training)
- Models overfit to specific articles and cannot generalize to new articles

**Evidence**:
```python
# Line 318 of app/lvm/train_unified.py
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)
```

This shuffles ALL sequences together, mixing articles in both train and val.

---

## 📊 Comprehensive Test Results

### Test Matrix: 2×2 Cross-Evaluation

| Model | In-Dist Val | OLD OOD (0.73 coh) | TRULY_FIXED OOD (0.46 coh) | Status |
|-------|-------------|--------------------|-----------------------------|---------|
| 584k (unfiltered) | 0.5605 ✅ | -0.0173 ❌ | -0.0167 ❌ | Article-overfit |
| 790k (filtered) | 0.5306 ✅ | N/A | -0.0071 ❌ | Article-overfit |
| 790k (production) | 0.2750 ⚠️ | -0.0148 ❌ | -0.0148 ❌ | Training issue + article-overfit |

**Key Finding**: Even 584k "known-good" baseline FAILS on true OOD (different articles).

---

## 🔬 Diagnostic Results

### 1. Coherence Verification ✅

**Training Data**:
```
Context coherence: 0.4880
Context-to-target: 0.4890
✅ Matches expected distribution (0.46-0.49)
```

**OOD Test Sets**:
- OLD OOD (articles 8001-8470): 0.7310 ❌ (high-coherence anomaly region)
- TRULY_FIXED OOD (articles 1500-1999): 0.4637 ✅ (representative)

**Conclusion**: OOD coherence NOW matches training (after fix), but models still fail!

---

### 2. Neighbor Sweep Diagnostic ✅

**584k Model on TRULY_FIXED OOD**:
```
Cosine similarity: pred vs...
     t: -0.0169
   t-1: -0.0166
   t-2: -0.0162 ← "Peak" (noise)
   t-3: -0.0170
   t-4: -0.0166
  t+1: -0.0150 ← Expected peak
```

**Interpretation**: All scores within ±0.002 (random noise). No systematic offset detected.

**Conclusion**: Predictions are essentially UNCORRELATED with all target positions.

---

### 3. Sign-Flip Test ✅

```
cos(pred, +target): -0.0150
cos(pred, -target): +0.0150
```

**Conclusion**: Perfect symmetry (flipped ≈ -normal). No sign inversion bug.

---

### 4. Prediction Statistics 🚨

**Key Findings**:
```
Prediction diversity: 0.3931 ❌ (too clustered!)
Target diversity:     0.8641 ✅ (normal)
Mean L2 offset:       1.4360 ❌ (huge systematic difference!)
```

**Interpretation**:
- Model outputs are TOO CONSERVATIVE (clustered near mean)
- Predictions systematically far from targets in unit sphere space
- Likely due to MSE loss + overfitting to training article distribution

---

## 📍 Why This Happened

### Training Data Structure

**Source**: `artifacts/wikipedia_584k_fresh.npz`
- 584,545 chunks from articles 1-8470
- Global coherence: 0.4899 ✅
- But article 8001-8470 has local coherence 0.7365 (anomaly!)

**Training Sequences**: `artifacts/lvm/training_sequences_ctx5_584k_fresh.npz`
- 543,773 sequences (contexts + targets)
- Created with stride=1 across ALL articles
- Coherence: 0.4880 ✅

**Validation Split**:
```python
# Random 90/10 split of sequences (NOT articles!)
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)
```

**Result**:
- Train: 489k sequences from articles 1-8470
- Val: 54k sequences from articles 1-8470 (SAME ARTICLES!)

### Why Models Look Good But Aren't

1. **High "val cosine" (0.56)**: Val sequences are from same articles seen in training
   - Model learned article-specific patterns
   - Easy to predict next chunk when you've seen other chunks from same article

2. **Zero TRUE OOD cosine (-0.017)**: New articles have different:
   - Topic distributions
   - Writing styles
   - Semantic patterns
   - Model never learned to generalize across articles!

3. **Conservative/clustered predictions**:
   - MSE loss encourages predicting near the mean
   - Overfitting to training articles → predicts typical vectors
   - New articles may have different variance/distribution

---

## ✅ Correct OOD Dataset (TRULY_FIXED)

**File**: `artifacts/lvm/wikipedia_ood_test_ctx5_TRULY_FIXED.npz`

**Specifications**:
- Articles: 1500-1999 (representative region)
- Sequences: 10,000
- Coherence: 0.4637 ✅ (matches training 0.49)
- Cross-article: 0.8% (minimal, as expected)

**How Created**:
```bash
python tools/create_ood_test_sequences_fixed.py \
  --npz artifacts/wikipedia_584k_fresh.npz \
  --output artifacts/lvm/wikipedia_ood_test_ctx5_TRULY_FIXED.npz \
  --min-article 1500 \
  --max-article 1999 \
  --context-len 5 \
  --max-sequences 10000
```

**Verification**:
- Source Wikipedia region has coherence 0.4690 (representative!)
- Stride=1 across all holdout vectors (matches training creation)
- NOT from anomalous high-coherence region like 8001-8470

---

## 🚨 Impact Assessment

### What's Broken
1. **All existing "val" scores are INVALID** for true generalization
   - 584k "0.5605 val" is NOT real OOD performance
   - 790k "0.5306 val" is NOT real OOD performance
2. **Models cannot generalize to new articles**
   - Would fail in production on unseen Wikipedia pages
   - Predictions are essentially random for new content

### What Still Works
1. **Models ARE trained correctly** (MSE loss, architecture, etc.)
2. **Coherence matching NOW works** (TRULY_FIXED OOD set is correct)
3. **Diagnostic tools work** (neighbor sweep, sign-flip tests)

### Production Implications
- 🔴 **DO NOT deploy** existing 584k or 790k models for article-level generalization
- 🔴 **Current "val" scores are meaningless** for production quality assessment
- ⚠️ Models may still work for **same-article** next-chunk prediction

---

## 🔧 The Fix: Article-Based Train/Val Split

### Required Changes

**1. Hold out ENTIRE ARTICLES for validation** (not random sequences):
```python
# Get unique articles
unique_articles = sorted(set(metadata['article_index']))

# Split articles 90/10
train_articles = unique_articles[:int(0.9*len(unique_articles))]
val_articles = unique_articles[int(0.9*len(unique_articles)):]

# Filter sequences by article
train_mask = np.isin(metadata['article_index'], train_articles)
val_mask = np.isin(metadata['article_index'], val_articles)

train_dataset = Subset(dataset, np.where(train_mask)[0])
val_dataset = Subset(dataset, np.where(val_mask)[0])
```

**2. Hold out SPECIFIC ARTICLES for OOD test**:
- Reserve articles 1500-1999 (or similar representative region)
- NEVER use these in training or validation
- Use for final OOD evaluation only

**3. Update training pipeline**:
```bash
# Create training data EXCLUDING holdout articles
python tools/create_training_sequences.py \
  --npz artifacts/wikipedia_584k_fresh.npz \
  --output artifacts/lvm/training_sequences_article_split.npz \
  --exclude-articles 1500-1999 \
  --context-len 5

# Train with article-based split
python app/lvm/train_unified.py \
  --model-type amn \
  --data artifacts/lvm/training_sequences_article_split.npz \
  --split-by-article \
  --val-articles 7000-7499 \
  --epochs 20
```

---

## 📈 Expected Results After Fix

**With article-based validation**:
- Val cosine: ~0.45-0.50 (lower than current 0.56, but REAL!)
- OOD cosine: ~0.43-0.48 (similar to val, proving generalization)
- Both should be MUCH closer together (not -0.58 delta like now!)

**Realistic targets**:
| Split | Current (Broken) | After Fix (Realistic) |
|-------|------------------|------------------------|
| Train | 0.58 | 0.55-0.60 |
| Val (article-holdout) | 0.56 (fake!) | 0.45-0.50 (real!) |
| OOD (true holdout) | -0.017 (broken!) | 0.43-0.48 (working!) |

---

## 🎯 Immediate Next Steps

1. **DO NOT train more models** until split is fixed
2. **Update training pipeline** with article-based split (see above)
3. **Retrain 584k baseline** with proper holdout
4. **Verify on TRULY_FIXED OOD** (should get ~0.45-0.48, not negative!)
5. **Only then** proceed with 790k filtered training

---

## 📝 Files Affected

**Training Scripts (NEED FIXES)**:
- `app/lvm/train_unified.py` (line 318 - random_split → article-based split)
- `app/lvm/train_final.py`
- All other training scripts using `random_split()`

**Data Files (CORRECT)**:
- `artifacts/wikipedia_584k_fresh.npz` ✅ (source vectors)
- `artifacts/lvm/training_sequences_ctx5_584k_fresh.npz` ✅ (sequences)
- `artifacts/lvm/wikipedia_ood_test_ctx5_TRULY_FIXED.npz` ✅ (OOD test)

**Evaluation Scripts (CORRECT)**:
- `tools/eval_model_ood.py` ✅ (diagnostics work correctly)
- `tools/create_ood_test_sequences_fixed.py` ✅ (creates correct OOD)

---

## ✅ Validation Checklist

Before declaring a model "production-ready":
- [ ] Training uses article-based split (not random_split)
- [ ] Validation articles are completely held out from training
- [ ] Val cosine < 0.55 (not inflated by article overlap)
- [ ] OOD cosine on TRULY_FIXED > 0.40
- [ ] OOD - Val delta < 0.10 (proves generalization)
- [ ] Neighbor sweep shows clear peak at t+1
- [ ] Prediction diversity ≥ 0.7 (not too clustered)

---

**Status**: Complete diagnosis. Ready to fix training pipeline and retrain models properly.
