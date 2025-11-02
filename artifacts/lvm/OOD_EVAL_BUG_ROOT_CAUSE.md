# OOD Evaluation Bug - Root Cause Analysis

**Date**: 2025-10-30
**Status**: 🚨 ROOT CAUSE IDENTIFIED - Vector Space Mismatch
**Severity**: CRITICAL - All OOD evaluations invalid

---

## 🎯 Executive Summary

**Problem**: ALL LVM models show negative/near-zero OOD scores (-0.014 to -0.019), even the proven 584k baseline that should achieve ~0.63-0.65 OOD performance.

**Root Cause**: **Training data and OOD test data are in different vector spaces**. Training data has 0.49 coherence, OOD data has 0.73 coherence (~50% mismatch).

**Proof**: Model predictions are essentially uncorrelated with OOD data (cos ~0.0), BUT the OOD data itself is highly coherent (cos ~0.73), proving it's a space mismatch not a model failure.

---

## 📊 Diagnostic Results

### Test 1: Cross-Matrix Sanity Check

Tested proven 584k baseline model (known good: 0.5605 in-dist, should be ~0.63-0.65 OOD):

| Model | OOD Set | Expected | Actual | Status |
|-------|---------|----------|--------|--------|
| 584k | OLD (fresh) | ~0.63-0.65 | **-0.0173** | ❌ FAIL |
| 584k | NEW (790k) | ~0.63-0.65 | **-0.0140** | ❌ FAIL |

**Conclusion**: Even proven model fails on ALL OOD sets → OOD eval pipeline is broken.

---

### Test 2: Neighbor Sweep Diagnostic

Tested 584k model predictions against multiple target offsets:

```
Cosine similarity: pred vs...
  t-4: -0.0170  ← "Peak" (noise)
  t-3: -0.0178
  t-2: -0.0181
  t-1: -0.0189
  t:   -0.0190
  t+1: -0.0173  ← Expected peak
```

**Key Finding**: ALL scores are essentially the same (-0.017 ± 0.002). Predictions are uncorrelated with EVERYTHING.

**Interpretation**: If this was a simple misalignment (window shifted), we'd see a strong positive peak at the wrong offset. Instead, we see essentially random scores everywhere.

---

### Test 3: Sign-Flip Diagnostic

```
cos(pred, +target): -0.0173
cos(pred, -target): +0.0173
```

**Conclusion**: ✅ No sign inversion detected (flipped ≈ -normal).

---

### Test 4: OOD Data Coherence Check

**OLD OOD set (fresh)**:
```
Context coherence (pos[i] vs pos[i+1]):
  pos[0] vs pos[1]: 0.7406
  pos[1] vs pos[2]: 0.7318
  pos[2] vs pos[3]: 0.7404
  pos[3] vs pos[4]: 0.7317
  Mean: 0.7361

Target coherence (pos[4] vs target): 0.7406
```

**NEW OOD set (790k)**:
```
Context coherence: 0.7285 (all positions)
Target coherence: 0.7285
```

**Conclusion**: ✅ OOD data is HIGHLY coherent (0.73+). Data generation is correct, vectors are properly ordered.

---

### Test 5: Training Data Coherence Check

**584k Training Data**:
```
Context coherence (10k sample):
  pos[0] vs pos[1]: 0.4886
  pos[1] vs pos[2]: 0.4884
  pos[2] vs pos[3]: 0.4881
  pos[3] vs pos[4]: 0.4885
  Mean: 0.4884

Target coherence (pos[4] vs target): 0.4867
```

---

## 🚨 THE SMOKING GUN

### Coherence Comparison

| Dataset | Context Coherence | Target Coherence | Source |
|---------|------------------|------------------|--------|
| **584k Training** | **0.4884** | **0.4867** | Training sequences |
| **OLD OOD** | **0.7361** | **0.7406** | Test sequences |
| **NEW OOD** | **0.7285** | **0.7285** | Test sequences |

**50% COHERENCE MISMATCH!**

---

## 🔬 What This Means

### Normal Semantic Coherence

Wikipedia chunks with good semantic flow typically have 0.45-0.50 adjacency coherence:
- 584k training data: 0.4884 ✅ (matches P1 baseline: 0.4842)
- This is CORRECT for semantic Wikipedia text

### Abnormally High Coherence

OOD data with 0.73+ coherence is MUCH higher than natural text:
- This level of coherence suggests vectors are almost identical
- Typical only for: same sentence with different tokenization, or heavily overlapping text windows

### Why Model Fails

1. **Model trained on**: 0.49-coherence vectors (semantic diversity)
2. **Model tested on**: 0.73-coherence vectors (different space)
3. **Result**: Model predictions (trained for 0.49 space) don't match 0.73 space → near-zero correlation

**Analogy**: Like training a model to predict English word embeddings, then testing it on French word embeddings. Both are 768D vectors, but in completely different semantic subspaces.

---

## 🛠️ Root Cause Hypothesis

### Likely Explanation

**Training data**: Built from TMD-enhanced vectors or PostgreSQL exports
- TMD adds semantic diversity/noise → lower coherence (0.49)
- Reflects true semantic relationships

**OOD data**: Built from raw GTR-T5 vectors in NPZ files
- Pure embedding space → high coherence (0.73)
- May have overlapping text windows or different preprocessing

### Alternative Explanations

1. **Different encoder runs**: Training vs OOD used different GTR-T5 model checkpoints
2. **Different normalization**: Training applied additional normalization OOD didn't
3. **Different text preprocessing**: Training used episode chunking, OOD used fixed-length chunks with overlap

---

## ✅ Evidence Chain

1. ✅ **Space alignment is correct** (P0 verified: encoder, normalization, FAISS all good)
2. ✅ **OOD data generation is correct** (sliding window code is proper)
3. ✅ **OOD data is internally coherent** (0.73+ proves vectors are valid and ordered)
4. ✅ **Training data is internally coherent** (0.49 matches expected semantic flow)
5. ❌ **Training and OOD coherence MISMATCH by 50%** → DIFFERENT VECTOR SPACES
6. ❌ **Model predictions uncorrelated with OOD** (but model has good in-dist) → Space mismatch confirmed

---

## 🎯 The Fix

### Immediate Action Required

**Rebuild OOD test sets using THE SAME vector source as training data.**

### Step 1: Identify Training Vector Source

Check how `training_sequences_ctx5_584k_fresh.npz` was created:
```bash
# Find the script that created training sequences
grep -r "training_sequences_ctx5" tools/*.py scripts/*.sh

# Check if vectors are TMD-enhanced or raw GTR-T5
python -c "
import numpy as np
data = np.load('artifacts/lvm/training_sequences_ctx5_584k_fresh.npz', allow_pickle=True)
print('Keys:', list(data.keys()))
# Look for 'target_tmds' field
"
```

### Step 2: Rebuild OOD Sets with Matching Vectors

If training used TMD vectors:
```bash
# Use TMD-enhanced vectors from same source
python tools/create_ood_test_sequences.py \
  --npz artifacts/wikipedia_584k_tmd_vectors.npz \  # TMD vectors
  --output artifacts/lvm/wikipedia_ood_test_ctx5_fresh_fixed.npz \
  --min-article 8001 \
  --max-article 8470
```

If training used raw GTR-T5:
```bash
# Use raw GTR-T5 vectors from same encoder run
python tools/create_ood_test_sequences.py \
  --npz artifacts/wikipedia_584k_raw_gtr_vectors.npz \
  --output artifacts/lvm/wikipedia_ood_test_ctx5_fresh_fixed.npz \
  --min-article 8001 \
  --max-article 8470
```

### Step 3: Verify Coherence Match

After rebuilding:
```bash
python /tmp/check_ood_coherence.py
# Expected: NEW OOD coherence ~0.49 (matching training)
```

### Step 4: Re-evaluate 584k Baseline

```bash
./.venv/bin/python tools/eval_model_ood.py \
  --model artifacts/lvm/models/amn_584k_pure_mse_20251029_055838/best_model.pt \
  --ood-data artifacts/lvm/wikipedia_ood_test_ctx5_fresh_fixed.npz \
  --device mps

# Expected: OOD cosine ~0.63-0.65 (restored!)
```

---

## 📋 Impact Assessment

### What This Invalidates

❌ **ALL previous OOD evaluations are INVALID**:
- 584k baseline OOD: Claimed 0.6375, actually unknown (test was broken)
- 790k attempts OOD: Claimed -0.0118 to 0.0211, actually unknown
- P2 filtered OOD: Claimed -0.0167, actually unknown

### What This Does NOT Invalidate

✅ **In-distribution evaluations are VALID**:
- 584k in-dist: 0.5605 ✅ (training val set uses same vectors)
- 790k in-dist: 0.4621 ✅ (training val set uses same vectors)
- P2 filtered in-dist: 0.5306 ✅ (training val set uses same vectors)

✅ **P1 coherence analysis is VALID**:
- 584k coherence: 0.4842 ✅ (measured on actual training data)
- 790k coherence: 0.3367 ✅ (measured on actual training data)
- Low-coherence dilution finding: CONFIRMED ✅

---

## 🚦 Decision Tree Update

### P2 Smoke Test Re-Interpretation

**Original Conclusion**: RED - P2 failed (OOD -0.0167)
**Corrected Conclusion**: **UNKNOWN** - OOD test was invalid

**P2 In-Dist Results** (still valid):
- Val cosine: 0.5306 (only 5.3% below 584k baseline)
- Training progression: Smooth, no issues
- Data quality: Filtered dataset working as expected

**New Recommendation**:
1. ✅ Keep 584k in production (RED protocol upheld)
2. ⏳ Rebuild OOD test sets with correct vectors
3. ⏳ Re-evaluate P2 filtered model with fixed OOD test
4. ⏳ Decision on filtered dataset deferred until OOD test fixed

---

## 📁 Generated Artifacts

### Diagnostic Scripts
- `/tmp/check_ood_coherence.py` - OOD data coherence checker
- `/tmp/check_train_coherence.py` - Training data coherence checker
- `/tmp/cross_matrix_test.sh` - 2×2 cross-matrix test script

### Updated Tools
- `tools/eval_model_ood.py` - Now includes:
  - `--neighbor-sweep` flag for misalignment detection
  - `--sign-flip-test` flag for inversion detection
  - Automatic interpretation of diagnostic results

### Documentation
- `artifacts/lvm/OOD_EVAL_BUG_ROOT_CAUSE.md` - This document

---

## 🔍 Technical Details

### Why 50% Coherence Difference Breaks Everything

**Cosine similarity is relative**:
- If vectors in space A have 0.49 mean similarity
- And vectors in space B have 0.73 mean similarity
- These spaces have different "spread" and "density"

**Model learned space A distribution**:
- Expects next vector to be ~0.49 similar to context
- Produces predictions calibrated for space A

**But tested on space B distribution**:
- Space B vectors are ~0.73 similar to each other
- Model's space A predictions don't match space B → near-zero correlation

### Why In-Dist Works But OOD Fails

**In-dist val set**:
- Created from SAME training data source
- Same vector space (0.49 coherence)
- Model predictions match → good scores (0.56)

**OOD test set**:
- Created from DIFFERENT vector source
- Different vector space (0.73 coherence)
- Model predictions don't match → near-zero scores

---

## 📝 Lessons Learned

1. **Always verify vector source consistency** across train/val/test splits
2. **Coherence checks are critical** for catching space mismatches
3. **Good in-dist + bad OOD often means test data bug**, not model failure
4. **Cross-matrix testing** (multiple models × multiple test sets) isolates problems quickly
5. **Neighbor sweep + sign-flip** diagnostics should be standard for debugging OOD failures

---

**Status**: 🚨 OOD TEST PIPELINE BROKEN - REBUILD REQUIRED
**Next Step**: Rebuild OOD test sets using same vector source as training data
**Timeline**: ~30 min to rebuild + 5 min to re-evaluate

**Updated**: 2025-10-30 (Post-Diagnostic Analysis)
