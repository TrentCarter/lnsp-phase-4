# Mamba Phase-5: Root Cause Analysis
## 0% Retrieval Despite Good Training Metrics

**Date**: 2025-10-26
**Models Affected**: All 5 Mamba models (S, H, XL, Sandwich, GR)
**Status**: 🔴 **CRITICAL - Data Mismatch Between Training and Retrieval**

---

## Executive Summary

All Mamba Phase-5 models achieve **0% retrieval** (R@1, R@5, R@10, Contain@50) despite excellent training metrics (0.54-0.58 validation cosine).

**Root cause**: Training data targets and FAISS payload vectors come from **different sources/encodings** with only ~0.14 cosine similarity to each other.

**Impact**: Models learned to predict vectors from distribution A, but retrieval searches distribution B → complete retrieval failure.

**Solution**: Option B - Retrain all models using payload-aligned training data.

---

## Timeline of Discovery

### 1. Initial Observation
**Symptom**: All models showed 0% retrieval metrics
```
R@1:          0.00%
R@5:          0.00%
R@10:         0.00%
Contain@50:   0.00%
```

**But**: Validation metrics looked good during training
```
Mamba-S:         0.5760 val cosine
Mamba-H:         0.5767 val cosine
Mamba-Sandwich:  0.5797 val cosine
```

### 2. Hypothesis 1: Retrieval Gap
**Theory**: Models not learning GTR-T5 space well enough
- Validation cosine measures pred-to-target similarity
- Retrieval requires beating 584k distractors
- 0.54-0.58 might not be high enough

**Investigation**: Created `diagnose_retrieval_gap.py`

**Results** (Mamba-Sandwich, 1000 samples):
```json
{
  "validation_cosine": {"mean": 0.5369},     // Pred vs eval target
  "rank_of_truth": {
    "in_top_50_pct": 0.0,
    "in_top_500_pct": 0.0,
    "in_top_5000_pct": 0.4,
    "not_in_top_10000_pct": 99.0,
    "median_rank": 6301                       // Out of 584k
  },
  "cosine_margins": {
    "mean": -0.4146,                          // Negative!
    "positive_pct": 0.0
  },
  "impostor_analysis": {
    "mean_cos_rank1": 0.6869,                 // Impostor wins
    "mean_cos_true": 0.2724,                  // Ground truth LOSES
    "gap": 0.4146
  }
}
```

**Key Finding**: Cosine to ground truth (0.27) << Cosine to impostor (0.69)

### 3. Hypothesis 2: Eval Target Mismatch
**Theory**: Eval targets don't match payload vectors

**Investigation**: Created `verify_vector_mismatch.py`
```python
# Compare eval targets vs payload vectors
for i in range(len(truth_keys)):
    eval_target = eval_targets[i]
    payload_vec = payload[payload_id]
    cos = np.dot(eval_target, payload_vec)

mean_cosine = 0.1430  # ❌ VERY LOW!
```

**Consultant Feedback**:
> "A 0.14 mean cosine between eval targets and payload vectors is a smoking-gun data mismatch. The eval targets were generated from a different model/process than the payload vectors!"

**Action Taken**: Created alignment tool to replace eval targets with payload vectors
- Coverage: 100.00% (5244/5244 samples)
- Cosine improvement: 0.1430 → 1.0000
- Output: `artifacts/lvm/eval_v2_payload_aligned.npz`

### 4. Smoke Test with Aligned Data
**Expectation**: Contain@50 should jump to 60-75%, R@5 to 40-55%

**Actual Results**:
```
Validation cosines (pred vs target):
  Mean: 0.2517                                // ❌ DROPPED from 0.54!

Retrieval Metrics:
  R@1:          0.00%
  R@5:          0.00%
  R@10:         0.00%
  Contain@50:   0.00%                         // ❌ Still 0%!
```

**Critical Insight**: Aligning eval targets EXPOSED the real problem:
- Before alignment: 0.54 cosine (pred vs mismatched eval target)
- After alignment: 0.25 cosine (pred vs actual payload vector)
- The 0.25 matches the diagnostic's `mean_cos_true: 0.2724`

### 5. Root Cause: Training Data Mismatch
**Investigation**: Compare training targets vs payload vectors

**Training Data** (`artifacts/lvm/training_sequences_ctx5.npz`):
```
context_sequences: (232600, 5, 768)
target_vectors:    (232600, 768)
- L2 normalized (norm = 1.0)
- 232,600 training sequences
- NO truth_keys (no linkage to payload!)
```

**Payload** (`artifacts/wikipedia_584k_payload.npy`):
```
584,545 vectors
- L2 normalized (norm = 1.0)
- Includes (article_index, chunk_index) metadata
- Different source/encoding!
```

**Comparison** (100 random pairs):
```python
Random cosine (train_targets vs random_payload_vecs): 0.1442
```

**Interpretation**: ~0.14 cosine indicates **completely different vector distributions**

---

## Technical Analysis

### Why Models Trained Successfully
Models achieved 0.54-0.58 validation cosine because:
1. They learned to predict vectors from the training distribution
2. Training data was internally consistent (contexts → targets from same source)
3. Validation split came from same distribution as training

### Why Retrieval Failed
Retrieval failed because:
1. **Training targets**: From `training_sequences_ctx5.npz` (unknown source)
2. **Payload vectors**: From Wikipedia GTR-T5 encoding (known source)
3. **Cosine between distributions**: ~0.14 (essentially unrelated)
4. **Result**: Models predict into space A, retrieval searches space B

### Visualization
```
Training Phase:
  Context (5 vecs) → Model → Prediction
                             ↓ cosine=0.54-0.58 ✅
                          Target (from training_sequences_ctx5.npz)

Retrieval Phase:
  Context (5 vecs) → Model → Prediction
                             ↓ cosine=0.25 ❌
                          Truth (from wikipedia_584k_payload.npy)
                             ↓ cosine=0.69 ⚠️
                          Impostor (random payload vector)
```

---

## Data Provenance Gap

### Training Data
**File**: `artifacts/lvm/training_sequences_ctx5.npz`
- ❌ No provenance metadata
- ❌ No truth_keys linking to payload
- ❌ No record of encoder/embedder used
- ❌ Unknown creation date/method

### Payload Data
**File**: `artifacts/wikipedia_584k_payload.npy`
- ✅ Created: 2025-10-24
- ✅ Embedder: GTR-T5-base-768
- ✅ Contains article_index, chunk_index
- ✅ Includes original text
- ✅ SHA256 hash: `12cfd8d7d92dca99`

### Eval Data (Aligned)
**File**: `artifacts/lvm/eval_v2_payload_aligned.npz`
- ✅ Targets replaced with payload vectors
- ✅ Provenance:
  ```json
  {
    "embedder_id": "GTR-T5-base-768",
    "payload_build_id": "payload584k_2025-10-24@sha256:12cfd8d7",
    "norm": "l2_once",
    "metric": "ip",
    "aligned_at": "2025-10-26T19:49:10.592753"
  }
  ```

---

## Path Forward

### Option A: Continue with Current Models ❌
**NOT RECOMMENDED** - Models fundamentally incompatible with retrieval index

### Option B: Retrain with Payload-Aligned Data ✅
**RECOMMENDED** - Rebuild training data using actual payload vectors

**Steps**:
1. Extract training sequences from payload (context windows from Wikipedia chunks)
2. Ensure targets are actual payload vectors
3. Preserve truth_keys for traceability
4. Add provenance metadata
5. Retrain all 5 Mamba models
6. Re-evaluate with aligned eval data

**Tool Needed**: `tools/build_payload_aligned_training.py`
- Input: `wikipedia_584k_payload.npy`
- Output: `training_sequences_payload_aligned.npz`
- Structure:
  ```python
  {
    'context_sequences': [N, 5, 768],
    'target_vectors': [N, 768],        # FROM PAYLOAD!
    'truth_keys': [N, 2],              # (article_idx, chunk_idx)
    'provenance': {...}
  }
  ```

**Expected Results After Retraining**:
- Validation cosine: 0.54-0.58 (same as before)
- **BUT**: Cosine to payload will also be 0.54-0.58 (aligned!)
- Contain@50: 60-75% (actual retrieval success)
- R@5: 40-55%

### Option C: Realign Payload to Training Data ❌
**NOT FEASIBLE** - Would require re-encoding all 584k Wikipedia chunks with unknown encoder

---

## Lessons Learned

1. **Always verify data provenance** - Training and inference data MUST come from same source
2. **Test end-to-end early** - Retrieval test should be smoke-tested before full training
3. **Use truth_keys everywhere** - Enables tracing data through pipeline
4. **Cosine to random baseline** - Quick sanity check: cos(train, payload_random) should be ~0
5. **Alignment !== Compatibility** - Aligning eval data revealed but didn't fix the model incompatibility

---

## Artifacts Created

### Diagnostic Tools
- `tools/diagnose_retrieval_gap.py` - Rank-of-truth analysis
- `tools/align_eval_to_payload.py` - Eval target alignment
- `tools/smoke_test_aligned_eval.py` - Quick validation test

### Data Files
- `artifacts/lvm/eval_v2_payload_aligned.npz` - Aligned eval data
- `artifacts/lvm/diagnosis_mamba_sandwich.json` - Diagnostic results
- `artifacts/lvm/smoke_test_aligned.json` - Smoke test results

### Documentation
- `docs/MAMBA_PHASE5_ROOT_CAUSE_ANALYSIS.md` (this file)

---

## Recommendation

**Proceed with Option B: Retrain with Payload-Aligned Training Data**

1. Create `tools/build_payload_aligned_training.py`
2. Generate `training_sequences_payload_aligned.npz` from payload
3. Retrain all 5 models (estimated 12-24 hours total)
4. Re-evaluate with aligned eval data
5. Expect Contain@50: 60-75%, R@5: 40-55%

**DO NOT** continue evaluation with current models - they are fundamentally incompatible with the retrieval index.

---

## References

- **PRD**: `docs/PRDs/PRD_5_Mamba_Models.md`
- **Training Data**: `artifacts/lvm/training_sequences_ctx5.npz`
- **Payload**: `artifacts/wikipedia_584k_payload.npy`
- **Aligned Eval**: `artifacts/lvm/eval_v2_payload_aligned.npz`
- **Consultant Feedback**: See conversation history (2025-10-26)
