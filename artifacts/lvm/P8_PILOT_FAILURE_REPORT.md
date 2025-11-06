# P8 Pilot Training - Failure Report

**Date**: 2025-11-04 (Late Evening)
**Status**: ❌ **FAILED** - Margin negative after epoch 2
**Duration**: ~2 minutes (pilot with 10k sequences)

---

## Executive Summary

**P8 "Constrained Mixture" architecture successfully eliminated orthogonal escape (cos_anchor = 0.97), but STILL learned backward prediction (margin = -0.021).**

This is the **8th consecutive failure** (P1→P8) and provides **decisive evidence** that the problem is NOT architectural, but rooted in the **vector space geometry itself**.

---

## Training Configuration

### Data
- **Training**: 10,000 sequences (pilot subset from 97k arXiv sequences)
- **Validation**: 2,000 sequences (stratified random sample)
- **Source**: arXiv papers (87k train + 11k val combined, then resampled)

### Model
- **Architecture**: TransformerP8Constrained (constrained mixture head)
- **Key innovation**: `q = normalize(Σ_i alpha_i · c_i)` - output constrained to span(C)
- **No free prediction**: Cannot escape to arbitrary orthogonal space

### Loss Function
- **Listwise ranking**: Cross-entropy over [next, prev, 30 random negatives]
- **Prev-repel margin**: `max(0, margin - cos(pred,next) + cos(pred,prev))`
- **Margin threshold**: 0.07
- **Temperature**: 0.07
- **w_prev**: 1.0

### Hyperparameters
- **Epochs**: 2 (aborted due to negative margin)
- **Batch size**: 128
- **Learning rate**: 1e-3 (AdamW)
- **Device**: CPU (macOS with KMP_DUPLICATE_LIB_OK=TRUE)

---

## Results

### Epoch-by-Epoch Metrics

| Epoch | Loss | Margin | cos_next | cos_prev | cos_anchor | R@5 |
|-------|------|--------|----------|----------|------------|-----|
| 1 | 2.186 | **-0.023** | 0.591 | 0.614 | 0.969 | 0.458 |
| 2 | 2.147 | **-0.021** | 0.593 | 0.614 | 0.974 | 0.454 |

### Pass/Fail Gate Analysis

| Gate | Threshold | Result | Status | Notes |
|------|-----------|--------|--------|-------|
| **Margin > 0** | Required | **-0.021** | ❌ **FAIL** | Model predicts backward |
| **cos_anchor ≥ 0.95** | Required | **0.974** | ✅ **PASS** | Perfect constraint to span(C) |
| **R@5 trending up** | Desired | 0.458 → 0.454 | ❌ **FAIL** | Slight decline |
| **cos_prev < cos_next** | Required | 0.614 > 0.593 | ❌ **FAIL** | Backward prediction |

**Decision**: Abort pilot after epoch 2 (margin negative ≥ 2 epochs).

---

## Critical Findings

### 1. Architectural Hypothesis VALIDATED ✅

**Question**: Can constraining output to span(context) prevent orthogonal escape?

**Answer**: YES
- `cos_anchor = 0.974` (near-perfect alignment with context subspace)
- No collapse (stable across both epochs)
- Geometric constraint works as designed

**Comparison to P7**:
- P7 (free prediction): cos_anchor dropped to 0.39 at epoch 3 (orthogonal escape)
- P8 (constrained): cos_anchor stable at 0.97 (no escape possible)

### 2. Training Hypothesis FALSIFIED ❌

**Question**: Can we train forward prediction with architectural constraints + explicit losses?

**Answer**: NO
- Despite perfect geometric constraint: **margin = -0.021** (negative!)
- Despite explicit prev-repel loss: **cos_prev (0.614) > cos_next (0.593)**
- Despite listwise ranking: No improvement in R@5

**After 8 attempts** (P1→P8), pattern is clear:
- P1: Baseline MSE → margin -0.167
- P2-P4: Directional losses → collapsed or negative
- P5.1: Curriculum learning → margin -0.046
- P6: NEXT token → margin -0.082
- P6b v2.1: 6-layer defense → margin -0.047
- P6b v2.2: Stronger pressure → orthogonal escape
- P7: Ranker + InfoNCE → margin -0.067 (val)
- **P8: Constrained mixture → margin -0.021** ❌

### 3. Root Cause Identified 🔬

**The problem is NOT architectural - it's in the VECTOR SPACE GEOMETRY itself.**

Evidence:
1. **Perfect constraint doesn't help**: cos_anchor = 0.97, but margin still negative
2. **Explicit losses don't help**: Prev-repel margin loss with w=1.0, but cos_prev > cos_next
3. **Consistent across data**: Wikipedia (Δ=-0.069), arXiv (Δ=-0.021 observed)
4. **Stable but wrong**: No collapse, no instability, just learns backward

**Hypothesis**: GTR-T5 embeddings capture **explanatory structure** better than **narrative flow**:
- Later chunks reference earlier concepts ("as mentioned earlier", "the Einstein...")
- Embeddings encode semantic similarity, not temporal directionality
- Vector geometry naturally points backward (toward established concepts) more than forward (toward novel concepts)

---

## Detailed Analysis

### Why P8 Should Have Worked (But Didn't)

**P8 design rationale**:
1. **Constrained output** → q ∈ span(C) by construction → orthogonal escape impossible
2. **No λ-blend** → no conflicting gradients → no epoch-3 collapse (P7's failure mode)
3. **Listwise ranking** → task-specific candidates → no batch artifacts (InfoNCE's failure mode)
4. **Explicit prev-repel** → direct backward penalty → should enforce forward directionality

**What actually happened**:
1. ✅ Constraint worked perfectly (cos_anchor = 0.97)
2. ✅ No collapse (stable loss, metrics)
3. ✅ Listwise ranking executed correctly
4. ❌ **Prev-repel loss FAILED to flip margin positive**

**Why it failed**:
- The **vector geometry itself** has backward bias
- Context vectors (c_0, ..., c_4) naturally align more with `target_prev` than `target_next`
- No amount of architectural constraint can overcome this geometric property
- Mixture weights (alpha_i) learn to emphasize backward-pointing components

### Comparison: Free vs. Constrained Prediction

| Metric | P7 (Free) | P8 (Constrained) | Interpretation |
|--------|-----------|------------------|----------------|
| cos_anchor | 0.391 (E3) | **0.974** (E2) | P8 eliminates escape ✅ |
| cos_next | 0.271 | **0.593** | P8 has better target alignment ✅ |
| cos_prev | 0.338 | **0.614** | P8 STILL predicts backward ❌ |
| Margin | -0.067 (val) | **-0.021** | P8 improves but stays negative ❌ |

**Key insight**: Constraining to span(C) **improves absolute similarity** but **doesn't flip directional preference**.

---

## Implications

### 1. Autoregressive Vector LVM May Be Fundamentally Flawed

**After 8 failed attempts with diverse approaches**, the evidence suggests:
- Vector embeddings (GTR-T5, likely others) **do not encode strong forward temporal signal**
- Explanatory text structure (Wikipedia, arXiv) has **inherent backward bias**
- MSE/cosine loss **cannot overcome geometric bias** (even with strong regularization)

**Possible root causes**:
1. **Embedding models trained on masked language modeling** (bidirectional context, no temporal bias)
2. **Scientific/encyclopedic text structure** (later chunks reference earlier concepts more than preview)
3. **Semantic similarity ≠ temporal causality** (embeddings capture "related to", not "follows from")

### 2. Decision Tree

After P8 failure, we have **three strategic options**:

**Option A: Abandon Autoregressive LVM** ⚠️ (RECOMMENDED)
- Accept vector-to-vector next-chunk prediction is fundamentally limited
- **Pivot to retrieval-only vecRAG**:
  - Use existing FAISS + reranking (already works: 73.4% Contain@50, 50.2% R@5)
  - Focus on improving retrieval quality (better embeddings, graph-aware reranking)
  - No generative component needed for RAG pipelines
- **Pros**: Stop wasting time on doomed approach, leverage what already works
- **Cons**: Abandon 2 months of LVM work, no "emergent reasoning" potential

**Option B: Test Truly Narrative Data** 🔬 (QUICK VALIDATION)
- **Hypothesis**: Problem is data-specific (Wikipedia, arXiv both explanatory)
- **Test**: Sample narrative data (stories, tutorials, code walkthroughs)
- **Quick check**: Compute Δ = cos(ctx[-1], next) - cos(ctx[-1], prev) on 50-100 sequences
- **Time**: 15 minutes to download + encode + analyze
- **Decision gate**: If Δ < +0.10, abandon LVM; if Δ > +0.15, retry P8 on narrative data

**Option C: Pivot to Bi-Directional or Retrieval-Augmented** 🔄
- **Accept backward bias as feature, not bug**:
  - Train model to predict "most relevant related chunk" (not specifically next)
  - Use both forward and backward signals
  - Rank candidates instead of generate
- **Architecture**: Bi-LSTM or Transformer with both directions
- **Loss**: Contrastive ranking (positive = any related chunk, negative = unrelated)
- **Pros**: Leverage signal that exists (backward + forward)
- **Cons**: Not autoregressive, different use case (more like retrieval)

---

## Recommendation

**Immediate action**: **Option B** (Test narrative data) - 15 min to validate/falsify

**If Option B fails** (Δ < +0.10 on narrative data):
→ **Option A** (Abandon autoregressive LVM, pivot to retrieval-only)

**If Option B succeeds** (Δ > +0.15 on narrative data):
→ Retry P8 on narrative data (full training run)

**Do NOT**:
- ❌ Try P9 with more architectural tricks (8 attempts proved architecture not the issue)
- ❌ Train on Wikipedia/arXiv again (proven backward bias)
- ❌ Increase margin loss weight to extreme values (doesn't fix geometry)

---

## Session Summary

**What we proved**:
1. ✅ Constrained mixture head works (cos_anchor = 0.97)
2. ✅ Listwise ranking executes correctly
3. ❌ **Vector geometry has backward bias that architecture cannot overcome**

**What we learned**:
- After 8 attempts, problem is **data/embedding geometry**, not model architecture
- GTR-T5 embeddings may not be suitable for autoregressive next-chunk prediction
- Need to either: (1) find better data, (2) abandon approach, or (3) pivot to different task

**Files created**:
- `tools/subset_sequences.py` (270 lines) - Stratified NPZ subset creation
- `app/lvm/train_p8_pilot.py` (160 lines) - P8 pilot training script
- `artifacts/lvm/pilot_train_10k.npz` (10k sequences)
- `artifacts/lvm/pilot_val_2k.npz` (2k sequences)
- This report: `artifacts/lvm/P8_PILOT_FAILURE_REPORT.md`

**Next steps**: User decision on Option A/B/C above.

---

## Technical Details

### Model Architecture

```python
# P8 Constrained Mixture Head (from app/lvm/models_p8_constrained.py)
class TransformerP8Constrained:
    def forward(self, C):  # C: (B, 5, 768)
        # Encode context
        h = transformer_encoder(C.flatten(1, 2))  # (B, 2560)

        # Predict mixture weights
        alpha = softmax(W_attn @ h)  # (B, 5)

        # Constrained mixture
        q = normalize(sum_i alpha_i * c_i)  # q ∈ span(C) by construction

        return q, alpha
```

**Key property**: `q = Σ_i alpha_i · c_i` where `sum(alpha) = 1` and `alpha_i ≥ 0`
→ `q` is **convex combination** of context vectors
→ **Cannot escape to orthogonal space** (unlike P7's free prediction)

### Loss Function

```python
# Listwise ranking + prev-repel (from app/lvm/losses_p8_listwise.py)
def listwise_loss_with_prev_margin(q, candidates, margin=0.07, w_prev=1.0):
    # candidates: (B, 32, 768) where [0]=next, [1]=prev, [2:32]=random negatives

    scores = cosine_similarity(q, candidates)  # (B, 32)

    # Listwise ranking (InfoNCE-style)
    loss_rank = cross_entropy(scores / temperature, target=0)

    # Prev-repel margin
    loss_margin = max(0, margin - scores[:, 0] + scores[:, 1])

    return loss_rank + w_prev * loss_margin
```

**Expected behavior**: Model should learn `cos(q, next) > cos(q, prev) + 0.07`

**Actual behavior**: Model learned `cos(q, prev) = 0.614 > cos(q, next) = 0.593`

### Why Constraint Alone Isn't Enough

Even though `q ∈ span(c_0, ..., c_4)`, the model can still prefer backward:

**Example**: If context vectors all point backward (toward prev), then:
```
c_0 · prev = 0.50, c_0 · next = 0.35
c_1 · prev = 0.55, c_1 · next = 0.38
c_2 · prev = 0.58, c_2 · next = 0.40
c_3 · prev = 0.60, c_3 · next = 0.42
c_4 · prev = 0.65, c_4 · next = 0.45

Any mixture q = Σ alpha_i · c_i will satisfy:
  q · prev > q · next  (because ALL components prefer prev)
```

**This is exactly what we observed**: cos_anchor = 0.97 (good alignment with C), but cos_prev > cos_next.

---

## Appendix: All 8 Failed Approaches

| Approach | Key Innovation | Margin Result | Failure Mode |
|----------|---------------|---------------|--------------|
| P1 | Baseline MSE | -0.167 | Follows dominant (backward) signal |
| P2-P4 | Directional margin loss | Negative or collapsed | λ too weak or unstable |
| P5.1 | Curriculum learning | -0.046 | Landscape reshaping insufficient |
| P6 | NEXT token (remove identity) | -0.082 | Proved data has backward bias |
| P6b v2.1 | 6-layer defense (ρ-controller) | -0.047 | Guardrails too conservative |
| P6b v2.2 | Stronger pressure (ρ=0.35) | Orthogonal escape | Directional loss overwhelmed MSE |
| P7 | Ranker + InfoNCE | -0.067 (val) | λ-blend instability, batch artifacts |
| **P8** | **Constrained mixture** | **-0.021** | **Vector geometry backward bias** |

**Common thread**: All approaches failed to flip margin positive, regardless of architecture or loss design.

**Conclusion**: Problem is **data/embedding space**, not model design.

---

*Report generated: 2025-11-04 23:45 PST*
*Session: P8 Pivot → Decisive Failure*
