# P8 "Constrained Mixture" Implementation Summary
**Date:** 2025-11-04 (Late Evening)
**Status:** ✅ Architecture complete, ready for pilot

---

## Executive Summary

P7 failed despite all defenses because:
1. **λ-blend instability**: Conflicting gradients between raw predictions and anchored output
2. **Epoch-3 collapse**: cos_anchor dropped 33% when teacher warmup ended
3. **InfoNCE spurious correlations**: Random in-batch negatives caused train/val divergence

**P8 fixes this with:**
1. **Constrained output space**: q = mixture of context vectors (orthogonal escape impossible)
2. **Listwise ranking**: Task-specific candidates only (no random in-batch)
3. **Order verifier**: Self-supervised temporal prior (learns "forward > backward")

**Split parity check**: ✅ Train/val distributions MATCH - rules out data issues

---

## P8 Architecture

### 1. Mixture-of-Context Head (NO Free Prediction)

```python
# Predict attention weights over context vectors
alpha = softmax(W_attn(encoder_output), dim=-1)  # (B, K)

# Output is ALWAYS a weighted mixture
q = normalize(Σ_i alpha_i · c_i)  # (B, 768)
```

**Key property:** q ∈ span(C) by construction
- Output MUST be a mix of context vectors (all from past)
- To predict forward, model must learn which context vectors are most relevant
- Orthogonal escape is geometrically impossible
- No λ-blend → no conflicting gradients

**Why this helps:**
- P7's λ-blend: q' = norm(0.8·q_raw + 0.2·c)
  - Backprop through non-linear re-normalization
  - Raw predictions (q_raw) can drift arbitrarily
  - Anchoring pulls back, but gradients fight each other
  - Collapsed at epoch 3 when teacher pull ended

- P8's mixture: q = norm(Σ alpha_i · c_i)
  - No free prediction to drift
  - Gradients flow directly through attention weights
  - Model learns "which past chunks predict forward"
  - Stable by design (no teacher pull needed)

### 2. Listwise Temporal Ranking (NO Global InfoNCE)

**Candidate set construction:**
```
candidates = [
    target_next,      # Index 0 (positive)
    target_prev,      # Index 1 (hard negative)
    hard_neg_1,       # Index 2 (same-article distractor)
    hard_neg_2,       # Index 3 (same-article distractor)
    ...               # Optional: limited in-batch negatives
]
```

**Loss:**
```python
scores = query @ candidates.T  # (B, L)
loss = -log_softmax(scores / T)[:, 0].mean()  # Rank index 0 highest
```

**Why this helps:**
- P7's InfoNCE: contrasted with ALL in-batch samples (63 negatives per batch)
  - Created spurious correlations based on batch composition
  - Model optimized batch-specific patterns that didn't generalize
  - Train margin positive, val margin negative

- P8's listwise: only contrasts with task-relevant candidates (4-8 total)
  - Focused on temporal ordering within articles
  - No batch-specific artifacts
  - Should generalize better

### 3. Order Verifier Auxiliary Head

```python
# Given (c_i, c_j) from same article, predict if j > i
v = [c_i, c_j, c_i * c_j, c_i - c_j]  # (B, 4*768)
logits = MLP(v)  # (B,)
loss_order = BCE(logits, labels)  # y=1 if j>i, y=0 if j<i
```

**Why this helps:**
- Self-supervised temporal prior
- Teaches model "forward > backward" as an invariant
- Cheap auxiliary task (tiny MLP, no extra data)
- Stabilizes training from first epoch

### 4. Combined Loss

```python
loss = (
    w_listwise * loss_listwise +      # Rank next highest (1.0)
    w_margin * loss_prev_repel +      # cos(q, next) > cos(q, prev) + 0.07 (0.5)
    w_order * loss_order_verifier     # Predict temporal order (0.2)
)
```

---

## Files Created

**Models:**
- `app/lvm/models_p8_constrained.py` (330 lines)
  - TransformerP8Constrained
  - LSTMP8Constrained
  - OrderVerifier

**Losses:**
- `app/lvm/losses_p8_listwise.py` (350 lines)
  - listwise_ranking_loss()
  - prev_repel_margin_loss()
  - order_verifier_loss()
  - combined_p8_loss()
  - create_candidate_set()
  - sample_order_pairs()

**Tools:**
- `tools/check_split_parity.py` (270 lines)
  - Verified train/val distributions match (✅ PASSED)

---

## Expected P8 Results

### Immediate (Epoch 1)

| Metric | P7 Baseline | P8 Expected | Why Different? |
|--------|-------------|-------------|----------------|
| cos_anchor | 0.579 (E1) → 0.391 (E3) | **≥ 0.95** (always) | q ∈ span(C) by construction |
| Margin (train) | +0.125 | **+0.15 to +0.25** | Focused listwise ranking |
| Margin (val) | **-0.059** ❌ | **+0.10 to +0.20** ✅ | No batch artifacts |
| cos_next | 0.391 (E1) → 0.241 (E3) | **≥ 0.35** | No drift (constrained) |
| R@5 (in-article) | Unknown | **≥ 75%** | Order verifier helps |

**Kill criteria:**
- If margin stays negative after 2 epochs → STOP
- If R@5 ≤ DIRECT -5pts after 2 epochs → STOP
- Then pivot to retrieval-only (no autoregressive LVM)

### After Full Training (10 epochs)

| Target | Threshold | Why This Matters |
|--------|-----------|------------------|
| **Margin (val)** | ≥ +0.20 | Proves forward learning |
| **R@5 (in-article)** | ≥ 80% | Beats DIRECT baseline |
| **MRR@10** | ≥ 0.65 | Retrieval quality |
| **cos_anchor** | ≥ 0.90 | No orthogonal drift |
| **Order accuracy** | ≥ 75% | Temporal prior learned |

---

## Next Steps (Immediate)

### Option 1: Quick Pilot (Recommended, ~30 min)

**Purpose:** Verify P8 architecture works before committing to full training

1. **Subset data** (5-10k sequences):
   ```bash
   ./.venv/bin/python tools/subset_sequences.py \
       --input artifacts/lvm/arxiv_train_sequences.npz \
       --output artifacts/lvm/arxiv_train_pilot_10k.npz \
       --n-samples 10000
   ```

2. **Run P8 pilot** (1-2 epochs, ~15-30 min CPU):
   ```bash
   # TODO: Create train_p8_pilot.py
   ./.venv/bin/python app/lvm/train_p8_pilot.py \
       --train artifacts/lvm/arxiv_train_pilot_10k.npz \
       --val artifacts/lvm/arxiv_val_sequences.npz \
       --epochs 2 \
       --batch-size 32 \
       --device cpu
   ```

3. **Watch for:**
   - ✅ cos_anchor ≥ 0.95 (should be trivial)
   - ✅ Margin stays positive (both train and val!)
   - ✅ No epoch-3 collapse (no teacher warmup needed)
   - ❌ If margin negative after E2 → abort P8, pivot to retrieval-only

### Option 2: Full P8 Training (~10 hrs CPU)

**If pilot passes:**

```bash
# TODO: Create train_p8_constrained.py (similar to train_p7_ranker.py)
./.venv/bin/python app/lvm/train_p8_constrained.py \
    --train artifacts/lvm/arxiv_train_sequences.npz \
    --val artifacts/lvm/arxiv_val_sequences.npz \
    --epochs 10 \
    --batch-size 64 \
    --w-listwise 1.0 \
    --w-margin 0.5 \
    --w-order 0.2 \
    --margin 0.07 \
    --temperature 0.07 \
    --device cpu \
    --exp-name p8_constrained_baseline
```

**Monitor:**
- Margin (should be positive from epoch 1)
- cos_anchor (should stay ≥ 0.90)
- Order accuracy (should reach ~75%)
- R@5 on validation set (target ≥ 80%)

---

## Why P8 Should Work (Technical)

### 1. Constrained Optimization is Stable

**P7 problem:**
```
Optimize: q_raw ∈ R^768 (unconstrained)
Apply: q' = norm(λ·q_raw + (1-λ)·c)
Backprop: ∂L/∂q_raw flows through non-linear norm()
Result: Conflicting gradients (raw vs anchored)
```

**P8 solution:**
```
Optimize: alpha ∈ R^K (attention weights)
Apply: q = norm(Σ alpha_i · c_i)
Backprop: ∂L/∂alpha flows through linear mixture
Result: Clean gradients, stable optimization
```

### 2. Listwise Ranking Avoids Batch Artifacts

**P7 problem:**
- InfoNCE: max_{neg ∈ batch} sim(q, neg)
- Batch composition affects loss
- Model learns batch-specific patterns
- Doesn't generalize to validation

**P8 solution:**
- Listwise: rank(target_next | candidates)
- Candidates are task-specific (prev, same-article)
- Batch composition doesn't affect ranking
- Generalizes better

### 3. Order Verifier Provides Strong Prior

**P7 problem:**
- No explicit temporal prior
- Model must discover "forward > backward" from weak Δ signal
- Takes many epochs, may not converge

**P8 solution:**
- Order verifier: direct supervision for "j > i"
- Learned from epoch 1
- Strong, stable signal
- Helps main model learn temporal direction

---

## Fallback Plan (If P8 Fails)

**If P8 pilot fails (margin negative after 2 epochs):**

1. **Abandon autoregressive LVM entirely**
   - After P1-P7 all failing, plus P8 failing → signal is too weak
   - Vector-space autoregression may be fundamentally flawed for semantics

2. **Pivot to retrieval-only LVM**
   - Train Q-tower with pairwise order verifier + listwise ranker
   - Use for retrieval/ranking only (not generation)
   - Keep vec2text for decoding retrieved chunks
   - Simpler, more reliable, still useful

3. **Alternative: Abandon vecRAG, use textRAG**
   - Text-based retrieval is more reliable
   - No vector quantization artifacts
   - Easier to debug and understand

---

## Implementation TODO (If Proceeding)

### Immediate (for pilot):

1. ✅ **Models** - `models_p8_constrained.py` (DONE)
2. ✅ **Losses** - `losses_p8_listwise.py` (DONE)
3. ✅ **Split check** - `check_split_parity.py` (DONE, PASSED)
4. ⏳ **Dataset** - Need to add candidate set construction
5. ⏳ **Training loop** - Create `train_p8_pilot.py`
6. ⏳ **Validation** - Add R@K metrics

### For full training:

1. Create `train_p8_constrained.py` (full training loop)
2. Add evaluation script with R@K/MRR metrics
3. Create training script `scripts/train_p8_constrained.sh`
4. Document kill criteria and fallback plans

---

## Key Learnings from P7 Failure

1. **Architectural complexity ≠ stability**
   - P7 had 5 defense mechanisms, all failed
   - Root cause: unstable λ-blend + InfoNCE artifacts

2. **Train/val metrics can diverge for subtle reasons**
   - Data distributions matched (verified)
   - But training dynamics differed (batch artifacts)
   - Need simpler, more focused objectives

3. **Forward temporal signal is WEAK (+6.3%)**
   - Can't rely on data bias alone
   - Need strong architectural constraints
   - Need explicit temporal priors (order verifier)

4. **Geometric constraints > regularization**
   - Constraining output space (q ∈ span(C)) is more stable
   - Than regularizing free predictions (λ-blend)

---

## Summary

P8 represents a fundamental architectural pivot:
- **P7**: Free prediction + λ-blend anchor + InfoNCE contrast
- **P8**: Constrained mixture + listwise ranking + order verifier

**Key hypothesis:** Constraining the output space removes the instability that caused P7's epoch-3 collapse and train/val divergence.

**Test plan:** Run quick pilot (1-2 epochs, 10k samples) to verify:
1. cos_anchor ≥ 0.95 (trivial due to constraint)
2. Margin positive on both train AND val
3. No collapse when auxiliary losses change

**If pilot passes:** Proceed to full training (10 epochs, 87k samples)
**If pilot fails:** Pivot to retrieval-only LVM or abandon vecRAG

---

**Status:** Ready for pilot. Need to implement training loop and dataset.
