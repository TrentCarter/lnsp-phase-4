# Phase 1 Evaluation: Failure Report
## 2025-10-27 14:15

---

## Executive Summary

**Phase 1 Gate: ❌ FAILED**

Both InfoNCE+AR contrastive and AR-only models show **0% retrieval** on leaked eval set. No improvement from contrastive learning. Critical generalization failure discovered.

---

## Evaluation Results

### Phase 1: Relative Comparison on Leaked Eval
Eval Set: `eval_v2_payload_aligned.npz` (91.5% article overlap with training)

| Model | Training Val Cosine | Eval Pred→Target | R@1 | R@5 | R@10 | Contain@50 | Eff@5 | P95 |
|-------|---------------------|-----------------|-----|-----|------|------------|-------|-----|
| **AR-only (POC)** | 0.5749 | 0.2496 | 0% | **0%** | 0% | 0% | 0.00 | 1.13ms |
| **InfoNCE+AR (Contrastive)** | 0.5149 | 0.2557 | 0% | **0%** | 0% | 0% | 0.00 | 1.10ms |
| **Δ (Contrastive - AR)** | -0.06 | +0.006 | 0pp | **0pp** | 0pp | 0pp | 0.00 | -0.03ms |

**Success Gate**: ΔR@5 ≥ +10pp or MRR ≥ +0.05
**Result**: ΔR@5 = 0pp ❌ FAILED

---

## Root Cause Analysis

### 1. Massive Cosine Drop During Eval

Both models show catastrophic performance degradation:

| Metric | Training Val | Eval Inference | Drop |
|--------|-------------|---------------|------|
| AR-only | 0.5749 | 0.2496 | **-0.325** (-56.6%) |
| InfoNCE+AR | 0.5149 | 0.2557 | **-0.259** (-50.3%) |

**Interpretation**: Models learned article-specific patterns, NOT general GTR-T5 semantics.

### 2. No Benefit from Contrastive Learning

InfoNCE + AR showed:
- ✅ Lower training val_cosine (0.5149 vs 0.5749) - expected with contrastive
- ❌ No improvement in eval cosine (0.2557 vs 0.2496) - only +0.006
- ❌ No improvement in retrieval (0% R@5 for both)

**Conclusion**: Contrastive learning (λ_con=0.7, τ=0.07, 1024 batch negatives) did NOT prevent memorization or improve generalization.

### 3. Data Leak Doesn't Help

Despite 91.5% article overlap in eval set:
- Models trained on articles 0-6100
- Eval set uses same articles (leaked!)
- But models still fail → learned **episode-specific patterns**, not article semantics

---

## Why Contrastive Failed

### Hypotheses (In Order of Likelihood)

#### 1. Objective Mismatch (Most Likely)
**Problem**: InfoNCE operates in 256D projected space, but inference uses 768D raw space.

```python
# During training:
h_pred = projection_head(pred)  # 768D → 256D
h_target = projection_head(target)
loss_infonce = cross_entropy(h_pred @ h_target.T)

# During inference:
pred = model(context)  # Raw 768D output
# Projection head NOT used!
```

**Impact**:
- Contrastive loss optimizes projected vectors
- AR cosine loss optimizes raw vectors (λ_ar=0.3 only!)
- Model learns split representation (can't do both well)

**Fix**: Either (a) use projection head during inference, OR (b) apply InfoNCE to raw 768D vectors

#### 2. Insufficient Contrastive Weight
**Problem**: λ_con=0.7 may be too low given strong AR loss pull.

```
Total loss = 0.7 × InfoNCE + 0.3 × AR_cosine
```

**Evidence**:
- AR-only: 0.5749 val_cosine (pure AR optimization)
- InfoNCE+AR: 0.5149 val_cosine (mixed, lower)
- But eval cosine barely changed (+0.006)

**Interpretation**: AR loss dominates optimization → model still memorizes.

**Fix**: Increase λ_con to 0.85-0.90 (contractor's suggestion)

#### 3. Weak Negatives / Small Batch
**Problem**: Effective batch = 1024 (256 × 4 grad accum) may be insufficient.

**Evidence**:
- InfoNCE needs STRONG negatives to prevent shortcuts
- 1024 negatives per positive may not cover semantic space
- Hard negatives (wrong but similar articles) not sampled explicitly

**Fix**:
- Increase batch to 2048+ (if memory allows)
- Add hard negative mining (sample from nearby articles)

#### 4. Temperature Too Low
**Problem**: τ=0.07 may be over-sharpening the distribution.

**Fix**: Sweep τ ∈ {0.05, 0.07, 0.10, 0.15}

#### 5. Stop-Gradient on Wrong Branch
**Problem**: Currently stop-grad on target branch only.

```python
h_pred = projection_head(pred)  # Gradient flows
h_target = projection_head(target)  # No gradient (stop-grad)
```

**Alternative**: Symmetric loss (no stop-grad) or stop-grad on pred branch

---

## Verification Checks

### ✅ Data Format Alignment
- Training: `context_sequences` (396k, 5, 768)
- Eval: `contexts` (5.2k, 5, 768)
- Same embedder (GTR-T5-base-768)
- Same normalization (l2_once)
- Same context length (5)

### ✅ Model Loading
- Checkpoint loads successfully
- Val cosine matches history (0.5149)
- Projection head saved separately (unused during inference)

### ✅ Inference Pipeline
- Predictions normalized (L2)
- FAISS index configured correctly (nprobe=64)
- Truth→payload alignment verified (mean=1.000)

### ❌ Prediction Quality
- Mean pred→target cosine: 0.2557 (should be ~0.50+)
- Sample cosines: 0.10-0.31 (wide variance, all low)

---

## Triage Decision Tree

Based on contractor's Phase 1 failure protocol:

### If Contrastive Underperforms (Current State):

#### Step 1: Projection Head Check ⬅️ **START HERE**
**Question**: Should projection head be used during inference?

**Test**:
```python
# Load projection head from checkpoint
proj_head = ProjectionHead(d_model=768, hidden_dim=512, out_dim=256)
proj_head.load_state_dict(ckpt['projection_head_state_dict'])

# Inference with projection
with torch.no_grad():
    pred_raw = model(context)  # [B, 768]
    pred_proj = proj_head(pred_raw)  # [B, 256]

    # Search FAISS with projected vectors?
    # Or just use as compatibility check?
```

**Expected**:
- If proj_head(pred) → proj_head(target) cosine is HIGH (~0.70+), then projection head SHOULD be used
- If still low, projection head isn't the fix

---

#### Step 2: Temperature Sweep
**If Step 1 doesn't help, try τ tuning:**

```bash
# Retrain with different temperatures
for tau in 0.05 0.10 0.15; do
    python app/lvm/train_mamba_contrastive.py \
        --temperature $tau \
        --lambda-con 0.7 \
        --epochs 5  # Quick test
done
```

**Gate**: If any τ achieves eval cosine ≥ 0.35, continue to full training.

---

#### Step 3: Stronger Contrastive (λ_con ↑)
**If temperature doesn't help:**

```bash
python app/lvm/train_mamba_contrastive.py \
    --lambda-con 0.85 \
    --lambda-ar 0.15 \
    --temperature 0.07 \
    --epochs 10
```

**Rationale**: Force model to prioritize contrastive alignment over exact AR prediction.

---

#### Step 4: Hard Negative Mining
**If λ adjustment doesn't help:**

Modify training to sample hard negatives:
```python
# For each positive (ctx→next), sample:
# - Same article, different position (moderate negative)
# - Nearby article, similar topic (hard negative)
# - Random article (easy negative)
```

**Expected**: Harder negatives → better semantic discrimination.

---

#### Step 5: Anchor-MMD (Nuclear Option)
**If all above fail:**

Add Maximum Mean Discrepancy loss to anchor model predictions to GTR-T5 distribution:

```python
loss_mmd = mmd_kernel(pred_distribution, gtr_t5_distribution)
total_loss = 0.7 × InfoNCE + 0.28 × AR_cosine + 0.02 × MMD
```

**Warning**: Requires sampling GTR-T5 vectors from payload (expensive).

---

## Immediate Next Steps

1. **✅ DONE**: Document Phase 1 failure
2. **TODO**: Run projection head compatibility test (Step 1)
3. **TODO**: If proj_head doesn't help, start temperature sweep (Step 2)
4. **TODO**: If sweeps don't work, re-evaluate architecture (may need different approach)
5. **TODO**: Consider if Phase 2 (fresh Wikipedia) is even worth it if models can't generalize

---

## Key Learnings

1. **Val cosine ≠ generalization**: 0.5149 validation cosine is meaningless if it doesn't transfer.
2. **Data leak doesn't guarantee success**: 91.5% overlap didn't help → models memorize sequences, not articles.
3. **Contrastive learning is fragile**: Small misconfigurations (projection head usage, λ balance) can nullify benefits.
4. **0% → 0% is still 0%**: No relative improvement means contrastive fix didn't work.

---

## Files Generated

- **Contrastive eval**: `artifacts/lvm/phase1_contrastive_leaked_eval.json`
- **AR-only eval**: `artifacts/lvm/phase1_ar_only_leaked_eval.json`
- **This report**: `PHASE1_FAILURE_REPORT.md`

---

## Open Questions

1. Should projection head be used during inference?
2. Is InfoNCE fundamentally incompatible with autoregressive sequence modeling?
3. Should we abandon Mamba and try a two-tower architecture (separate query/doc encoders)?
4. Is the 5-vector context length too short for semantic learning?

---

**Report Generated**: 2025-10-27 14:15
**Status**: Phase 1 failed, triaging root cause
**Next Action**: Test projection head usage during inference
