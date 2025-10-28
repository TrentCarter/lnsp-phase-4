# Phase 1 Evaluation: Results & Diagnosis
## 2025-10-27 14:30

---

## TL;DR

**Phase 1 Gate: ❌ FAILED**

- Both models show 0% R@5 (no improvement from contrastive)
- Root cause: **Projection head anti-correlated with targets**
- Projected vectors: **-0.12 cosine** (negative!)
- Diagnosis: InfoNCE (70%) and AR cosine (30%) learned conflicting representations

---

## Phase 1 Results

### Retrieval Metrics (Leaked Eval, 91.5% Overlap)

| Model | R@1 | R@5 | R@10 | Contain@50 | P95 |
|-------|-----|-----|------|------------|-----|
| AR-only | 0% | **0%** | 0% | 0% | 1.13ms |
| InfoNCE+AR | 0% | **0%** | 0% | 0% | 1.10ms |
| **Δ** | 0pp | **0pp** | 0pp | 0pp | -0.03ms |

**Gate**: ΔR@5 ≥ +10pp → **Got 0pp** ❌

### Cosine Similarity

| Model | Training Val | Eval Pred→Target | Drop |
|-------|-------------|-----------------|------|
| AR-only | 0.5749 | 0.2496 | -56.6% |
| InfoNCE+AR | 0.5149 | 0.2557 | -50.3% |

---

## Root Cause: Split Objective Conflict

### The Problem

Contrastive training used **two different objectives**:

```python
# 70% of loss: InfoNCE on projected vectors (256D)
h_pred = projection_head(pred)      # 768D → 256D
h_target = projection_head(target)
loss_infonce = cross_entropy(h_pred @ h_target.T)

# 30% of loss: AR cosine on raw vectors (768D)
loss_ar = 1.0 - cosine_similarity(pred, target)

# Combined
total_loss = 0.7 × loss_infonce + 0.3 × loss_ar
```

### Projection Head Compatibility Test

Tested if projection head improves alignment:

| Space | Pred→Target Cosine | Interpretation |
|-------|-------------------|----------------|
| **Raw 768D** | 0.2163 | Poor but positive |
| **Projected 256D** | **-0.1195** | **ANTI-CORRELATED!** |
| **Change** | **-0.3357** | Projection actively harms |

**Conclusion**: Projection head learned to optimize InfoNCE in a way that's **incompatible** with raw vector predictions.

---

## Why Contrastive Failed

### Hypothesis: Objective Mismatch

1. **InfoNCE (70%)** optimizes projected 256D space
   - Forces model to distinguish between 1024 in-batch negatives
   - Projection head learns to amplify semantic differences
   - But operates in compressed 256D space

2. **AR cosine (30%)** optimizes raw 768D space
   - Forces model to match exact GTR-T5 vectors
   - No compression, direct regression
   - Operates in full 768D space

3. **Conflict**: Model can't satisfy both objectives!
   - If raw pred matches target (AR loss low), projection may push them apart (InfoNCE loss high)
   - If projected vectors are well-separated (InfoNCE loss low), raw vectors may be misaligned (AR loss high)

### Evidence

Training history shows conflicting signals:
- `train_ar_cosine`: 0.48-0.50 (AR objective struggling)
- `train_infonce`: 0.60-1.03 (InfoNCE improving but not converging)
- `val_cosine`: 0.5149 (raw space, mediocre)

During eval:
- Raw 768D: 0.2163 (poor but expected from AR loss)
- Projected 256D: -0.1195 (catastrophic, InfoNCE learned wrong thing)

---

## What This Means

### 1. Contrastive Learning Didn't Work As Intended

InfoNCE was supposed to:
- ✅ Prevent episode-specific memorization
- ✅ Learn global GTR-T5 semantic space
- ✅ Improve generalization

But actually:
- ❌ Created conflicting representations
- ❌ Projection head learned anti-correlated space
- ❌ No improvement over AR-only

### 2. The Fix Wasn't the Right Fix

Contractor's hypothesis:
> "AR-only learns episode patterns. InfoNCE forces global semantics."

Reality:
- AR-only: 0% R@5 (correct diagnosis)
- InfoNCE+AR: 0% R@5 (fix didn't work)
- Both: Massive eval degradation (both memorized, different ways)

### 3. Fundamental Architecture Issue

The problem may not be solvable with **autoregressive LVMs**:
- Autoregression: Predict next vector from context sequence
- Requires: Temporal/causal modeling (X → Y → Z)
- But we want: Semantic similarity (X ≈ Y if similar meaning)

These objectives are **orthogonal**!

---

## Triage Options

### Option 1: Fix Split Objective (Try Once)

**Approach**: Apply InfoNCE to raw 768D vectors (no projection head)

```python
# No projection head!
loss_infonce = cross_entropy(pred @ target.T / temperature)
loss_ar = 1.0 - cosine_similarity(pred, target)
total_loss = 0.85 × loss_infonce + 0.15 × loss_ar
```

**Rationale**: Single representation space, no conflict.

**Expected**: If InfoNCE can work in 768D, should see improvement.

**Risk**: 768D InfoNCE may not have enough capacity to distinguish negatives.

**Time**: 1 day (5-hour training + eval)

---

### Option 2: Temperature Sweep (Quick Test)

**Approach**: Retry contrastive with τ ∈ {0.05, 0.10, 0.15}

**Rationale**: Current τ=0.07 may be too aggressive.

**Expected**: Higher τ → softer distribution → less conflict?

**Time**: 6 hours (3 × 2-hour training)

---

### Option 3: Pure InfoNCE (No AR Loss)

**Approach**: Train with InfoNCE only (λ_con=1.0, λ_ar=0.0)

**Rationale**: Eliminate conflict entirely.

**Expected**: Model learns pure similarity, no exact prediction.

**Risk**: May not work for sequence autoregression.

**Time**: 1 day

---

### Option 4: Abandon Contrastive, Try Different Architecture

**Approach**: Two-tower architecture (separate query/doc encoders)

**Rationale**: Autoregressive LVMs may be fundamentally wrong for this task.

**Alternatives**:
- Dual encoder (BERT-style)
- Siamese network (shared encoder, separate heads)
- Retrieve-then-rerank (LVM for reranking, not initial retrieval)

**Time**: 3-5 days (new architecture)

---

### Option 5: Abandon LVM, Use AMN Baseline

**Approach**: AMN (Attention Memory Network) already works well.

**Evidence**: AMN showed strong retrieval on payload-aligned eval (before leak discovery).

**Trade-off**: Larger model, but proven to work.

**Time**: Immediate (already trained)

---

## Recommendation

### For Contractor:

**Short-term (24 hours)**:
1. Try Option 1 (InfoNCE on raw 768D, no projection)
   - If works → proceed to Phase 2 (fresh Wikipedia)
   - If fails → consider architecture pivot

**Medium-term (3-5 days)**:
2. If Option 1 fails, evaluate Option 4 (two-tower)
   - More standard for retrieval tasks
   - Proven in literature (BERT, DPR, Sentence-BERT)

**Long-term**:
3. Consider if autoregressive LVM is right tool for semantic retrieval
   - LVMs excel at: Sequence prediction, temporal modeling
   - Retrieval needs: Semantic similarity, not sequence generation
   - May need hybrid: LVM for reranking, not first-stage retrieval

### For Production:

**Immediate**: Use AMN baseline (proven to work)
- Rerun on clean eval (Phase 2) to verify without leak
- AMN is larger but reliable

**Future**: Revisit LVM when architecture is proven on toy dataset

---

## Key Learnings

1. **Split objectives are dangerous**: 70/30 InfoNCE/AR created conflicting representations.

2. **Projection heads can harm**: Compressed space (256D) learned wrong semantics.

3. **Validation cosine ≠ generalization**: 0.5149 training val → 0.2163 eval cosine.

4. **Data leak doesn't save bad models**: 91.5% overlap didn't help → models memorize sequences, not meanings.

5. **Contrastive learning is fragile**: Small misconfigurations nullify benefits.

---

## Files

- **Full report**: `PHASE1_FAILURE_REPORT.md`
- **Contrastive eval**: `artifacts/lvm/phase1_contrastive_leaked_eval.json`
- **AR-only eval**: `artifacts/lvm/phase1_ar_only_leaked_eval.json`

---

## Next Steps

**Waiting for contractor decision**:
- [ ] Try Option 1 (raw 768D InfoNCE)?
- [ ] Skip to Option 4 (architecture pivot)?
- [ ] Accept AMN as baseline and move on?

**My suggestion**: Try Option 1 once (1 day), then pivot if it fails. Don't waste time on temperature sweeps if fundamental approach is flawed.

---

**Report completed**: 2025-10-27 14:30
**Status**: Phase 1 failed, root cause diagnosed, awaiting guidance
