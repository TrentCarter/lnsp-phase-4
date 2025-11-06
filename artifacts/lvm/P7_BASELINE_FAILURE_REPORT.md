# P7 Baseline Training Failure Report
**Date:** 2025-11-04
**Model:** TransformerP7Ranker (context=5, margin=0.07, λ=0.8)
**Training:** 10 epochs on arXiv data (87k train, 11k val)
**Result:** ❌ **FAILED - Model learned backward prediction**

---

## Executive Summary

Despite implementing the complete P7 "Directional Ranker" architecture with:
- ✅ InfoNCE ranking loss (prevents orthogonal escape)
- ✅ Prev-repel margin loss (explicit backward penalty)
- ✅ Semantic anchoring (geometric constraint: q' = λ·q + (1-λ)·c)
- ✅ Directional gating (filter weak sequences)
- ✅ Teacher pull for warmup

**The model STILL learned to predict backward on validation data.**

---

## Final Metrics

| Metric | Train | Validation | Analysis |
|--------|-------|------------|----------|
| **Margin** | +0.124 | **-0.067** | ❌ Train is forward, val is backward! |
| cos(pred, next) | N/A | 0.271 | Lower than expected |
| cos(pred, prev) | N/A | 0.338 | Model prefers previous chunk |
| cos(pred, anchor) | N/A | 0.430 | Decent context alignment |
| Loss | 0.526 | N/A | Training loss decreased smoothly |

**Critical Finding:** Training margin stayed positive (+0.12 to +0.13) while validation margin was consistently negative (-0.06 to -0.07). This is a severe train/val mismatch, not a simple backward bias!

---

## Epoch 3 Collapse

**When teacher warmup ended (epoch 3), the model collapsed:**

| Epoch | cos_next | cos_prev | cos_anchor | Margin |
|-------|----------|----------|------------|--------|
| 2 | 0.396 | 0.455 | **0.588** | -0.059 |
| 3 | **0.241** ↓39% | **0.305** ↓33% | **0.391** ↓33% | -0.064 |

**All three metrics dropped ~33-39%!** The model's predictions drifted away from:
- The target next chunk
- The previous chunk
- The context subspace itself

**Interpretation:**
- Raw predictions (q_raw) drifted from ~41% context alignment to ~17%
- Semantic anchoring (λ=0.8) pulled them back, but not enough
- The loss functions couldn't recover from this drift

---

## Root Cause Analysis

### 1. **Data Validation ✅ (Data is forward-biased)**

Checked validation data temporal structure:
```
cos(c0, target): 0.461 (first context → target)
cos(c4, target): 0.523 (last context → target)
Δ = +0.063 (6.3% forward bias)
```

**Conclusion:** The data IS forward-biased. This rules out data quality issues.

### 2. **Architecture Issues**

**Problem:** Semantic anchoring creates conflicting gradients
- Loss functions optimize anchored output: q' = norm(0.8·q + 0.2·c)
- Gradients flow back to raw predictions q, but anchoring transformation is non-linear
- Model can learn q that's far from context, relying on anchoring to pull it back
- This creates instability when teacher warmup ends

**Evidence:**
- cos_anchor dropped from 0.588 → 0.391 when teacher pull disabled
- Raw predictions (q) lost ~60% of context alignment (0.41 → 0.17)
- Anchored output retained only 43% context alignment despite λ=0.8

### 3. **Loss Function Imbalance**

**Weights used:**
- w_rank = 1.0 (InfoNCE ranking loss)
- w_margin = 0.5 (prev-repel margin loss)
- w_teacher = 0.2 (teacher pull, warmup only)

**Problem:** Ranking loss (w=1.0) dominated margin loss (w=0.5)
- InfoNCE optimizes for ranking positive above negatives IN-BATCH
- In-batch negatives are random samples, not necessarily backward samples
- Margin loss (which explicitly penalizes backward) had only 50% weight

**Training shows:** loss_margin stayed constant (~0.12-0.13) while loss_rank decreased (2.68 → 2.04)
- Model optimized ranking, ignoring margin constraint
- Validation shows margin NEGATIVE despite margin loss being active

### 4. **Train/Val Mismatch** ⚠️

**Critical observation:**
- Training margin: +0.12 (positive, learning forward)
- Validation margin: -0.06 (negative, predicting backward)

This is NOT a simple "data is backward" problem. The model IS learning forward on training data, but it's NOT generalizing to validation!

**Possible causes:**
1. Directional gating (gate_weight=0.25) filters sequences differently in train vs val
2. In-batch negatives create spurious correlations in training batches
3. Batch size (64) might be too small for effective InfoNCE contrast
4. Validation set has different temporal characteristics than training set

---

## What Didn't Work

1. ❌ **InfoNCE ranking loss:** Did not prevent backward prediction
2. ❌ **Semantic anchoring (λ=0.8):** Insufficient to keep model in context subspace
3. ❌ **Prev-repel margin loss:** Overwhelmed by ranking loss, didn't enforce constraint
4. ❌ **Directional gating:** Either ineffective or causing train/val mismatch
5. ❌ **Teacher pull warmup:** Model collapsed immediately when it was turned off

---

## Next Steps (Ranked by Priority)

### Option 1: Investigate Train/Val Distribution Mismatch 🔍
**Hypothesis:** Training and validation sets have different temporal characteristics

**Actions:**
```bash
# Analyze train vs val temporal bias
python tools/compare_train_val_bias.py \
  --train artifacts/lvm/arxiv_train_sequences.npz \
  --val artifacts/lvm/arxiv_val_sequences.npz
```

**If mismatch confirmed:** Re-split data with proper stratification (by article, by topic, etc.)

---

### Option 2: Increase Margin Loss Weight 💪
**Hypothesis:** Margin loss (w=0.5) was too weak vs InfoNCE (w=1.0)

**Changes:**
- w_margin: 0.5 → **1.5** (3x stronger)
- w_rank: 1.0 → **0.5** (reduce InfoNCE dominance)
- Keep teacher pull longer: warmup_epochs: 2 → **5**

**Script:**
```bash
./scripts/train_p7_ranker.sh \
  --device cpu \
  --epochs 15 \
  --exp-name p7_strong_margin
```

---

### Option 3: Stronger Semantic Anchoring 🔗
**Hypothesis:** λ=0.8 (80% raw, 20% context) is too weak

**Changes:**
- anchor_lambda: 0.8 → **0.6** (60% raw, 40% context)
- Add anchor loss: penalize when cos(q_raw, context) < threshold
- Disable anchor learning (keep fixed at 0.6)

**Expected:** cos_anchor should stay > 0.5 throughout training

---

### Option 4: Pure Margin-Based Training (No InfoNCE) 🎯
**Hypothesis:** InfoNCE ranking loss is causing the train/val mismatch

**Changes:**
- Disable InfoNCE entirely (w_rank = 0.0)
- Use ONLY: margin loss + teacher pull + cosine MSE to target
- Simpler objective: cos(pred, next) - cos(pred, prev) ≥ margin

**Rationale:** Remove complexity, focus purely on forward vs backward signal

---

### Option 5: Abandon Autoregressive LVM ⚠️
**Hypothesis:** Vector-space autoregression is fundamentally flawed for semantic prediction

**Why this might be right:**
- All attempts (P1-P6b, P7) have failed or barely succeeded
- Forward temporal signal in text is WEAK (+6.3% in arXiv, -6.9% in Wikipedia)
- Even with ranking losses and constraints, models prefer identity/backward shortcuts

**Alternative directions:**
1. **Retrieval-only approach:** Use LVM for retrieval ranking, not generation
2. **Bi-directional training:** Predict both directions, use for relationship modeling
3. **Multi-scale training:** Predict at sentence/paragraph/section granularity
4. **Switch to textRAG:** Abandon vecRAG entirely, use text-based retrieval

---

## Recommended Immediate Action

**Before running more expensive experiments:**

1. **Run train/val distribution analysis** (Option 1, ~5 min)
   - If distributions differ significantly, fix data split first
   - This could explain the train/val mismatch entirely

2. **If distributions are same, try Option 2** (strong margin, ~10 hrs CPU)
   - Simplest fix: rebalance loss weights
   - Keep all P7 components, just adjust hyperparameters

3. **If Option 2 fails, try Option 4** (pure margin, ~10 hrs CPU)
   - Remove InfoNCE complexity
   - Simpler objective might generalize better

4. **If all fail, seriously consider Option 5**
   - After P1-P7 all failing, may need to pivot strategy
   - Retrieval-focused LVM or abandon vecRAG approach

---

## Key Learnings

1. **Architectural complexity ≠ better results**
   - P7 had 5 defense mechanisms, all failed
   - Simpler approaches (pure margin loss) might work better

2. **Train/val metrics can diverge dramatically**
   - Positive train margin, negative val margin = overfitting or distribution mismatch
   - Need better validation strategy (multiple val sets, OOD tests)

3. **Forward temporal signal is WEAK even in "good" data**
   - arXiv papers: only +6.3% forward bias
   - This weak signal is easily overwhelmed by shortcuts

4. **Semantic anchoring alone is insufficient**
   - Even with 80% anchoring, model drifted 33% at epoch 3
   - Need additional constraints or different architecture

---

## Files Created This Session

- `app/lvm/losses_ranking.py` (430 lines) - P7 loss functions
- `app/lvm/models_p7_ranker.py` (330 lines) - TransformerP7Ranker + LSTMP7Ranker
- `app/lvm/train_p7_ranker.py` (470 lines) - Training loop (JSON serialization fixed)
- `scripts/train_p7_ranker.sh` - Training interface
- `scripts/run_p7_grid.sh` - Hyperparameter grid
- `artifacts/lvm/P7_RANKER_IMPLEMENTATION_GUIDE.md` (500+ lines)
- `artifacts/lvm/P7_TRAINING_STATUS.md`
- `artifacts/lvm/P7_QUICK_START.md`
- `artifacts/lvm/SESSION_SUMMARY_2025_11_04_P7_COMPLETE.md`
- Model checkpoint: `artifacts/lvm/models/p7_ranker_c5_m0.07_l0.8_20251104_222516/best_model.pt`

---

**Status:** P7 baseline training complete, but FAILED to learn forward prediction. Need to investigate train/val mismatch or pivot to simpler approach.
