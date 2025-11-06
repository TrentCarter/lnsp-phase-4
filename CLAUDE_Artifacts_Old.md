# CLAUDE_Artifacts_Old.md

**Historical Reference Archive** - Detailed records from abandoned LVM training experiments

**Purpose**: This file contains detailed implementation guides and failure analyses from the AR-LVM training period (Oct-Nov 2025). These are preserved for historical reference but are NOT part of active operational guidance.

**Note**: For current operational guidance, see CLAUDE.md and LNSP_LONG_TERM_MEMORY.md

---

## 📌 P7 "Directional Ranker" - Full Failure Report (2025-11-04 Late Evening)

**STATUS**: ❌ **P7 BASELINE TRAINING FAILED** - Model learned backward prediction despite all defenses

### Training Results (10 epochs, arXiv data)
- **Training margin**: +0.12 (positive, learning forward ✓)
- **Validation margin**: **-0.067** (NEGATIVE, predicting backward ✗)
- cos(pred, next): 0.271 (low similarity to target)
- cos(pred, prev): 0.338 (HIGH similarity to previous chunk!)
- cos(pred, anchor): 0.430 (moderate context alignment)

### Critical Finding
Train/val mismatch - model learned forward on training but predicts backward on validation!

### Epoch 3 Collapse
- When teacher warmup ended (epoch 3), all metrics dropped 33-39%:
- cos_anchor: 0.588 → 0.391 (predictions drifted from context subspace)
- cos_next: 0.396 → 0.241 (lost alignment with target)
- Raw model predictions (q_raw) lost ~60% of context alignment
- Semantic anchoring (λ=0.8) couldn't prevent drift

### Root Cause Analysis
1. ✅ **Data is forward-biased** (Δ = +6.3% in validation set) - data quality is NOT the issue
2. ❌ **InfoNCE ranking loss** dominated margin loss (w_rank=1.0 > w_margin=0.5)
3. ❌ **Semantic anchoring** created conflicting gradients between raw predictions and anchored output
4. ❌ **Teacher pull** - model collapsed immediately when warmup ended at epoch 3
5. ⚠️ **Train/val distribution mismatch** - positive train margin but negative val margin suggests overfitting

### What ALL Defenses Failed
- InfoNCE ranking loss (supposed to prevent escape)
- Prev-repel margin loss (supposed to enforce forward directionality)
- Semantic anchoring λ=0.8 (supposed to keep predictions in context subspace)
- Directional gating (supposed to filter weak sequences)
- Teacher pull warmup (model collapsed when it ended)

### Files Created
- `app/lvm/losses_ranking.py` (430 lines) - P7 loss functions
- `app/lvm/models_p7_ranker.py` (330 lines) - TransformerP7Ranker + LSTMP7Ranker
- `app/lvm/train_p7_ranker.py` (470 lines, JSON bug fixed)
- `scripts/train_p7_ranker.sh` - Training interface
- `artifacts/lvm/P7_BASELINE_FAILURE_REPORT.md` - Complete analysis
- Model: `artifacts/lvm/models/p7_ranker_c5_m0.07_l0.8_20251104_222516/best_model.pt` (DO NOT USE)

### Next Steps Considered (Never Executed)
1. Investigate train/val mismatch - Check if distributions differ (5 min analysis)
2. Increase margin loss weight - w_margin: 0.5→1.5, w_rank: 1.0→0.5 (~10 hrs CPU)
3. Stronger semantic anchoring - λ: 0.8→0.6, add anchor loss (~10 hrs CPU)
4. Pure margin training - Disable InfoNCE entirely, use only margin + MSE (~10 hrs CPU)
5. **⚠️ Abandon autoregressive LVM** - After P1-P7 all failing

**Decision**: Abandoned after narrative delta test proved fundamental limitation

---

## 🎓 P6b v2.3 "GOLDILOCKS" - Full Implementation Guide (Never Trained)

**STATUS**: 🚀 **READY BUT NEVER TRAINED** - Superseded by abandonment decision

### Design Philosophy
P6b v2.3 attempted to balance between v2.1 (too conservative) and v2.2 (too aggressive) by using directional-when-confident gating.

### Architecture Changes from v2.2

**1. Directional-when-confident gate** (CRITICAL):
```python
# Scale loss by cos(pred, target)
confidence = cos(pred, target)
if confidence < 0.30:
    scale = 0  # directional OFF when misaligned
elif confidence > 0.45:
    scale = 1  # directional FULL when aligned
else:
    scale = (confidence - 0.30) / 0.15  # linear ramp

directional_loss = scale * directional_margin_loss(pred, next, prev)
```

**2. Lower ρ targets**:
- Epochs 1-2: ρ_target = 0.15 (baseline)
- Epochs 3-6: ρ_target = 0.20 (gentle climb)
- Epochs 7-12: ρ_target = 0.25 (not 0.35 like v2.2)

**3. Weaker penalties** (back to v2.1 values):
- τ = 0.10 (positive floor threshold)
- β = 1e-3 (positive floor penalty weight)
- κ = 1e-4 (orthogonality penalty weight)

**4. Lower λ_max**:
- 0.018 (was 0.03 in v2.2)

**5. Gentler margins**:
- Epochs 1-2: margin = 0.02
- Epochs 3-6: margin = 0.03
- Epochs 7-12: margin = 0.04 (not 0.06-0.07 like v2.2)

### Expected Results (12 epochs)

**Epoch-by-Epoch Prediction**:
- Epochs 1-2: Margin ≈ -0.04, ρ ≈ 0.15 (baseline)
- Epochs 3-4: Margin ≈ -0.02 to -0.01, ρ ≈ 0.20 (climbing)
- Epochs 5-6: **Margin flips positive** (0.00 → +0.01), ρ ≈ 0.25
- Epochs 7-9: Margin +0.02 to +0.04, ρ stable at 0.25
- Epochs 10-12: Margin +0.03 to +0.05 (stable positive)

**Final Model Targets**:
- ✅ Margin: +0.03 to +0.05 (POSITIVE!)
- ✅ R@5: ≥ 70% (high accuracy)
- ✅ Val cosine: ≥ 0.48 (good similarity)
- ✅ ρ: 0.25-0.35 (controlled by ρ-controller)
- ✅ Pass 3/5 5CAT gates minimum

### Training Script
```bash
# ⚠️ DO NOT TRAIN ON WIKIPEDIA - Use forward-flow data instead!
./scripts/train_transformer_p6b_v23.sh
```

**Reason Never Trained**: Narrative delta test (Nov 4) proved GTR-T5 lacks temporal signal at fundamental level

---

## 🎓 P6b v2.2 Implementation Details (Failed at Epoch 8)

### Core Loss Functions (`app/lvm/losses_directional.py`)
- `directional_margin_loss_v21()` - Scale-aware loss (α=0.7 mix of gap + ratio)
- `positive_floor_penalty()` - ReLU(τ - cos(pred, next))² with τ=0.12 (STRONGER)
- `norm_regularization()` - (||pred||₂ - 1)² penalty
- `orthogonality_penalty()` - (cos(pred, prev))² penalty (NEW)

### Training Integration (`app/lvm/train_unified.py`)
- **ρ-controller**: Actively pushes ρ to target (0.15 → 0.25 → 0.35)
- Epoch-gated schedule with higher margins (0.06-0.07)
- Higher λ_max (0.03, was 0.02)
- Stronger pos_floor (τ=0.12, β=2e-3)
- All v2.1 guardrails retained (skip logic, safety caps)
- CLI flag: `--p6b-v22`

### Training Script
```bash
./scripts/train_transformer_p6b_v22.sh
# 12 epochs, batch_size=32, lr=5e-4
# Automatic 5CAT validation at end
# Enhanced diagnostics (ρ vs ρ_target)
```

### Failure Mode (Epoch 8)
- Model: `artifacts/lvm/models/transformer_p6b_v22_20251102_203637/best_model.pt`
- Result: Margin +0.002 at E8 (briefly positive!), but **FAKE WIN** - orthogonal escape
- Val cosine: 0.44 → 0.18 (60% collapse!)
- R@5: 100% → 12% (retrieval broke)
- **Failure mode**: Directional pressure too strong (ρ=0.35), overwhelmed MSE loss
- Model learned to predict vectors FAR from target (negative cosine to prev: -0.086)
- Passed only 1/5 5CAT gates (need 3/5)
- **Verdict**: Proved that training tricks can't overcome backward data bias

---

## 🎓 P6b v2.1 "Six-Layer Defense" - Complete Results

### Implementation (✅ COMPLETED)
- Model: `artifacts/lvm/models/transformer_p6b_v21_20251102_182615/best_model.pt`
- Result: R@5 = 77% ✅, margin = **-0.047** ⚠️ (improved but still negative!)
- **Improved margin 43%** (-0.082 → -0.047) but didn't flip positive
- **Root cause**: Guardrails too conservative (ρ capped at 25%, directional loss too weak)
- **Verdict**: Stability proven ✅, but need stronger directional pressure

### Six Defense Layers
1. **Margin loss**: cos(pred, next) - cos(pred, prev) ≥ margin
2. **Positive floor**: cos(pred, next) ≥ τ (prevent negative cosines)
3. **Norm regularization**: ||pred||₂ ≈ 1 (stay on unit hypersphere)
4. **Directional gating**: Skip low-quality sequences (signal < 0.08)
5. **Epoch scheduling**: Gradual margin increase (0.02 → 0.04 → 0.06)
6. **Safety caps**: ρ ≤ 0.25 (prevent over-reliance on directional loss)

### Results
- Val cosine: 0.488 (vs 0.511 for P6 baseline)
- R@5: 0.769 (vs 0.700 for P6)
- Margin: -0.047 (vs -0.082 for P6) - **43% improvement!**
- Training stable, no collapse

### Why Margin Still Negative
- Guardrails too conservative (ρ capped at 25%)
- Directional loss weight (λ_max=0.02) too weak to overcome Wikipedia backward bias
- Safety caps prevented aggressive optimization

---

## 📊 P6 Data Details (Ready for Training, Never Used)

### P6 NEXT Token Architecture
Predict target_next instead of target to remove identity path.

**Data Files**:
- Training: 431,895 sequences (`artifacts/lvm/training_sequences_ctx5_p6_next_token.npz`)
- Validation: 18,360 sequences (`artifacts/lvm/validation_sequences_ctx5_p6_next_token.npz`)
- OOD: 9,920 sequences (`artifacts/lvm/ood_sequences_ctx5_p6_next_token.npz`)
- **Identity path removed**: cos(ctx[4], target_next) = 0.395 (vs ~0.8 for regular target)

### P6 Baseline Results (10 epochs, NO directional loss)
- Model: `artifacts/lvm/models/transformer_p6_20251102_131816/best_model.pt`
- Val cosine: 0.511, R@5: 0.700 ✅
- Margin: -0.082 ❌ (proves data has backward bias)
- Use for: Baseline comparison for P6b

### Direction Diagnostics (`tools/diagnose_p6_direction.py`)
- Forward (ctx[-1] → target_next): 0.3876
- Backward (ctx[-1] → target_prev): 0.4569
- **Δ = -0.0692** (backward is 7% stronger!)

---

## 📊 LVM DATA QUALITY REQUIREMENTS (Historical Reference)

**Note**: These requirements were developed for LVM training, now abandoned. Preserved for potential Q-tower ranker work.

### Mandatory Pre-Training Validation

**BEFORE training ANY LVM model:**
```bash
./.venv/bin/python tools/tests/diagnose_data_direction.py \
  artifacts/lvm/YOUR_TRAINING_DATA.npz --n-samples 5000
```

### Quality Gates (ALL Must Pass)

| Metric | Minimum | Target | What It Measures |
|--------|---------|--------|------------------|
| **Coherence** | ≥ 0.40 | 0.45-0.50 | Adjacent context positions are similar |
| **Temporal Signal** | ≥ +0.08 | +0.10 to +0.15 | pos[4] much closer to target than pos[0] |
| **Temporal Order** | Monotonic | Strictly increasing | pos[0] < pos[1] < ... < pos[4] → target |

**Good data example**: 584k clean (coherence 0.46, signal +0.12, monotonic ✅)
**Bad data example**: 340k old (coherence 0.35, signal +0.01, non-monotonic ❌)

### If Data Fails Diagnostic

**DO NOT PROCEED WITH TRAINING!** Instead:
1. Check sequence creation script and chunk boundaries
2. Verify source vectors have proper article/chunk ordering
3. Regenerate using `tools/create_training_sequences_with_articles.py`
4. Document root cause and update scripts to prevent recurrence

### 5→1 Causal Alignment Test (5CAT)

**AFTER training, before deployment:**
```bash
./.venv/bin/python tools/tests/test_5to1_alignment.py \
  --model artifacts/lvm/models/YOUR_MODEL/best_model.pt \
  --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \
  --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \
  --articles-npz artifacts/wikipedia_584k_fresh.npz \
  --device mps --max-samples 5000
```

### Passing Criteria (must pass 3/5 gates minimum)

| Gate | What It Tests | VAL Threshold | OOD Threshold |
|------|---------------|---------------|---------------|
| **A: Offset Sweep** | Predicts NEXT, not previous | Margin ≥ +0.12 | ≥ +0.10 |
| **B: Retrieval Rank** | Finds target in article | R@1≥60%, R@5≥95% | R@1≥55%, R@5≥92% |
| **C: Ablations** | Order matters | Shuffle delta ≤ -0.15 | ≤ -0.15 |
| **D: Rollout** | Multi-step coherence | Avg cos@H=5 ≥ 0.45 | ≥ 0.42 |
| **E: Bins Delta** | Generalization | abs(Val-OOD) ≤ 0.05 | ≤ 0.05 |

**🚨 CRITICAL**: If margin is **NEGATIVE**, model learned backward prediction! DO NOT DEPLOY!

### Pre-Training & Post-Training Checklists

**Before Training**:
- [ ] Run `diagnose_data_direction.py` on training data
- [ ] Verify coherence ≥ 0.40, temporal signal ≥ +0.08, monotonic order
- [ ] Use article-based splits (no article overlap in train/val/OOD)
- [ ] Document data source and creation method

**After Training**:
- [ ] Run full 5CAT test (5000 samples)
- [ ] Verify margin is POSITIVE (+0.10 minimum)
- [ ] Pass at least 3/5 gates
- [ ] Document 5CAT results and compare to baseline

**Only deploy models that pass both data quality and 5CAT validation!**

---

## 📊 Training Data Requirements (Historical)

### Recommended Training Data (Never Used After Abandonment)
```bash
# Training: 438k sequences from articles 1-1499, 2000-3999, 4500-7671
artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz

# Validation: 18k sequences from articles 4000-4499
artifacts/lvm/validation_sequences_ctx5_articles4000-4499.npz

# OOD Test: 10k sequences from articles 1500-1999 (truly held-out)
artifacts/lvm/wikipedia_ood_test_ctx5_TRULY_FIXED.npz
```

### Key Learnings
- ❌ **NEVER train without validating data quality first** - use `diagnose_data_direction.py`
- ❌ **NEVER use `random_split()` for train/val splits** - causes data contamination
- ❌ **NEVER deploy without 5CAT validation** - must pass 3/5 gates minimum
- ✅ **ALWAYS use article-based splits** - no article overlap between train/val/OOD
- ✅ **ALWAYS integrate 5CAT testing during training** - detects backward bias early
- ✅ **ALWAYS verify OOD generalization** - val score alone is not enough
- ✅ **Require minimum coherence ≥ 0.40, signal ≥ +0.08**

### Old Models (DO NOT USE)
- `artifacts/lvm/models_340k/*` - Backward bias (trained on low-quality data)
- `artifacts/lvm/models/transformer_directional_v3/` - Collapsed
- `artifacts/lvm/models/transformer_p{2,3,4}_*/` - Failed experiments
- `artifacts/lvm/models/transformer_p6_*/` - Baseline (backward bias)
- `artifacts/lvm/models/transformer_p6b_v21_*/` - Improved but still negative margin
- `artifacts/lvm/models/transformer_p6b_v22_*/` - Orthogonal escape failure
- `artifacts/lvm/models/p7_ranker_*/` - Train/val mismatch failure

---

## 📚 Related Documentation (Historical)

### Full Analysis Documents
- `artifacts/lvm/NARRATIVE_DELTA_TEST_FINAL.md` - Decisive test proving abandonment
- `artifacts/lvm/P8_PILOT_FAILURE_REPORT.md` - P8 constrained mixture failure
- `artifacts/lvm/P7_BASELINE_FAILURE_REPORT.md` - P7 ranker failure analysis
- `artifacts/lvm/P6B_V21_IMPLEMENTATION.md` - Six-layer defense implementation (500+ lines)
- `artifacts/lvm/P6B_EPOCH3_COLLAPSE_ANALYSIS.md` - P6b v1 collapse post-mortem
- `docs/WIKIPEDIA_BACKWARD_BIAS_ROOT_CAUSE.md` - Why Wikipedia is backward
- `artifacts/lvm/SESSION_SUMMARY_2025_11_02_BACKWARD_BIAS.md` - Complete session notes
- `artifacts/lvm/OOD_EVALUATION_FIX_COMPLETE_SUMMARY.md` - Data quality improvements
- `artifacts/lvm/TRAINING_SESSION_2025_11_01.md` - Training experiments log
- `artifacts/lvm/BACKWARD_BIAS_ROOT_CAUSE_REPORT.md` - Full investigation

### Tools Developed
- `tools/diagnose_p6_direction.py` - Direction diagnostics
- `tools/narrative_delta_check.py` - Narrative delta validation
- `tools/tests/diagnose_data_direction.py` - Data quality diagnostics
- `tools/tests/test_5to1_alignment.py` - 5CAT validation suite
- `tools/create_training_sequences_with_articles.py` - Proper data generation

---

**END OF HISTORICAL ARCHIVE**

_This file contains detailed records from Oct-Nov 2025 LVM training experiments._
_All approaches (P1-P8) ultimately failed due to fundamental GTR-T5 limitations._
_Decision made Nov 4, 2025: Pivot to retrieval-only vecRAG._
