# P3: Tiny Late Guards on P1 Baseline Architecture

**Status**: 🚀 Training Launched (2025-10-31)

## Approach Evolution

### V3: Directional Guardrails ❌ FAILED
- **Collapse**: Epoch 4 (val_cos 0.540 → 0.354)
- **Root cause**: Guards too strong (λ=0.01 = 10x MSE loss)
- **Result**: 5CAT margin -0.132, backward prediction bias persists

### P1: Clean MSE Baseline ✅ SUCCESS
- **Purpose**: Verify pipeline health
- **Result**: val_cos 0.550 by epoch 3
- **Conclusion**: Training stack works perfectly, no fundamental issues

### P2: Residual Prediction ❌ FAILED
- **Architecture**: `ŷ = norm(u + α·Δ)` where `u = ctx[-1]`
- **Theory**: Force model to predict delta, breaking identity copy
- **Result**: val_cos stuck at 0.472 (worse than P1), margin -0.534
- **Root cause**: **Made copying EASIER** - model outputs Δ≈0 → ŷ≈last_frame
- **Conclusion**: Architectural change backfired, abandon approach

### P3: Tiny Late Guards on Original Architecture 🚀 CURRENT
- **Philosophy**: P1 proved architecture works → add minimal nudges, not architectural surgery
- **Key insight**: Don't change the model head, just guide it gently during training

## P3 Configuration

### Architecture
- **Model**: TransformerVectorPredictor (original, NO residual wrapper)
- **Input**: 768D (no positional encoding)
- **Output**: 768D normalized vectors
- **Parameters**: ~6.3M

### Training Schedule

**Epochs 1-5: Pure MSE Warm-up**
```python
loss = mse_loss(y_pred, y_target)
# No guards, no context drop, pure learning
```

**Epochs 6-20: Tiny Late Guards**
```python
loss = mse_loss + tiny_guards
where:
  λ_dir = 0.002  # Directional: cos(pred,next) > cos(pred,prev) + 0.01
  λ_fut = 0.002  # Future rank: cos(pred,next) > cos(pred,{+2,+3}) + 0.008
  λ_ac  = 0.0    # Anti-copy: OFF (only if margin stalls)
  context_drop = 0.05  # Random noise in last frame
```

### Rationale for Tiny Guards

**Why λ=0.002 (not 0.01)?**
- V3 collapse showed λ=0.01 ≈ 10x stronger than MSE
- Guard losses should be **gentle nudges**, not dominant forces
- 5x reduction: λ=0.002 ≈ 2x MSE (balanced influence)

**Why start at epoch 6 (not 4)?**
- P1 showed epoch 3 val_cos = 0.540 (already learning well)
- Epochs 1-5: Let model learn basic MSE patterns freely
- Epoch 6+: Model already has good representations, now guide directionality

**Why context_drop p=0.05?**
- Prevents overfitting to exact last frame
- Forces model to use full context, not just ctx[-1]
- Small probability = minimal disruption to learning

### Success Criteria

**By Epoch 3** (warm-up phase):
- ✅ val_cos ≥ 0.50 (matching P1 baseline)
- ✅ No collapse, steady improvement

**By Epoch 10** (guards active):
- ✅ margin > 0 (positive directional preference)
- ✅ val_cos ≥ 0.52

**Final** (epoch 20):
- ✅ val_cos ≥ 0.54
- ✅ margin ≥ +0.08 (clear forward prediction)
- ✅ 5CAT: Pass 3/5 gates minimum

### Monitoring Plan

**Per-Epoch Validation**:
- Train/val cosine similarity
- Loss breakdown (MSE + guard components)
- Margin metrics (once guards activate)

**Early Warning Signs**:
- Epoch 3 val_cos < 0.50 → Warm-up failing (should match P1)
- Epoch 6 sudden drop → Guards too strong (shouldn't happen with λ=0.002)
- Epoch 10 margin < 0 → Still backward (need to investigate)

**Stop Conditions**:
- Val cosine collapse (> 0.05 drop in single epoch)
- Persistent negative margin after epoch 10 (guards not working)

## Key Learnings Applied

1. **Don't Fix What Isn't Broken**
   - P1 architecture already works (0.550 val_cos)
   - P2 architectural change made things worse
   - → Stick with proven architecture

2. **Guards Are Guides, Not Governors**
   - V3 taught us: strong guards (λ=0.01) cause collapse
   - → Use tiny guards (λ=0.002) as gentle nudges

3. **Warm-up Is Critical**
   - Let model learn basic patterns before adding constraints
   - 5 epochs pure MSE → solid foundation
   - Then add directional guidance

4. **Context Drop Prevents Shortcuts**
   - Model might learn to just copy ctx[-1]
   - Small random noise (p=0.05) forces broader context usage
   - Too high → disrupts learning; too low → allows shortcuts

## Files

**Training Script**:
- `scripts/train_transformer_p3_tiny_guards.sh`

**Model Output**:
- `artifacts/lvm/models/transformer_p3_tiny_guards/best_model.pt`
- `artifacts/lvm/models/transformer_p3_tiny_guards/training_history.json`
- `artifacts/lvm/models/transformer_p3_tiny_guards/training.log`

**Validation**:
```bash
./.venv/bin/python tools/tests/test_5to1_alignment.py \
  --model artifacts/lvm/models/transformer_p3_tiny_guards/best_model.pt \
  --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \
  --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \
  --articles-npz artifacts/wikipedia_584k_fresh.npz \
  --device mps --max-samples 5000
```

## Next Steps

1. **Monitor P3 Training** (epochs 1-5)
   - Expect: val_cos ≈ 0.50-0.55 (same as P1)
   - Watch for: Collapse or stalling

2. **Guards Activation** (epoch 6)
   - Should see: Smooth transition, no sudden drops
   - Watch for: Margin becomes positive

3. **Final Validation** (epoch 20)
   - Run full 5CAT test (5000 samples)
   - Expect: Positive margin, pass 3/5 gates

4. **If P3 Succeeds**
   - Document final metrics
   - Deploy to production ports (9007?)
   - Update training procedures

5. **If P3 Fails**
   - Analyze: Where did it break? (warm-up? guards? both?)
   - Consider: Even tinier guards (λ=0.001)?
   - Consider: Later guard activation (epoch 10)?

## Timeline

- **P3 Launch**: 2025-10-31 (evening)
- **Expected Duration**: ~4-5 hours (20 epochs @ 12-15 min/epoch)
- **Interim Check**: After epoch 6 (guards activate)
- **Final Check**: After epoch 20 (full 5CAT)
