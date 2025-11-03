# LVM Training Session Handoff: P6b v2.2 (Nov 2, 2025)

## 🎯 Current Status: P6b v2.2 READY TO TRAIN

### What Happened This Session

**P6b v2.1 COMPLETED** ✅
- Training finished: 12 epochs, all 6 guardrails worked perfectly
- **Results**: margin -0.047, R@5 77%, val_cos 0.488
- **Improved 43%** from P6 baseline (-0.082 → -0.047)
- **But**: Margin still NEGATIVE (didn't flip positive)
- **Root cause**: Guardrails too conservative (ρ capped at 25%, directional loss too weak)

**P6b v2.2 IMPLEMENTED** 🚀
- Surgical escalation: stronger directional pressure while keeping v2.1's stability
- ρ-controller: Makes ρ a TARGET (0.35), not just a cap
- Stronger anchors: pos_floor τ=0.12, β=2e-3
- Orthogonality penalty: κ=5e-4 (anti-prev bias)
- Higher margins: 0.06-0.07 (was 0.05)
- All v2.1 guardrails retained

---

## 📦 P6b v2.2 Implementation Summary

### 1. New Loss Components (losses_directional.py)

**Orthogonality Penalty** (lines 488-525):
```python
def orthogonality_penalty(pred, y_prev):
    """κ = 5e-4: Gently penalizes cos(pred, prev)^2"""
    # Chips away at backward bias without breaking legitimate similarity
```

### 2. ρ-Controller Logic (train_unified.py:687-710)

**Active Control** (not passive capping):
```python
# Target-based control
lambda_eff = (mse_val * rho_target) / dir_val

# If ρ < target: increase λ
if rho < rho_target * 0.8:
    lambda_eff *= (rho_target / max(rho, 1e-6))

# If ρ > cap: safety decrease
elif rho > rho_cap:
    lambda_eff *= (rho_cap / rho)
```

### 3. Epoch-Gated Schedule (train_unified.py:650-662)

| Epochs | ρ_target | ρ_cap | margin_gap | margin_ratio |
|--------|----------|-------|------------|--------------|
| 1-2 | 0.15 | 0.35 | 0.02 | 0.05 |
| 3-4 | 0.25 | 0.45 | 0.04 | 0.05 |
| 5-8 | 0.35 | 0.50 | 0.06 | 0.05 |
| 9-12 | 0.35 | 0.50 | 0.07 | 0.05 |

### 4. Stronger Anchors

- **pos_floor**: τ=0.12 (was 0.10), β=2e-3 (was 1e-3)
- **λ_max**: 0.03 (was 0.02)
- **Orthogonality**: κ=5e-4 (NEW)

### 5. Training Script

**File**: `scripts/train_transformer_p6b_v22.sh`
- 12 epochs, batch_size=32, lr=5e-4
- Automatic 5CAT validation at end
- Enhanced diagnostics (ρ vs ρ_target)

---

## 🎯 Expected P6b v2.2 Results

### Epoch-by-Epoch Timeline

| Epochs | Margin | R@5 | Val Cos | ρ | Status |
|--------|--------|-----|---------|---|--------|
| 1-2 | -0.04 | 72% | 0.49 | 0.15 | Baseline |
| 3-4 | -0.02 to -0.01 | 72% | 0.49 | 0.25 | Climbing |
| 5-6 | **0.00 → +0.01** | 72% | 0.48 | 0.35 | **FLIP!** |
| 7-9 | +0.02 to +0.04 | 72% | 0.48 | 0.35 | Stable |
| 10-12 | +0.03 to +0.05 | ≥70% | ≥0.48 | 0.35 | **TARGET** |

### Final 5CAT Targets

**Must pass 3/5 gates**:
- ✅ **Gate A (Offset Sweep)**: Margin ≥ +0.10 (VAL), ≥ +0.08 (OOD)
- ✅ **Gate B (Retrieval Rank)**: R@1 ≥ 55%, R@5 ≥ 92%
- ✅ **Gate D (Rollout)**: avg_cos@H=5 ≥ 0.45
- ✅ **Gate E (Bins Delta)**: |VAL-OOD| ≤ 0.05

---

## 🚀 Next Session Action Items

### Immediate (Priority 1)

1. **Launch P6b v2.2 training**:
   ```bash
   ./scripts/train_transformer_p6b_v22.sh
   ```

2. **Monitor first 3 epochs closely**:
   - Watch for ρ tracking ρ_target (should be within ±0.05)
   - Verify skip=0 (no collapse warnings)
   - Check pos > 0, neg > 0 (both cosines positive)

3. **Check epoch 6 margin**:
   - Should be near zero or slightly positive
   - If still strongly negative (<-0.02), consider v2.3

### Post-Training (Priority 2)

4. **Run full 5CAT validation**:
   - Automatically runs at end of script
   - Check results in `artifacts/lvm/models/transformer_p6b_v22_*/5cat_results.json`

5. **Compare with v2.1 results**:
   - Margin: -0.047 (v2.1) vs ??? (v2.2)
   - R@5: 0.769 (v2.1) vs ??? (v2.2)
   - ρ: 0.10-0.25 (v2.1) vs 0.35-0.50 (v2.2)

### Decision Points

**If v2.2 succeeds** (margin > 0):
- ✅ Deploy as production LVM
- ✅ Update all documentation
- ✅ Run extended 5CAT on OOD test set
- ✅ Benchmark inference performance

**If v2.2 fails** (margin still negative):
- Option A: v2.3 with exponential ramp (0.02 → 0.05 → 0.12 → 0.30)
- Option B: Remove adaptive guard entirely (nuclear)
- Option C: Investigate synthetic data with forward bias

---

## 📊 P6b Evolution Summary

| Version | Margin | R@5 | Issue | Fix |
|---------|--------|-----|-------|-----|
| P6 | -0.082 | 70% | No directional loss | Add directional |
| P6b v1 | N/A | N/A | Collapsed epoch 3 | 6 guardrails |
| P6b v2.1 | -0.047 | 77% | Guards too weak | ρ-controller |
| **P6b v2.2** | **TBD** | **TBD** | **-** | **Stronger pressure** |

**Key Insight**: Each version improved stability or margin, but not both. v2.2 aims to improve margin while keeping v2.1's stability.

---

## 🔬 Monitoring Checklist

### During Training (Every 200 Steps)

- [ ] ρ ≈ ρ_target (±0.05)
- [ ] skip = 0 (no collapse warnings)
- [ ] pos > 0.0 AND neg > 0.0 (both cosines positive)
- [ ] gap climbing (should increase over time)
- [ ] λ_eff adjusting dynamically (controller active)

### Per Epoch (Mini-5CAT)

- [ ] Margin improving ≥ +0.01 every 2-3 epochs
- [ ] R@5 ≥ 70% (maintained)
- [ ] Val cosine ≥ 0.48 (stable)
- [ ] No sudden drops or spikes

### Final (Full 5CAT)

- [ ] Margin > 0 (POSITIVE!)
- [ ] Pass ≥ 3/5 gates
- [ ] R@5 ≥ 70%
- [ ] Val cosine ≥ 0.48

---

## 📁 Key Files

### Models
- **P6 Baseline**: `artifacts/lvm/models/transformer_p6_20251102_131816/best_model.pt`
- **P6b v2.1**: `artifacts/lvm/models/transformer_p6b_v21_20251102_182615/best_model.pt`
- **P6b v2.2**: `artifacts/lvm/models/transformer_p6b_v22_*/` (will be created)

### Data
- Training: `artifacts/lvm/training_sequences_ctx5_p6_next_token.npz` (431,895 sequences)
- Validation: `artifacts/lvm/validation_sequences_ctx5_p6_next_token.npz` (18,360 sequences)
- OOD: `artifacts/lvm/ood_sequences_ctx5_p6_next_token.npz` (9,920 sequences)

### Code
- Loss functions: `app/lvm/losses_directional.py`
- Training loop: `app/lvm/train_unified.py`
- Training script: `scripts/train_transformer_p6b_v22.sh`
- 5CAT validation: `tools/tests/test_5to1_alignment.py`

### Documentation
- Session handoff: `artifacts/lvm/SESSION_HANDOFF_P6B_V22_2025_11_02.md` (this file)
- Implementation guide: `artifacts/lvm/P6B_V21_IMPLEMENTATION.md` (500+ lines, still relevant)
- Root cause analysis: `docs/WIKIPEDIA_BACKWARD_BIAS_ROOT_CAUSE.md`
- v2.1 results: Check 5CAT output from v2.1 training logs

---

## 🎓 Design Rationale

### Why ρ-controller vs just higher λ?

Fixed λ can oscillate:
- Too high → collapse (v1 failed)
- Too low → no effect (v2.1 issue)

Controller adapts to MSE magnitude → stable across epochs

### Why τ=0.12 for pos_floor?

- v2.1 used τ=0.10 (10% minimum similarity)
- Data backward bias is Δ = -0.069 (7%)
- Need τ > 0.10 to prevent "barely positive" victories

### Why κ=5e-4 for orthogonality?

- Too high → breaks legitimate prev similarity
- Too low → no effect
- 5e-4 = gentle nudge (0.05% of main loss)

### Why skip directional sprints?

- Requires global state tracking (batch-to-batch)
- Adds complexity with minimal gain
- Can add in v2.3 if v2.2 doesn't flip margin

---

## ✅ Pre-Flight Checklist (COMPLETED)

- [x] Orthogonality penalty implemented (losses_directional.py:488-525)
- [x] ρ-controller implemented (train_unified.py:687-710)
- [x] Epoch-gated schedule configured (train_unified.py:650-662)
- [x] Stronger anchors configured (τ=0.12, β=2e-3, λ_max=0.03)
- [x] Training script created (scripts/train_transformer_p6b_v22.sh)
- [x] All components tested (import + functional tests passed)
- [x] --p6b-v22 flag added to argparse
- [x] Documentation updated (CLAUDE.md)

**Status**: ✅ READY TO TRAIN

---

**Generated**: 2025-11-02
**Author**: Claude Code + User
**Next Steps**: Launch training with `./scripts/train_transformer_p6b_v22.sh`
