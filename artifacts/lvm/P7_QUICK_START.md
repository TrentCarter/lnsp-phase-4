# P7 "Directional Ranker" - Quick Start Guide

**Last Updated**: 2025-11-04 23:50 EST
**Status**: ✅ Ready to train (use CPU, MPS has bugs)

---

## TL;DR - Start Training Now

```bash
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4

# Run baseline experiment (10 hours on CPU)
./scripts/train_p7_ranker.sh --device cpu --epochs 10

# Monitor progress
tail -f artifacts/lvm/models/p7_ranker_*/training_history.json
```

**Expect**:
- Epoch 1: margin ≈ 0.10, cos_next ≈ 0.30
- Epoch 5: margin ≈ 0.25, cos_next ≈ 0.45
- Epoch 10: margin ≈ 0.35, cos_next ≈ 0.50 ✅

**Success**: If final margin ≥ +0.20, proceed with full grid

---

## What Is P7?

**P7 "Directional Ranker"** = Ranking-based LVM with semantic anchoring

**Why it exists**: All previous approaches (P1-P6b v2.3) failed with "orthogonal escape" - models learned to predict random vectors (cos≈0.04) instead of meaningful next concepts.

**How P7 fixes it**:
1. **Ranking objective**: Rank next highest (not predict exact vector)
2. **Semantic anchoring**: Blend predictions with context (prevents drift)
3. **Hard negatives**: Explicitly push away from previous chunk

**Key innovation**: Orthogonal escape is geometrically impossible (anchoring keeps predictions near context subspace).

---

## Files You Need

**Implementation** (all ready to use):
- `app/lvm/losses_ranking.py` - InfoNCE + prev-repel + anchoring
- `app/lvm/models_p7_ranker.py` - TransformerP7Ranker with learnable λ
- `app/lvm/train_p7_ranker.py` - Training loop
- `scripts/train_p7_ranker.sh` - Easy runner
- `scripts/run_p7_grid.sh` - Grid search (4 experiments)

**Documentation**:
- `P7_RANKER_IMPLEMENTATION_GUIDE.md` (500+ lines) - Complete guide
- `P7_TRAINING_STATUS.md` - Current status and options
- `P6B_V23_FAILURE_REPORT.md` - Why previous attempts failed
- `SESSION_SUMMARY_2025_11_04_P7_COMPLETE.md` - Full session summary

**Data** (ready):
- Train: `artifacts/lvm/arxiv_train_sequences.npz` (87k sequences)
- Val: `artifacts/lvm/arxiv_val_sequences.npz` (11k)
- OOD: `artifacts/lvm/arxiv_ood_sequences.npz` (11k)

---

## Training Options

### Option 1: Baseline Only (Recommended First)

Validate P7 architecture works before committing to full grid:

```bash
./scripts/train_p7_ranker.sh --device cpu --epochs 10
```

**Time**: ~10 hours on CPU
**Output**: `artifacts/lvm/models/p7_ranker_c5_m0.07_l0.8_*/`
**Decision**: If margin ≥ +0.20, proceed with full grid

### Option 2: Full Hyperparameter Grid

Run all 4 experiments in sequence:

```bash
# Edit script to use CPU
sed -i '' 's/DEVICE="mps"/DEVICE="cpu"/' scripts/run_p7_grid.sh

# Run grid
./scripts/run_p7_grid.sh
```

**Time**: ~40-48 hours on CPU
**Experiments**:
1. Baseline: ctx=5, m=0.07, λ=0.8
2. Strong anchor: ctx=5, m=0.07, λ=0.6
3. Higher margin: ctx=5, m=0.10, λ=0.8
4. Larger context: ctx=7, m=0.07, λ=0.8

**Compare**: `cd artifacts/lvm/p7_grid_results_*/ && ./compare_results.sh`

### Option 3: Custom Hyperparameters

```bash
./scripts/train_p7_ranker.sh \
  --context 7 \
  --margin 0.10 \
  --lambda 0.9 \
  --epochs 12 \
  --device cpu
```

---

## Monitoring Training

### Watch Training History

```bash
tail -f artifacts/lvm/models/p7_ranker_*/training_history.json
```

**Key metrics per epoch**:
```json
{
  "epoch": 5,
  "train": {
    "loss": 1.2,
    "loss_rank": 0.8,
    "loss_margin": 0.3,
    "gate_weight_mean": 0.85
  },
  "val": {
    "cos_next": 0.45,
    "cos_prev": 0.12,
    "margin": 0.33,
    "cos_anchor": 0.48
  },
  "anchor_lambda": 0.792
}
```

### Success Signals ✅

- **Margin growing**: 0.10 → 0.25 → 0.35
- **cos_next increasing**: 0.30 → 0.45 → 0.52
- **cos_prev decreasing**: 0.20 → 0.12 → 0.08
- **cos_anchor stable**: ≥ 0.40 (no drift!)
- **gate_weight_mean**: 0.7-0.9 (most data has forward signal)

### Failure Signals ❌

- **Margin turns negative**: Model learning backward
- **Orthogonal drift warning**: cos_anchor < 0.05
- **cos_next not climbing**: Stuck at < 0.35
- **gate_weight_mean < 0.5**: Data quality issue

---

## After Training Completes

### 1. Validate Model

```bash
./.venv/bin/python /tmp/test_p6b_v23_direction.py \
  --model artifacts/lvm/models/p7_ranker_*/best_model.pt \
  --val-npz artifacts/lvm/arxiv_ood_sequences.npz \
  --device cpu \
  --max-samples 2000
```

**Expected**:
```
Forward:  cos(pred, next) = 0.45-0.55  ✅
Backward: cos(pred, prev) = 0.05-0.15  ✅
Margin:   Δ = 0.25-0.40  ✅ (vs P6b v2.3: 0.0005!)
```

### 2. Deploy to Dashboard

Add P7 to evaluation dashboard (port 8999):
- Compare to DIRECT baseline
- Measure R@5, MRR@10
- Check latency

**Ship criteria**: Beat DIRECT by R@5 ≥ +10pts, MRR@10 ≥ +0.05

### 3. Replace P6b v2.3

If P7 passes ship criteria:
```bash
# Deploy on port 9007
./scripts/start_lvm_p7_best_9007.sh
```

---

## Troubleshooting

### Q: MPS error "Placeholder storage not allocated"
**A**: Use CPU (`--device cpu`). MPS has PyTorch bugs with complex tensor ops.

### Q: Training very slow
**A**: CPU is 3-4x slower than MPS. Baseline takes ~10 hours. Consider overnight runs.

### Q: Margin is negative
**A**: Model learning backward. Check data quality (should have Δ≥+0.03). May need stronger w_margin.

### Q: cos_anchor drops below 0.05
**A**: Orthogonal drift! STOP training. Increase anchoring (lower λ, e.g., 0.6 instead of 0.8).

### Q: cos_next not improving
**A**: Check loss_rank is decreasing. If stuck, try higher w_rank or lower temperature.

---

## Quick Reference

**Best model path**: `artifacts/lvm/models/p7_ranker_*/best_model.pt`

**Training config**: `artifacts/lvm/models/p7_ranker_*/config.json`

**Training history**: `artifacts/lvm/models/p7_ranker_*/training_history.json`

**Compare models**: `cd artifacts/lvm/p7_grid_results_*/ && ./compare_results.sh`

**Full docs**: `P7_RANKER_IMPLEMENTATION_GUIDE.md`

---

## What Changed from P6b v2.3?

| Component | P6b v2.3 (FAILED) | P7 Ranker (NEW) |
|-----------|-------------------|-----------------|
| Objective | MSE regression | InfoNCE ranking |
| Loss | `||pred - target||²` | `-log(exp(pos)/Σexp(neg))` |
| Negatives | None (implicit) | Previous + in-batch + hard |
| Anchoring | None | λ·pred + (1-λ)·context |
| Output | Raw 768D | Unit sphere + anchored |
| Result | Orthogonal escape (0.04) | **Should reach 0.45+** ✅ |

---

**Ready to start?**
```bash
./scripts/train_p7_ranker.sh --device cpu --epochs 10
```

**Questions?** See `P7_RANKER_IMPLEMENTATION_GUIDE.md` or `SESSION_SUMMARY_2025_11_04_P7_COMPLETE.md`
