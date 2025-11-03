# Session Handoff: P6b Directional Margin Loss
**Date**: November 2, 2025
**Next Session**: Ready to implement P6b

---

## 🎯 Where We Are

**DISCOVERY**: Wikipedia data has backward temporal bias (Δ = -0.069)
- Forward correlation (ctx[-1] → target_next): 0.3876
- Backward correlation (ctx[-1] → target_prev): 0.4569
- **All 6 approaches (P1-P6) failed because data teaches backward, not model bug**

---

## ✅ What's Ready

### P6 Data (431k sequences)
```bash
artifacts/lvm/training_sequences_ctx5_p6_next_token.npz    # 431,895 seqs
artifacts/lvm/validation_sequences_ctx5_p6_next_token.npz  # 18,360 seqs
artifacts/lvm/ood_sequences_ctx5_p6_next_token.npz         # 9,920 seqs
```

### P6 Baseline Model
```bash
artifacts/lvm/models/transformer_p6_20251102_131816/best_model.pt
```
- Val cosine: 0.511, R@5: 0.700 ✅
- Margin: -0.082 ❌ (proves backward bias in data)

### Tools
```bash
tools/diagnose_p6_direction.py       # Direction diagnostics
tools/tests/test_5to1_alignment.py   # Fixed for P6 format
scripts/train_transformer_p6_next_token.sh  # P6 baseline training
```

### Documentation
```bash
docs/WIKIPEDIA_BACKWARD_BIAS_ROOT_CAUSE.md          # Root cause paper
artifacts/lvm/SESSION_SUMMARY_2025_11_02_BACKWARD_BIAS.md  # Full summary
CLAUDE.md                                            # Updated with P6b status
```

---

## ⏳ What's Next: Implement P6b

### 1. Add Directional Margin Loss
**File**: `app/lvm/losses_directional.py`

```python
def directional_margin_loss(pred, target_next, target_prev, margin=0.05):
    """Enforce cos(pred, target_next) > cos(pred, target_prev) + margin"""
    import torch.nn.functional as F

    pos = F.cosine_similarity(pred, target_next, dim=-1)
    neg = F.cosine_similarity(pred, target_prev, dim=-1)

    # Hinge loss: max(0, margin - (pos - neg))
    loss = F.relu(margin - (pos - neg)).mean()

    return loss, pos.mean().item(), neg.mean().item()
```

### 2. Update Data Loader
**File**: `app/lvm/train_unified.py`

Add target_prev lookup in dataset `__getitem__`:
```python
# Look up target_prev from metadata
target_chunk_idx = metadata['target_chunk_index']  # Points to target_next
prev_chunk_idx = target_chunk_idx - 1

if prev_chunk_idx >= 0:
    # Fetch from article store using (article_id, prev_chunk_idx)
    target_prev = wiki_lookup[(article_id, prev_chunk_idx)]
else:
    # Use hard negative from same article (random chunk)
    target_prev = random.choice(article_vectors[article_id])
```

### 3. Create P6b Training Script
**File**: `scripts/train_transformer_p6b_directional.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# P6b: P6 Architecture + Directional Margin Loss
#
# Strategy: Override backward data bias with explicit forward enforcement
#
# Usage:
#   ./scripts/train_transformer_p6b_directional.sh [DEVICE]

PY=${PY:-./.venv/bin/python}
TR=app/lvm/train_unified.py
TEST=tools/tests/test_5to1_alignment.py

# P6 data files
TRAIN_NPZ=artifacts/lvm/training_sequences_ctx5_p6_next_token.npz
VAL_NPZ=artifacts/lvm/validation_sequences_ctx5_p6_next_token.npz
OOD_NPZ=artifacts/lvm/ood_sequences_ctx5_p6_next_token.npz
ART_NPZ=artifacts/wikipedia_584k_fresh.npz

DEVICE=${1:-mps}

# P6b hyperparameters
LAMBDA_DIR=1.0         # Directional loss weight (equal to MSE)
DIR_MARGIN=0.05        # Minimum forward advantage
HARD_NEGS=true         # Use target_prev as hard negative

echo "============================================="
echo "P6b: Directional Margin Loss"
echo "============================================="
echo "Training Data: $TRAIN_NPZ"
echo "Device: $DEVICE"
echo "Lambda Dir: $LAMBDA_DIR"
echo "Margin: $DIR_MARGIN"
echo ""

BASE_DIR="artifacts/lvm/models/transformer_p6b_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_DIR"

$PY $TR \
  --model-type transformer \
  --data "$TRAIN_NPZ" \
  --epochs 10 \
  --batch-size 32 \
  --device "$DEVICE" \
  --lambda-dir "$LAMBDA_DIR" \
  --dir-margin "$DIR_MARGIN" \
  --use-hard-negatives \
  --fivecat-every-epoch 1 \
  --fivecat-max-samples 2000 \
  --output-dir "$BASE_DIR"

echo ""
echo "[P6b] Full 5CAT evaluation..."
$PY $TEST --model "$BASE_DIR/best_model.pt" \
  --val-npz "$VAL_NPZ" --ood-npz "$OOD_NPZ" --articles-npz "$ART_NPZ" \
  --device "$DEVICE" --max-samples 5000 | tee "$BASE_DIR/5cat_results.json"

echo ""
echo "✅ P6b Training Complete!"
echo "   Expected: Margin ≥ +0.05, R@5 ≥ 70%, Val cosine ≥ 0.50"
echo "   Model: $BASE_DIR/best_model.pt"
```

### 4. Launch P6b Training
```bash
./scripts/train_transformer_p6b_directional.sh mps
```

**Monitor (every epoch)**:
- Margin should flip positive by epoch 3-5
- R@5 should stay ≥ 70%
- Val cosine should stay ≥ 0.50

---

## 📊 Expected P6b Results

| Metric | P6 Baseline | P6b Target | Why |
|--------|-------------|------------|-----|
| **Margin** | -0.082 ❌ | **+0.05** ✅ | Directional loss enforces forward > backward |
| **R@5** | 0.700 ✅ | **≥ 0.70** ✅ | Architecture unchanged, retrieval maintained |
| **Val Cosine** | 0.511 ✅ | **≥ 0.50** ✅ | Embedding quality preserved |

---

## 🔑 Key Commands

**Run direction diagnostics**:
```bash
./.venv/bin/python tools/diagnose_p6_direction.py \
  --train-npz artifacts/lvm/training_sequences_ctx5_p6_next_token.npz \
  --wiki-npz artifacts/wikipedia_584k_fresh.npz \
  --n-samples 5000
```

**Run 5CAT on P6 baseline**:
```bash
./.venv/bin/python tools/tests/test_5to1_alignment.py \
  --model artifacts/lvm/models/transformer_p6_20251102_131816/best_model.pt \
  --val-npz artifacts/lvm/validation_sequences_ctx5_p6_next_token.npz \
  --ood-npz artifacts/lvm/ood_sequences_ctx5_p6_next_token.npz \
  --articles-npz artifacts/wikipedia_584k_fresh.npz \
  --device mps --max-samples 5000
```

---

## 📚 Key Documentation

| Document | What It Contains |
|----------|------------------|
| `docs/WIKIPEDIA_BACKWARD_BIAS_ROOT_CAUSE.md` | Root cause analysis, offset heatmaps, solution |
| `artifacts/lvm/SESSION_SUMMARY_2025_11_02_BACKWARD_BIAS.md` | Complete session summary, timeline, findings |
| `CLAUDE.md` (lines 191-279) | P6b status, quick reference |

---

## 🎓 Lessons Learned

1. **Data can have directional bias** - Wikipedia has backward referential structure
2. **Architecture alone insufficient** - P6 proved this (removed shortcuts, margin still negative)
3. **Loss must explicitly enforce direction** - MSE follows dominant signal (backward in this case)
4. **Diagnostics are essential** - Direction tests revealed the truth

---

## 🚦 Decision Points

**If P6b succeeds** (margin ≥ +0.05):
- ✅ Deploy for production
- ✅ Document as canonical approach for Wikipedia data
- ✅ Consider testing on other datasets (arXiv, OpenStax) to verify generalization

**If P6b fails** (margin still negative):
1. Increase `lambda_dir` from 1.0 → 2.0 (stronger enforcement)
2. Increase `margin` from 0.05 → 0.10 (wider gap required)
3. Add explicit anti-backward loss: penalize `cos(pred, target_prev) > 0.3`
4. Try bidirectional context encoder (no causal mask) with forward-only head

---

**Session complete. Safe to `/clear`. All state preserved in documentation.** 🚀
