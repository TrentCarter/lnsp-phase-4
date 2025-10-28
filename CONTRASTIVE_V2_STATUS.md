# Contrastive V2 Training Status
## Option 1: Raw 768D InfoNCE (No Projection Head)
## Started: 2025-10-27 14:45

---

## Training Configuration

**Model**: Mamba-S (53.2M params)
**Approach**: InfoNCE + AR cosine in **single 768D representation space**

### Loss Function Changes

| Component | V1 (Failed) | V2 (Current) |
|-----------|-------------|--------------|
| **Projection Head** | 768→512→256 | **NONE** |
| **InfoNCE Space** | 256D projected | **768D raw** |
| **AR Cosine Space** | 768D raw | **768D raw** |
| **λ_con** | 0.70 | **0.85** |
| **λ_ar** | 0.30 | **0.15** |
| **Temperature** | 0.07 | 0.07 |

**Key Fix**: Both objectives now optimize **same representation** → no split-objective conflict

---

## Training Details

- **Training samples**: 317,007
- **Validation samples**: 79,251
- **Batch size**: 256
- **Grad accum steps**: 4
- **Effective batch**: 1,024 (for InfoNCE negatives)
- **Device**: CPU
- **Early stop patience**: 3

**Regularization**:
- Article dropout: 20% (zero last k context positions)
- Span corruption: 10% (replace random position with different article)

---

## Expected Results

### Success Criteria (Epoch 2)

**Minimum gate** (same as V1):
- Val cosine ≥ 0.50

**Expected improvement** (if fix works):
- V1 at epoch 2: 0.4719 (missed gate)
- V2 at epoch 2: **≥ 0.50** (target)
- Interpretation: Single representation space helps convergence

### Success Criteria (Final)

**Phase 1 evaluation** (after training):
- R@5 > 0% (any retrieval)
- Ideally: R@5 ≥ 10% (meaningful improvement)
- Best case: R@5 ≥ 20% (strong generalization)

**Cosine comparison**:
- Training val: ~0.50-0.55
- Eval pred→target: **≥ 0.35** (not 0.22 like V1!)
- Gap should be smaller (no projection head conflict)

---

## Monitoring Commands

### Check Current Epoch
```bash
tail -20 logs/mamba_contrastive_v2_*.log
```

### Check Epoch 2 Progress
```bash
# Wait ~30 minutes, then:
python3 -c "
import json
try:
    h = json.load(open('artifacts/lvm/models/mamba_s_contrastive_v2/history.json'))
    if len(h) >= 2:
        e2 = h[1]
        print(f\"Epoch 2: val_cosine={e2['val_cosine']:.4f}\")
        if e2['val_cosine'] >= 0.50:
            print('✅ GATE PASSED')
        else:
            print(f\"❌ GATE MISSED (need ≥0.50)\")
    else:
        print('Epoch 2 not complete yet')
except: print('Training initializing...')
"
```

### Check Training Status
```bash
ps aux | grep 60830 | grep -v grep
```

### Full History
```bash
python3 -c "
import json
try:
    h = json.load(open('artifacts/lvm/models/mamba_s_contrastive_v2/history.json'))
    for e in h:
        print(f\"Epoch {e['epoch']}: val_cosine={e['val_cosine']:.4f} loss={e['train_loss']:.4f}\")
except: print('No history yet')
"
```

---

## Timeline

| Time | Milestone | Status |
|------|-----------|--------|
| **14:45** | Training started | ✅ Running (PID 60830) |
| **+25 min** (15:10) | Epoch 1 completes | ⏳ Pending |
| **+30 min** (15:15) | Epoch 2 gate check | ⏳ **Critical** |
| **+4-5 hours** (18:45-19:45) | Training completes | ⏳ Pending |
| **+5 hours** | Phase 1 re-evaluation | ⏳ Pending |

---

## What We're Testing

### Hypothesis
V1 failed because:
- InfoNCE optimized 256D projected space (70%)
- AR cosine optimized 768D raw space (30%)
- **Conflict**: Model learned incompatible representations

V2 fix:
- InfoNCE optimizes 768D raw space (85%)
- AR cosine optimizes 768D raw space (15%)
- **No conflict**: Both objectives align in same space

### Expected Outcomes

**If fix works**:
- Epoch 2 val_cosine ≥ 0.50 ✅
- Final eval cosine ≥ 0.35 (not 0.22)
- R@5 > 0% (some retrieval)
- Validates that split-objective was the problem

**If fix doesn't work**:
- Still 0% R@5
- Means InfoNCE fundamentally incompatible with autoregressive LVMs
- Proceed to Option 4 (two-tower architecture) or Option 5 (AMN baseline)

---

## Decision Tree

```
Epoch 2 completes (~15:15)
├─ val_cosine ≥ 0.50
│  ├─ ✅ Continue to completion
│  └─ Wait for full training
│
└─ val_cosine < 0.50
   ├─ Check if improving (0.40-0.49)
   │  └─ Give 2 more epochs
   │
   └─ Still low (<0.40)
      └─ ❌ Stop, move to Option 4

Training completes (~18:45)
├─ Run Phase 1 eval
│  ├─ R@5 > 10%
│  │  ├─ ✅ Fix worked!
│  │  └─ Proceed to Phase 2 (fresh Wikipedia)
│  │
│  └─ R@5 ≈ 0%
│     ├─ ❌ Fix didn't work
│     └─ Recommend Option 4 or 5
```

---

## Files

- **Training script**: `app/lvm/train_mamba_contrastive_v2.py`
- **Log**: `logs/mamba_contrastive_v2_20251027_*.log`
- **Checkpoint**: `artifacts/lvm/models/mamba_s_contrastive_v2/best.pt`
- **History**: `artifacts/lvm/models/mamba_s_contrastive_v2/history.json`

---

## Key Differences from V1

1. **No ProjectionHead class** → simpler architecture
2. **F.normalize() in loss function** → L2 norm in 768D
3. **λ_con increased to 0.85** → prioritize contrastive
4. **Both objectives in same space** → no representation conflict

---

**Status**: ⏳ Training in progress (Epoch 1)
**Next check**: ~15:15 (Epoch 2 gate)
**Expected completion**: ~18:45-19:45
