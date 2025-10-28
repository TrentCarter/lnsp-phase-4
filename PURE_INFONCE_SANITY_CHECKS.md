# Pure InfoNCE: Sanity Check Results
## 2025-10-27 15:00

---

## TL;DR

**All plumbing checks PASSED** ✅

Training in progress (PID 61274, ~3.5 hours elapsed).
Waiting for epoch 1 completion for retrieval test.

---

## Sanity Check Results

### 1. Oracle Recall (Index Sanity) ✅

**Test**: Use true payload vectors as queries → should find themselves

| Metric | Result | Status |
|--------|--------|--------|
| **R@1** | 95.2% | ✅ PASS |
| **R@5** | 100.0% | ✅ PASS |
| **R@50** | 100.0% | ✅ PASS |

**Interpretation**: FAISS IVF index working correctly with nprobe=64.

**Details**:
- Tested 1,000 random payload vectors as queries
- All found themselves in top-5 (100%)
- 95.2% found at rank 1 (small IVF quantization loss, expected)
- Index type: `IndexIVFFlat` with Inner Product metric

---

### 2. Vector Normalization ✅

**Payload vectors**:
- Mean norm: 1.000000
- Min: 0.999999, Max: 1.000001
- Std: 0.000000
- ✅ All vectors L2-normalized

**Index metric**: Inner Product (IP) ✓

**Conclusion**: Cosine similarity computed correctly via IP on normalized vectors.

---

### 3. Eval Alignment ✅

**Test**: Check if eval targets match payload vectors

| Check | Result | Status |
|-------|--------|--------|
| **Mean cosine** | 1.000000 | ✅ PASS |
| **Min cosine** | 1.000000 | ✅ PASS |
| **Aligned (cos > 0.999)** | 100/100 (100%) | ✅ PASS |

**Interpretation**: Eval NPZ targets are EXACTLY the same as payload vectors (no off-by-one errors).

**Details**:
- Checked 100 random eval samples
- All truth_keys correctly map to payload IDs
- Target vectors match payload vectors exactly (cosine = 1.0)

---

### 4. Eval Provenance

**File**: `artifacts/lvm/LEAKED_EVAL_SETS/eval_v2_payload_aligned.npz`

**Data**:
- Contexts: (5244, 5, 768)
- Targets: (5244, 768)
- Truth keys: (5244, 2) → (article_index, chunk_index)

**Alignment**: Verified at Oct 26, 2025

---

## Training Configuration

**Model**: Mamba-S (53.2M params)

**Loss**: Pure InfoNCE (NO AR loss, NO projection head)
```python
L = -log(exp(cos(pred, target) / τ) / Σ exp(cos(pred, neg) / τ))
```

**Parameters**:
- Temperature τ = 0.07
- Effective batch = 1,024 (256 × 4 grad accum)
- In-batch negatives = 1,023 per positive
- Epochs = 1 (smoke test)
- Device = CPU

**Regularization**:
- Article dropout = 0.2
- Span corruption = 0.1

---

## What's Being Tested

### Hypothesis

**V1 (failed)**:
- InfoNCE on 256D projected space (70%)
- AR cosine on 768D raw space (30%)
- **Conflict**: Split objectives → model learned neither

**Pure InfoNCE (current)**:
- InfoNCE on 768D raw space (100%)
- NO AR loss (removed conflict)
- **Expectation**: Model learns retrieval-aligned geometry

### Success Criteria (Epoch 1)

**Minimum**:
- R@5 > 0% (ANY retrieval)
- Δ (pos - neg) > 0.05 (some separation)

**Good**:
- R@5 ≥ 5% (meaningful signal)
- Δ (pos - neg) ≥ 0.10 (clear separation)

**Strong**:
- R@5 ≥ 10% (strong signal)
- Δ (pos - neg) ≥ 0.15 (wide separation)

---

## Pending Tests (After Epoch 1)

### 1. Pos/Neg Cosine Separation

**Tool**: `tools/inspect_batch_cosines.py`

**Metrics**:
- mean(cos(pred[i], target[i])) → positive cosine
- mean(cos(pred[i], target[j≠i])) → negative cosine
- Δ = pos - neg → separation
- AUC → discriminability

**Expected**:
- If Δ ≥ 0.10: Model learning useful geometry
- If Δ < 0.05: Model not learning task → pivot

### 2. Retrieval Evaluation

**Tool**: `tools/eval_checkpoint_unified.py`

**Settings**:
- Index: IVF with nprobe=64
- Eval: Leaked set (91.5% overlap)
- Top-K: 50

**Gate**: R@5 > 0%

### 3. Flat Index Ablation (If IVF fails)

**Purpose**: Rule out ANN quantization loss

**Method**: Build flat IP index, re-run retrieval

**Expected**: If R@5 jumps, it's an ANN issue (not model)

---

## Auto-Evaluation Script

**When epoch 1 completes**, run:

```bash
./scripts/eval_pure_infonce_epoch1.sh
```

This automatically runs:
1. Pos/neg separation check
2. IVF retrieval evaluation
3. Decision gate (continue or pivot)

---

## Decision Tree

```
Epoch 1 completes
│
├─ R@5 > 0%
│  ├─ R@5 ≥ 5%
│  │  └─ ✅ Continue training (2-10 more epochs)
│  │
│  └─ R@5 < 5%
│     └─ Train 1-2 more epochs, re-check
│
└─ R@5 = 0%
   ├─ Δ (pos-neg) ≥ 0.10
   │  └─ ⚠️  Check FLAT index (may be ANN issue)
   │
   └─ Δ (pos-neg) < 0.10
      └─ ❌ Pivot to Two-Tower (Option 4)
```

---

## Flat Index Backup (If Needed)

If IVF shows R@5=0% but Δ≥0.10:

```bash
# Build flat index
python3 tools/build_faiss.py \
  --vectors artifacts/wikipedia_584k_payload.npy \
  --index artifacts/payload_flat_ip.faiss \
  --metric ip \
  --flat

# Re-run retrieval
KMP_DUPLICATE_LIB_OK=TRUE python3 tools/eval_checkpoint_unified.py \
  --checkpoint artifacts/lvm/models/mamba_s_pure_infonce/best.pt \
  --eval-npz artifacts/lvm/LEAKED_EVAL_SETS/eval_v2_payload_aligned.npz \
  --payload artifacts/wikipedia_584k_payload.npy \
  --faiss artifacts/payload_flat_ip.faiss \
  --device cpu \
  --nprobe 1 \
  --out artifacts/lvm/pure_infonce_epoch1_flat.json
```

If R@5 still 0% on FLAT → model is the issue, not ANN.

---

## Two-Tower Pivot (If Pure InfoNCE Fails)

If epoch 1 shows R@5=0% AND Δ<0.10:

**Conclusion**: Single-tower autoregressive LVM can't learn retrieval geometry.

**Next**: Two-tower architecture (separate query/payload encoders)

```python
# Pseudo-code
Tower Q: encode(context) → 768D query vector
Tower P: encode(chunk) → 768D payload vector
Loss: InfoNCE on cos(Q(ctx), P(target))
```

**Why this works**:
- Removes "one head for generation + retrieval" conflict
- Each tower specializes
- Directly optimizes retrieval objective

---

## Summary Table

| Check | Result | Status |
|-------|--------|--------|
| **Vector norm** | mean=1.000 | ✅ PASS |
| **FAISS metric** | Inner Product | ✅ PASS |
| **Oracle R@5** | 100.0% | ✅ PASS |
| **Eval alignment** | 100.0% | ✅ PASS |
| **Pos/neg separation** | Pending | ⏳ |
| **R@5 (IVF nprobe=64)** | Pending | ⏳ |
| **Decision gate** | Pending | ⏳ |

---

## Files

- **Training log**: `logs/mamba_pure_infonce_20251027_*.log`
- **Checkpoint**: `artifacts/lvm/models/mamba_s_pure_infonce/best.pt`
- **Eval script**: `scripts/eval_pure_infonce_epoch1.sh`
- **Pos/neg tool**: `tools/inspect_batch_cosines.py`

---

## Current Status

- **Training**: ✅ Running (PID 61274, ~3.5 hours)
- **Epoch 1**: ⏳ In progress
- **Plumbing**: ✅ All checks passed
- **Next**: Wait for epoch 1 completion → run eval script

---

**Report Generated**: 2025-10-27 15:00
**Next Check**: Monitor for history.json creation
**ETA**: Unknown (CPU training slow, check every 15 min)
