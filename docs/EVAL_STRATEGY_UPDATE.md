# Evaluation Strategy Update
## After Data Leak Discovery

**Date**: 2025-10-27
**Issue**: All eval sets have 50-90% article/chunk overlap with training

---

## What We Discovered

### Sanity Check Results
```
Original "aligned" eval (eval_v2_payload_aligned.npz):
  - 91.5% article overlap
  - 74% chunk overlap
  - Query → Truth cosine: 1.000 (oracle!)
  - AMN baseline: 100% R@5 (invalid)
```

### Why This Happened
1. Training data uses Wikipedia articles 0-6100 (6101 articles)
2. Payload has 8447 articles total, but most are in training
3. Held-out articles (728) have duplicate chunks → can't build continuous sequences
4. All existing eval sets reuse training articles (different chunks)

---

## Evaluation Strategy Going Forward

### Phase 1: Contrastive Training Continues ✅
**Current**: Mamba-S with InfoNCE + AR (PID 48960)

**Why it's still valid**:
- Contrastive learning fixes the OBJECTIVE problem (memorization)
- InfoNCE forces global semantics, not episode-specific patterns
- Should improve on ANY eval set (overlapping or not)

###Phase 2: Multi-Level Evaluation

#### Level 1: Within-Article Generalization (Current Eval Sets)
**Use**: `eval_v2_ready.npz` or `wikipedia_ood_test_ctx5_legacy_eval.npz`

**Tests**: Can model predict next chunk in seen articles?
- **AR-only**: Memorizes training chunk sequences → may work
- **InfoNCE + AR**: Learns global semantics → should work BETTER

**Gates** (same as before):
- Val cosine ≥ 0.50 (was 0.22 with AR-only)
- Contain@50 ≥ 60%
- R@5 ≥ 40%
- Eff@5 ≥ 0.68

#### Level 2: Cross-Article Generalization (Gold Standard)
**Create**: Held-out Wikipedia articles (need fresh ingestion)

**Tests**: Can model predict in TRULY unseen articles?
- **AR-only**: Fails completely (0% R@5) ✓ observed
- **InfoNCE + AR**: Should approach Level 1 performance

**How to get this**:
1. Ingest 100-200 NEW Wikipedia articles (not in current payload)
2. Build eval sequences from these
3. Compare Mamba vs AMN on truly unseen articles

#### Level 3: Cross-Topic Generalization (Stretch)
**Test**: Predict in different Wikipedia topics (science → history → tech)

---

## Expected Results Matrix

| Model | Level 1 (Within-Article) | Level 2 (Cross-Article) | Level 3 (Cross-Topic) |
|-------|-------------------------|------------------------|---------------------|
| **AR-only** | 40-60% R@5? | 0% R@5 ✓ | 0% R@5 |
| **InfoNCE + AR** | 50-70% R@5 | 40-60% R@5 | 30-50% R@5 |
| **AMN (two-tower)** | 70-90% R@5 | 60-80% R@5 | 50-70% R@5 |

---

## Immediate Actions

### Now (Training Running)
1. ✅ Continue Mamba-S contrastive training
2. ✅ Monitor epoch 2 gate (val_cosine ≥ 0.50)
3. ⏳ Wait for training to complete (~4 hours)

### After Epoch 2 (IF Gate Passes)
```bash
# Eval on best available eval set (Level 1)
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python tools/eval_checkpoint_unified.py \
  --checkpoint artifacts/lvm/models/mamba_s_contrastive/best.pt \
  --eval-npz artifacts/lvm/wikipedia_ood_test_ctx5_legacy_eval.npz \
  --payload artifacts/wikipedia_584k_payload.npy \
  --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
  --device cpu \
  --limit 7140 \
  --nprobe 64 \
  --gate-contain50 0.60 \
  --gate-eff5 0.68 \
  --gate-r5 0.40 \
  --gate-p95 1.45 \
  --out artifacts/lvm/mamba_s_contrastive_level1_eval.json
```

**Expected** (Level 1):
- ✅ Val cosine: 0.50-0.58 (NOT 0.22!)
- ✅ R@5: 40-60% (NOT 0%!)
- Proves contrastive learning is working

### After Training Complete
1. **If Level 1 passes** → Ingest fresh Wikipedia articles for Level 2
2. **If Level 1 fails** → Apply triage (τ, λ, batch size)
3. **Compare vs AR-only** → Quantify improvement from contrastive learning

---

## Why Contrastive Learning Still Fixes The Problem

### AR-Only (Episode Memorization)
```
Training: "Article 1234, chunks 10→11→12"
→ Learns: "If I see 1234:10, 1234:11, predict pattern specific to article 1234"
→ Fails on: ANY unseen pattern (even unseen chunks in seen articles!)
```

### InfoNCE + AR (Global Semantics)
```
Training: "ŷ must be closer to true next than to 1000+ in-batch negatives"
→ Learns: Global GTR-T5 semantic geometry
→ Works on: Unseen chunks, unseen articles, unseen topics (with degradation)
```

Even with eval overlap, InfoNCE forces the model to learn transferable representations instead of memorizing article-specific shortcuts.

---

## Documentation Updates Needed

1. **MAMBA_POC_TRAINING_STATUS.md** - Update expected results for Level 1 eval
2. **EPOCH2_TRIAGE_PLAYBOOK.md** - Update eval commands to use OOD test set
3. **PHASE5_EVAL_QUICK_REFERENCE.md** - Add multi-level evaluation strategy

---

## Long-Term Fix (Post-Training)

### Option A: Ingest Fresh Wikipedia (Recommended)
```bash
# Ingest 200 new Wikipedia articles (not in current payload)
# Use articles 10000-10200 (way beyond training set)
./scripts/ingest_wikipedia_fresh.sh --start 10000 --limit 200
```

### Option B: Use Different Corpus
- Scientific papers (arXiv)
- News articles (different domain)
- Technical documentation

### Option C: Synthetic OOD Test
- Generate synthetic sequences using LLM
- Embed with GTR-T5
- Test if Mamba can predict next vector

---

## Key Insight

**The contractor's diagnosis was correct**:
- Problem: AR cosine → episode memorization
- Solution: InfoNCE + AR → global semantics

**Data overlap doesn't invalidate the fix**:
- InfoNCE prevents memorization even on overlapping eval
- If model still memorizes → contrastive weight too low (increase λ_con)
- If model generalizes → contrastive learning is working!

---

## Success Criteria (Updated)

### Minimum (Level 1)
- Epoch 2: val_cosine ≥ 0.50 (proves contrastive is active)
- Final: R@5 ≥ 40% on OOD test set (proves better than AR-only 0%)

### Target (Level 2)
- R@5 ≥ 40% on fresh Wikipedia articles (true generalization)

### Stretch (Level 3)
- R@5 ≥ 30% on cross-domain corpus (arXiv papers)

---

**Bottom Line**: Training continues as planned. Eval overlap is a data issue, not a fatal flaw. Contrastive learning addresses the objective problem and should show improvement regardless.
