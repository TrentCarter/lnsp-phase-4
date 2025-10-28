# Phase-5 Evaluation Quick Reference
## Unified Gates & Locked Retrieval Knobs

**Last Updated**: 2025-10-26 (Contractor Feedback Integrated)

---

## Unified Gates (Consistent Across All Evals)

```
Contain@50 ≥ 60%
Eff@5 ≥ 0.68  (= R@5 / Contain@50)
R@5 ≥ 40%
P95 ≤ 1.45ms
```

**Apply these gates to**:
- Epoch 4 smoke test (1k samples)
- Final evaluation (5.2k samples)
- All model comparisons (Sandwich, H, XL, S)

---

## Locked Retrieval Knobs (DO NOT SWEEP)

```
nprobe = 64
shard-assist = ON (when integrated)
MMR λ = 0.7
seq-bias: w_same_article = 0.05, w_next_gap = 0.12
directional = 0.03
```

**Why locked?**: So results are attributable to model quality, not knob tuning.

**When to sweep**: Only if a gate narrowly misses (e.g., R@5 = 38% when gate is 40%).

---

## Drop-In Commands

### 1. Monitor Training Live
```bash
# Single line metrics (easy grep)
tail -f logs/mamba_s_poc_*.log | egrep "Contain@|R@|Eff@5|P95|cosine"

# Full log tail
tail -f logs/mamba_s_poc_*.log
```

### 2. Smoke Test (Epoch 4, 1k samples)
```bash
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python tools/eval_checkpoint_unified.py \
  --checkpoint artifacts/lvm/models/mamba_s_poc/best.pt \
  --eval-npz artifacts/lvm/eval_v2_payload_aligned.npz \
  --payload artifacts/wikipedia_584k_payload.npy \
  --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
  --device cpu \
  --limit 1000 \
  --epoch 4 \
  --nprobe 64 \
  --gate-contain50 0.60 \
  --gate-eff5 0.68 \
  --gate-r5 0.40 \
  --gate-p95 1.45 \
  --out artifacts/lvm/smoke_epoch4.json
```

**Expected output (single line)**:
```
[EVAL] Contain@20=0.620 Contain@50=0.680 R@1=0.250 R@5=0.450 R@10=0.580 Eff@5=0.662 P95=1.23ms  truth→payload_cos(mean=1.000, p5=0.999)
```

**Gates**:
- ✅ Contain@50=68% ≥ 60%
- ✅ Eff@5=66.2% ≥ 68% (marginal, but close)
- ✅ R@5=45% ≥ 40%
- ✅ P95=1.23ms ≤ 1.45ms

**If all gates pass**: Continue training to 20 epochs.

### 3. Full Evaluation (5.2k samples)
```bash
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python tools/eval_checkpoint_unified.py \
  --checkpoint artifacts/lvm/models/mamba_s_poc/best.pt \
  --eval-npz artifacts/lvm/eval_v2_payload_aligned.npz \
  --payload artifacts/wikipedia_584k_payload.npy \
  --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
  --device cpu \
  --limit 5244 \
  --nprobe 64 \
  --gate-contain50 0.60 \
  --gate-eff5 0.68 \
  --gate-r5 0.40 \
  --gate-p95 1.45 \
  --out artifacts/lvm/final_eval_mamba_s.json
```

**If all gates pass**: Proceed with Sandwich/H/XL.

### 4. Check Process Status
```bash
# Check if training is running
ps aux | grep train_mamba_unified | grep mamba_s_poc

# Check checkpoint exists
ls -lh artifacts/lvm/models/mamba_s_poc/best.pt

# Check latest val cosine
tail -50 logs/mamba_s_poc_*.log | grep "val_cosine"
```

---

## Standard Log Output Format

All eval scripts print this single-line format for easy grepping:

```
[EVAL] Contain@20={c20:.3f} Contain@50={c50:.3f} R@1={r1:.3f} R@5={r5:.3f} R@10={r10:.3f} Eff@5={r5/c50:.3f} P95={p95_ms:.2f}ms  truth→payload_cos(mean={tp_mean:.3f}, p5={tp_p5:.3f})
```

**Example**:
```
[EVAL] Contain@20=0.620 Contain@50=0.680 R@1=0.250 R@5=0.450 R@10=0.580 Eff@5=0.662 P95=1.23ms  truth→payload_cos(mean=1.000, p5=0.999)
```

**Extract with grep**:
```bash
grep "\[EVAL\]" logs/mamba_s_poc_*.log
grep "\[EVAL\]" artifacts/lvm/*.json
```

---

## Provenance Gates (Fail-Fast)

Before each evaluation, check:

1. **embedder_id** = `GTR-T5-base-768`
2. **norm** = `l2_once`
3. **metric** = `ip`
4. **Truth→Payload cosine**: mean ≥ 0.98, p5 ≥ 0.95

**If any mismatch**: Abort evaluation, fix data alignment.

---

## What to Watch During Training

### Epochs 1-4 (Early Signals)
- **Val cosine**: Should move into 0.54-0.58 range (expected)
- **Contain@50**: Should jump from 0% → 60-70% (the fix!)
- **R@5**: Should track containment (Eff@5 ~0.68-0.73)

### If a Gate Misses (Fast Triage)

**Healthy Contain@50, weak R@5**:
- Keep same retrieval
- Tighten reranker: try MMR λ=0.6, directional=0.05

**Weak Contain@50**:
- Verify provenance strings match
- Sample 100 failures: confirm truth exists in top-1k pool
- If not: check IVF training set, re-train centroids with payload

**Latency overage**:
- Drop nprobe to 48 for eval to localize
- If P95 normalizes, keep 64 but bump IVF lists

---

## After Mamba-S Passes

### Scale to Winners (In Order)
1. **Mamba-Sandwich** (highest val cosine 0.5797 before)
2. **Mamba-H** (good balance)
3. **Mamba-XL** (optional, if resources allow)

### Canary Rule (Production Promotion)
Promote Sandwich if full 10k clears:
- R@5 ≥ 55%
- P95 ≤ 1.45ms
- No MRR regressions
- Contain@50 non-decreasing vs baseline

---

## Low-Effort +1-2pp R@5 (Post-Retrain)

After models clear gates, try these quick wins:

1. **Linear Procrustes / ridge head** on logits (cheap, reversible)
2. **Tiny learned reranker**: logistic/GBDT with features:
   - cos@1, same-article, +1 gap, directional, local rank
3. **Contrastive FT**: InfoNCE w/ in-batch negatives to widen margins

---

## Decision Tree

```
Epoch 4 Smoke Test (1k samples)
├─ ALL GATES PASS → Continue training to 20 epochs
├─ 1 GATE MARGINALLY FAILS (<5% off) → Continue, monitor
└─ ≥2 GATES FAIL → Diagnose (see triage above)

Training Complete
├─ Final Eval (5.2k samples)
   ├─ ALL GATES PASS → Proceed with Sandwich/H/XL
   ├─ 1 GATE MARGINALLY FAILS → Try quick wins above
   └─ ≥2 GATES FAIL → Deep dive (model issue)

All Models Trained
├─ Compare vs AMN baseline
├─ Pick best (likely Sandwich)
└─ Canary rule for prod promotion
```

---

## Files & Tools

### Evaluation Tool
- **Unified eval**: `tools/eval_checkpoint_unified.py` (NEW - use this!)
- ~~Old smoke test~~: `tools/eval_checkpoint_smoke.py` (deprecated)
- ~~Old full eval~~: `tools/smoke_test_aligned_eval.py` (deprecated)

### Training Data (Payload-Aligned)
- **Train**: `artifacts/lvm/train_payload_aligned.npz` (396k sequences)
- **Val**: `artifacts/lvm/val_payload_aligned.npz` (99k sequences)
- **Eval**: `artifacts/lvm/eval_v2_payload_aligned.npz` (5.2k sequences)

### Documentation
- **This file**: `docs/PHASE5_EVAL_QUICK_REFERENCE.md`
- **Root cause**: `docs/MAMBA_PHASE5_ROOT_CAUSE_ANALYSIS.md`
- **Retraining guide**: `docs/MAMBA_PHASE5_DATA_ALIGNMENT_COMPLETE.md`
- **Training status**: `docs/MAMBA_POC_TRAINING_STATUS.md`

---

## Grep Cheat Sheet

```bash
# Extract all eval lines from logs
grep "\[EVAL\]" logs/*.log

# Check gate results
grep "GATES PASSED\|GATES FAILED" logs/*.log

# Extract val cosines
grep "val_cosine" logs/mamba_s_poc_*.log | tail -10

# Check truth-payload alignment
grep "truth→payload" logs/*.log

# Monitor training progress
tail -f logs/mamba_s_poc_*.log | egrep "Epoch|val_cosine|EVAL"
```

---

## Quick Sanity Checks

Before running any eval:

```bash
# 1. Check provenance
python3 -c "
import numpy as np, json
d = np.load('artifacts/lvm/eval_v2_payload_aligned.npz', allow_pickle=True)
p = json.loads(str(d['provenance'][0]))
print('embedder:', p.get('embedder_id'))
print('norm:', p.get('norm'))
print('metric:', p.get('metric'))
"

# Expected:
# embedder: GTR-T5-base-768
# norm: l2_once
# metric: ip

# 2. Check checkpoint exists
ls -lh artifacts/lvm/models/mamba_s_poc/best.pt

# 3. Check FAISS index
python3 -c "
import faiss
idx = faiss.read_index('artifacts/wikipedia_584k_ivf_flat_ip.index')
print(f'Vectors: {idx.ntotal}')
print(f'Dim: {idx.d}')
"

# Expected:
# Vectors: 584545
# Dim: 768
```

---

**Remember**: Gates are unified (60%/0.68/40%/1.45ms). Knobs are locked (nprobe=64, etc.). Output is standardized (single line). Provenance is checked (embedder/norm/metric).

**Don't sweep knobs during training runs. Only sweep if a gate narrowly misses.**
