# Epoch-2 Triage Playbook
## Fast If/Then Decision Tree for Contrastive Training

**Status**: 🔄 Training + monitoring active
**Created**: 2025-10-27
**Training PID**: 48960 (Mamba-S contrastive)
**Monitor PID**: 51213 (epoch 2 gate checker, runs every 2min)
**AMN Baseline PID**: 51264 (running in parallel for comparison)

---

## Critical Gate (Epoch 2)

**Target**: `val_cosine ≥ 0.50` (was 0.22 with AR-only)

**What it means**:
- ✅ **≥0.50**: Contrastive learning is working! Model learns global GTR-T5 semantics
- ❌ **<0.50**: InfoNCE not effective, apply triage fixes below

**Check automatically** (runs every 2 minutes):
```bash
# Monitor logs
tail -f logs/epoch2_monitor.log

# Or check manually
python3 tools/check_contrastive_sanity.py --epoch 2
```

---

## Sanity Checks (After Epoch 2)

### Check A: No Collapse (Margin Test)
**Expected**: `mean(sim(ŷ,y)) >> mean(sim(ŷ, negatives))` by ≥0.10 margin

**Why**: Confirms model isn't collapsing (all vectors becoming identical)

### Check B: Batch Hygiene (Same-Article Negatives)
**Expected**: Zero same-article positives in negatives set

**Why**: InfoNCE negatives should be from different articles/topics

---

## Decision Tree

### IF Epoch-2 PASSES (val_cosine ≥ 0.50)

✅ **Action**: Lock knobs, continue training

**Next steps**:
1. Continue to Epoch 4
2. Run 1k smoke eval with locked knobs (see command below)
3. Compare against AMN baseline (already running in background!)
4. Log top-K failure modes (sample 100 misses)
5. Optional: Add tiny ridge head for +1-2pp R@5 boost

**Smoke eval at Epoch 4**:
```bash
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python tools/eval_checkpoint_unified.py \
  --checkpoint artifacts/lvm/models/mamba_s_contrastive/best.pt \
  --eval-npz artifacts/lvm/eval_v2_payload_aligned.npz \
  --payload artifacts/wikipedia_584k_payload.npy \
  --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
  --device cpu \
  --limit 1000 \
  --nprobe 64 \
  --gate-contain50 0.60 \
  --gate-eff5 0.68 \
  --gate-r5 0.40 \
  --gate-p95 1.45 \
  --out artifacts/lvm/smoke_contrastive_epoch4.json
```

**Expected results** (after smoke eval):
- Contain@50 ≥ 60% (was 0% with AR-only!)
- R@5 ≥ 40% (was 0%!)
- Eff@5 ≥ 0.68
- P95 ≤ 1.45ms

---

### IF Epoch-2 MISSES (val_cosine < 0.50)

❌ **Action**: Apply triage fixes (in order, each is cheap)

#### Fix 1: Temperature Sweep
**Current**: τ = 0.07
**Try**: τ = 0.05, 0.10

```bash
# Restart with τ=0.05
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python app/lvm/train_mamba_contrastive.py \
  --model-type mamba_s \
  --train-npz artifacts/lvm/train_payload_aligned.npz \
  --val-split 0.2 \
  --temperature 0.05 \
  --lambda-con 0.7 --lambda-ar 0.3 \
  [... other args ...]
```

**Why**: Lower τ → sharper distribution → more discriminative learning

---

#### Fix 2: Lambda Sweep (Reduce AR Influence)
**Current**: λ_con=0.7, λ_ar=0.3
**Try**: λ_con=0.85, λ_ar=0.15

```bash
# Restart with stronger contrastive weight
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python app/lvm/train_mamba_contrastive.py \
  --model-type mamba_s \
  --train-npz artifacts/lvm/train_payload_aligned.npz \
  --val-split 0.2 \
  --lambda-con 0.85 --lambda-ar 0.15 \
  [... other args ...]
```

**Why**: AR cosine pulls toward memorization; reduce its influence

---

#### Fix 3: Increase Effective Batch (More Negatives)
**Current**: 256 × 4 = 1024
**Try**: 512 × 4 = 2048

```bash
# Restart with larger effective batch
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python app/lvm/train_mamba_contrastive.py \
  --model-type mamba_s \
  --train-npz artifacts/lvm/train_payload_aligned.npz \
  --val-split 0.2 \
  --batch-size 512 \
  --grad-accum-steps 4 \
  [... other args ...]
```

**Why**: More in-batch negatives → harder contrastive task → better global learning

**Alternative**: Add 65k FIFO queue with ≥30% same-topic / different-article negatives

---

#### Fix 4: Projection Head Check
**Verify**:
- L2-norm after LayerNorm ✓
- No bias on final projection layer ✓

```python
# Check current implementation
class ProjectionHead(nn.Module):
    def __init__(self, d_model=768, hidden_dim=512, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=True),   # ✓ bias OK here
            nn.GELU(),
            nn.LayerNorm(hidden_dim),                    # ✓ norm before final
            nn.Linear(hidden_dim, out_dim, bias=False),  # ✓ NO bias on final!
        )

    def forward(self, x):
        h = self.mlp(x)
        return F.normalize(h, p=2, dim=-1)  # ✓ L2 normalize output
```

---

#### Fix 5: Batch Mixing (Cross-Article Negatives)
**Current**: Random sampling
**Enforce**: Cap ≤2 samples per article per batch

**Why**: Ensures diverse negatives (different articles/topics)

---

## Automated Monitoring

### Training Monitor (Every 2 minutes)
```bash
# Already running (PID 51213)
tail -f logs/epoch2_monitor.log
```

**Output when Epoch 2 completes**:
```
==========================================
EPOCH 2 COMPLETE - RUNNING GATE CHECK
==========================================
Val cosine: 0.5234
✅ GATE PASSED: val_cosine ≥ 0.50
```

### AMN Baseline (Running in Parallel)
```bash
# Already running (PID 51264)
tail -f logs/amn_baseline_*.log

# Results will be saved to:
# artifacts/lvm/amn_baseline_eval.json
```

**Purpose**: Bounds Mamba performance
- If AMN fails → data/retrieval issue (not model)
- If AMN passes, Mamba fails → model/objective issue
- Mamba should approach AMN with contrastive learning

---

## Expected Timeline

**Now**: Training Epoch 1 (loading data + first epoch ~30 min)
**+40 min**: Epoch 2 completes → **CRITICAL GATE CHECK**
**+60 min**: Epoch 4 completes → smoke eval (if gate passed)
**+4-5 hours**: Training complete → full evaluation

---

## Quick Status Checks

### Check Training Progress
```bash
# View training log
tail -f logs/mamba_s_contrastive_direct_*.log

# Check current epoch
python3 -c "
import json
try:
    with open('artifacts/lvm/models/mamba_s_contrastive/history.json') as f:
        history = json.load(f)
        print(f'Completed: {len(history)} epochs')
        if history:
            latest = history[-1]
            print(f'Latest val_cosine: {latest[\"val_cosine\"]:.4f}')
except FileNotFoundError:
    print('Training still initializing...')
"

# Verify process running
ps aux | grep 48960 | grep -v grep
```

### Check AMN Baseline Progress
```bash
# View AMN log
tail -f logs/amn_baseline_*.log

# Check results (when complete)
cat artifacts/lvm/amn_baseline_eval.json
```

---

## Post-Training Optimizations (Drop-In)

After training completes and passes gates, apply these for +1-2pp R@5:

### 1. Ridge/Procrustes Alignment Head
**Offline fit** (no retraining):
```bash
# Fit linear alignment head on validation set
python tools/fit_alignment_head.py \
  --checkpoint artifacts/lvm/models/mamba_s_contrastive/best.pt \
  --val-npz artifacts/lvm/val_payload_aligned.npz \
  --out artifacts/lvm/models/mamba_s_contrastive/alignment_head.pt
```

**Expected boost**: +1-2pp R@5 (cheap, reversible)

---

### 2. Mini Learned Reranker
**Logistic/GBDT** over features:
- cos@1
- local rank
- same-article bonus
- +1 gap bonus
- directional alignment

```bash
# Train tiny reranker (on top-K pool)
python tools/train_mini_reranker.py \
  --checkpoint artifacts/lvm/models/mamba_s_contrastive/best.pt \
  --eval-npz artifacts/lvm/eval_v2_payload_aligned.npz \
  --payload artifacts/wikipedia_584k_payload.npy \
  --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
  --out artifacts/lvm/models/mamba_s_contrastive/reranker.pkl
```

**Expected boost**: +1-2pp R@5 without touching retrieval

---

## Failure Signatures & Fixes

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| **High Contain@50, low R@5** | Ranking issue | Tweak reranker (λ=0.6, directional=0.05) or enable mini reranker |
| **Low Contain@50** | Retrieval gap | Verify vector-set strings; re-train IVF centroids with payload; tune nlist/nprobe |
| **Val cosine oscillates 0.40-0.48** | Too few/easy negatives | Increase batch/queue; raise hard-negative rate |
| **Eval cosine << val cosine** | Overfitting to val articles | More article dropout; cross-article batch mixing |
| **Training unstable (loss NaN)** | Gradient explosion | Lower LR; check gradient clipping (max_norm=1.0) |

---

## Files & Tools

### Training
- **Script**: `app/lvm/train_mamba_contrastive.py`
- **Process**: PID 48960
- **Log**: `logs/mamba_s_contrastive_direct_20251027_*.log`
- **Checkpoint**: `artifacts/lvm/models/mamba_s_contrastive/best.pt`
- **History**: `artifacts/lvm/models/mamba_s_contrastive/history.json`

### Monitoring
- **Epoch 2 gate**: `tools/check_contrastive_sanity.py`
- **Monitor script**: `scripts/monitor_epoch2_gate.sh` (PID 51213)
- **Monitor log**: `logs/epoch2_monitor.log`

### Evaluation
- **Unified eval**: `tools/eval_checkpoint_unified.py`
- **AMN baseline**: `tools/eval_amn_baseline.py` (PID 51264)
- **AMN results**: `artifacts/lvm/amn_baseline_eval.json` (when complete)

### Documentation
- **This playbook**: `docs/EPOCH2_TRIAGE_PLAYBOOK.md`
- **Training status**: `docs/MAMBA_CONTRASTIVE_TRAINING_STATUS.md`
- **Eval reference**: `docs/PHASE5_EVAL_QUICK_REFERENCE.md`
- **Root cause analysis**: `docs/MAMBA_PHASE5_ROOT_CAUSE_ANALYSIS.md`

---

## Grep Cheat Sheet

```bash
# Extract epoch 2 val_cosine
grep "epoch.*2" logs/mamba_s_contrastive_direct_*.log | grep val_cosine

# Check if gate passed
tail -50 logs/epoch2_monitor.log | grep "GATE"

# AMN baseline results
grep "\[EVAL\]" logs/amn_baseline_*.log

# Compare Mamba vs AMN
echo "=== Mamba Contrastive ===" && grep "\[EVAL\]" artifacts/lvm/smoke_contrastive_epoch4.json
echo "=== AMN Baseline ===" && jq -r '.metrics' artifacts/lvm/amn_baseline_eval.json
```

---

**Remember**:
- **Lock knobs** during training (no sweeping unless gate narrowly misses)
- **Check margin/hygiene** at epoch 2 (confirms contrastive is healthy)
- **Compare vs AMN** to bound expectations (AMN = upper bound)
- **Apply triage in order** if epoch 2 misses (τ → λ → batch → head → mixing)
- **Add drop-in optimizations** post-training for easy +2-4pp R@5

---

**Last Updated**: 2025-10-27 09:05
**Training ETA**: ~4-5 hours
**Epoch 2 ETA**: ~40 minutes from start (check monitor log!)
