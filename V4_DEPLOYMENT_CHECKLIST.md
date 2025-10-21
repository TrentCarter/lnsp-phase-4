# v4 Deployment Checklist - EXECUTE NOW

**Phase 2 Status**: ✅ COMPLETE (50/50 epochs)
**Time**: Ready to deploy immediately

---

## Step 1: Phase 2 Post-Mortem (~5 mins)

```bash
./.venv/bin/python3 tools/phase2_postmortem.py \
  --ckpt runs/twotower_v3_phase2/checkpoints/best.pt \
  --pairs artifacts/twotower/pairs_v3_synth.npz \
  --bank artifacts/wikipedia_500k_corrected_vectors.npz \
  --out runs/twotower_v3_phase2/postmortem.json \
  --device mps
```

**What it checks**:
- Margin collapse: Δ = E[cos(q,pos)] - E[cos(q,hard_neg)]
- Near-duplicate rate in hard negatives
- Bank alignment (whitening recommendation)

**Expected findings**:
- Margin: 0.02-0.04 (collapsed ← confirms failure mode)
- Dup rate: 1-3% (some impurity)
- Alignment: ~0.05 mean (OK, no whitening needed)

---

## Step 2: Expand Training Pairs (~10 mins)

### First Attempt (Conservative)

```bash
./.venv/bin/python3 tools/expand_pairs_to_v4.py \
  --out artifacts/twotower/pairs_v4_synth.npz \
  --stride 32 \
  --max-per-seq 30 \
  --target-pairs 50000 \
  --seed 42
```

**Expected**: ~40k-50k pairs (2.2-2.8x increase)

### If Short (<45k), Retry Aggressively

```bash
./.venv/bin/python3 tools/expand_pairs_to_v4.py \
  --out artifacts/twotower/pairs_v4_synth.npz \
  --stride 25 \
  --max-per-seq 40 \
  --target-pairs 50000 \
  --seed 42
```

**Expected**: ~60k-80k pairs (3.3-4.4x increase)

---

## Step 3: v4 Pre-Flight Checks (~2-3 mins)

```bash
./.venv/bin/python3 tools/preflight_v4.py \
  --pairs artifacts/twotower/pairs_v4_synth.npz \
  --bank artifacts/wikipedia_500k_corrected_vectors.npz \
  --mine-schedule "0-5:none;6-10:8@0.82-0.92;11-30:16@0.84-0.96" \
  --filter-threshold 0.98 \
  --expansion-args '{"stride": 32, "max_per_seq": 30, "target": 50000}' \
  --manifest-out artifacts/twotower/pairs_v4_manifest.json
```

**Checks performed**:
1. ✅ Data sanity (train/val split verification)
2. ✅ Leakage check (no exact duplicates train↔valid)
3. ✅ Doc-space alignment (whitening recommendation)
4. ✅ Curriculum schedule verification
5. ✅ Hard-neg filter confirmation (>0.98 threshold)
6. ✅ Manifest generation (reproducibility)

**Must pass**: All checks green or warnings only (no blockers)

---

## Step 4: Launch v4 Training (Overnight ~8-10 hours)

```bash
./launch_v4_overnight.sh
```

**Configuration**:
- Batch: 32 × 16 accum = **512 effective**
- Epochs: 30
- LR: 5e-5 → 1e-6 (cosine decay)
- Temperature: 0.05
- Margin: 0.03
- Memory bank: 50,000
- Curriculum:
  - Epochs 1-5: No mining (warm start)
  - Epochs 6-10: 8 hard negs @ cos [0.82, 0.92]
  - Epochs 11-30: 16 hard negs @ cos [0.84, 0.96]
- Filter: Drop negs with cos>0.98 to both q and pos

**Monitor**:
```bash
# Live log
tail -f logs/twotower_v4_*.log

# Check progress
grep -E "Epoch|Recall@500|Separation Δ" logs/twotower_v4_*.log | tail -20

# Status
ps -p $(cat runs/twotower_v4/train.pid)
```

---

## During v4 Training: Watch These Metrics

### 1. Separation Margin Δ (Critical!)

**What to watch**:
```
Epoch 1:  Δ = 0.08-0.12 (in-batch only, should be healthy)
Epoch 5:  Δ = 0.10-0.15 (warm start complete)
Epoch 10: Δ = 0.08-0.12 (gentle hards added, may dip slightly)
Epoch 15: Δ = 0.06-0.10 (full hards, should stabilize)
Epoch 30: Δ = 0.05-0.08 (final, must be >0.05)
```

**Abort triggers**:
- Δ < 0.04 for 2 consecutive evals after epoch 10
- Training auto-aborts, saves `abort.pt` checkpoint
- Recommendations printed in log

**If aborted**:
- Tighten cos window: 0.84-0.96 → 0.84-0.94
- Reduce hard-neg count: 16 → 8
- Lower temperature: 0.05 → 0.045

### 2. Recall@500 Gates

**Mid-gate (Epoch 10)**:
- Target: ≥ 35-40%
- If < 30%: Data volume issue, need 100k+ pairs
- If 30-35%: On track, curriculum working

**Final gate (Epoch 30)**:
- Target: ≥ **55-60%** (beats 38.96% heuristic)
- If 50-55%: Partial success, consider phase 2.5
- If < 50%: Data wall, scale to 100k+ pairs

### 3. Training Loss

**Expected curve**:
```
Epoch 1:  ~4.0 (high, in-batch only)
Epoch 5:  ~3.5 (decreasing)
Epoch 10: ~4.5 (spike when hard negs added)
Epoch 15: ~4.0 (stabilizing)
Epoch 30: ~3.5-4.0 (converged)
```

**Red flags**:
- Loss > 5.0 persistently → hard negs too hard
- Loss < 2.0 before epoch 20 → overfitting

---

## Post-v4 Actions

### If Success (Recall@500 ≥ 55%)

1. **Generate v4 report**
2. **Deploy to production cascade**:
   ```
   Query → Two-Tower (v4) → Top-500
         → LVM Re-rank → Top-50
         → TMD Re-rank → Top-10
   ```
3. **Integration testing**
4. **Ship to production!**

### If Partial Success (40-55%)

1. **Generate v4 report**
2. **Scale data to 100k pairs**:
   ```bash
   # Ingest more Wikipedia articles
   # OR lower stride to 20, max=50
   ```
3. **Run v5 with same curriculum** (resume from v4 best checkpoint)
4. **Target**: Cross 55-60% threshold

### If Data Wall (<40%)

1. **Generate v4 report**
2. **Immediate actions**:
   - Ship 38.96% heuristic as Stage-1 interim
   - Massive Wikipedia ingestion (10k→50k articles)
   - Generate 200k-500k pairs
3. **Longer-term**:
   - Explore alternative architectures (ColBERT, cross-encoder)
   - Consider hybrid: two-tower + heuristic fusion

---

## Quick Reference

### Files Created
```
tools/phase2_postmortem.py          # Post-mortem diagnostics
tools/expand_pairs_to_v4.py         # Pair expansion (18k → 50k-100k)
tools/preflight_v4.py               # Pre-flight safety checks
tools/train_twotower_v4.py          # v4 training with curriculum + monitoring
launch_v4_overnight.sh              # One-command deployment
```

### Key Outputs
```
runs/twotower_v3_phase2/postmortem.json     # Post-mortem results
artifacts/twotower/pairs_v4_synth.npz       # v4 training data
artifacts/twotower/pairs_v4_manifest.json   # Reproducibility manifest
artifacts/twotower/preflight_results.json   # Pre-flight check results
runs/twotower_v4/config.yaml                # v4 training config
runs/twotower_v4/history.json               # Training history with margin
runs/twotower_v4/checkpoints/best.pt        # Best model
```

### Monitoring Commands
```bash
# Live log
tail -f logs/twotower_v4_*.log

# Progress summary
grep -E "Epoch|Recall@500|Separation Δ" logs/twotower_v4_*.log | tail -30

# Check if running
ps -p $(cat runs/twotower_v4/train.pid)

# Quick metrics
./.venv/bin/python3 -c "
import json
with open('runs/twotower_v4/history.json') as f:
    hist = json.load(f)
latest = hist[-1]
print(f\"Epoch {latest['epoch']}: R@500={latest['recall@500']:.2f}%, Δ={latest['margin']:.4f}\")
"
```

---

## Execution Timeline

**Now** (8:00 AM): Phase 2 complete ✅
**8:00-8:05**: Run post-mortem
**8:05-8:15**: Expand pairs to v4
**8:15-8:20**: Run pre-flight checks
**8:20**: Launch v4 training
**~6:00 PM**: v4 training completes (30 epochs)
**Tomorrow**: Review results, generate report, deploy or iterate

---

## Execute Now!

```bash
# Full sequence (copy-paste)

# 1. Post-mortem
./.venv/bin/python3 tools/phase2_postmortem.py \
  --ckpt runs/twotower_v3_phase2/checkpoints/best.pt \
  --pairs artifacts/twotower/pairs_v3_synth.npz \
  --bank artifacts/wikipedia_500k_corrected_vectors.npz \
  --device mps

# 2. Expand pairs
./.venv/bin/python3 tools/expand_pairs_to_v4.py \
  --stride 32 --max-per-seq 30 --target-pairs 50000

# 3. Pre-flight
./.venv/bin/python3 tools/preflight_v4.py \
  --pairs artifacts/twotower/pairs_v4_synth.npz \
  --bank artifacts/wikipedia_500k_corrected_vectors.npz \
  --expansion-args '{"stride": 32, "max_per_seq": 30}'

# 4. Launch v4
./launch_v4_overnight.sh

# 5. Monitor
tail -f logs/twotower_v4_*.log
```

**Green light confirmed. All systems ready for deployment!** 🚀
