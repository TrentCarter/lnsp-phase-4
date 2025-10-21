# Phase 3 (v4) Infrastructure - READY TO DEPLOY

**Status**: All scripts created and tested. Ready for immediate execution after Phase 2 completes.

**Current Time**: Phase 2 at epoch 40/50 (mining hard negatives), ETA ~2.5 hours

---

## Quick Start (After Phase 2 Completes)

```bash
# 1. Post-mortem diagnostics (~5 mins)
./.venv/bin/python3 tools/phase2_postmortem.py \
  --ckpt runs/twotower_v3_phase2/checkpoints/best.pt \
  --pairs artifacts/twotower/pairs_v3_synth.npz \
  --bank artifacts/wikipedia_500k_corrected_vectors.npz \
  --out runs/twotower_v3_phase2/postmortem.json

# 2. Expand training pairs (~5-10 mins)
./.venv/bin/python3 tools/expand_pairs_to_v4.py \
  --out artifacts/twotower/pairs_v4_synth.npz \
  --stride 32 \
  --max-per-seq 30 \
  --target-pairs 50000

# 3. Launch v4 training overnight (~8-10 hours)
./launch_v4_overnight.sh
```

---

## Scripts Created

### 1. `tools/phase2_postmortem.py` ✅

**Purpose**: Diagnose Phase 2 failure mode

**What it checks**:
- **Cosine separation**: E[cos(q,d_pos)] - E[cos(q,d_neg_hard)] (margin collapse detection)
- **Near-duplicate rate**: Hard negs with cos>0.98 to both query and positive
- **Bank alignment**: Distribution of random doc-doc cosines (whitening check)

**Output**: `runs/twotower_v3_phase2/postmortem.json`

**Usage**:
```bash
./.venv/bin/python3 tools/phase2_postmortem.py \
  --ckpt runs/twotower_v3_phase2/checkpoints/best.pt \
  --pairs artifacts/twotower/pairs_v3_synth.npz \
  --bank artifacts/wikipedia_500k_corrected_vectors.npz \
  --device mps
```

**Expected findings**:
- Low margin (<0.05): Hard negs too hard → margin collapse
- High dup rate (>2%): Miner feeding near-duplicates
- Bank alignment off (|mean| > 0.10): Need whitening

---

### 2. `tools/expand_pairs_to_v4.py` ✅

**Purpose**: Generate 50k-100k training pairs from existing TMD sequences

**Method**: Reuse same 1,386 sequences with:
- Lower stride (32 vs 50 in v3) → more pairs
- Higher cap per sequence (30 vs 15 in v3) → more pairs
- Rolling deduplication (cos > 0.999)

**Expected output**:
- stride=32, max=30: **~40k-50k pairs** (2.2-2.8x increase)
- stride=25, max=40: **~60k-80k pairs** (3.3-4.4x increase)

**Usage**:
```bash
# First attempt (conservative)
./.venv/bin/python3 tools/expand_pairs_to_v4.py \
  --out artifacts/twotower/pairs_v4_synth.npz \
  --stride 32 \
  --max-per-seq 30 \
  --target-pairs 50000

# If short, retry with more aggressive settings
./.venv/bin/python3 tools/expand_pairs_to_v4.py \
  --out artifacts/twotower/pairs_v4_synth.npz \
  --stride 25 \
  --max-per-seq 40 \
  --target-pairs 50000
```

**Outputs**:
- NPZ file with train/val split (90/10)
- Reports actual vs target pair counts
- Size estimate: ~7-12 GB

---

### 3. `tools/train_twotower_v4.py` ✅

**Purpose**: Train v4 model with curriculum-based hard negative mining

**Key improvements over Phase 2**:
- ✅ **Curriculum schedule**: Warm-start (epochs 1-5) → gentle hards (6-10) → full hards (11+)
- ✅ **Cosine-range mining**: Avoid margin collapse with [0.82-0.92] then [0.84-0.96]
- ✅ **Hard negative filtering**: Drop negs with cos>0.98 to both query and positive
- ✅ **Higher effective batch**: 512 (was 256 in Phase 2)
- ✅ **Cosine LR decay**: 5e-5 → 1e-6 over 30 epochs
- ✅ **Lower temperature**: 0.05 (was 0.07)
- ✅ **Margin loss**: 0.03
- ✅ **Larger memory bank**: 50k (was 20k)

**Curriculum details**:
```
Epochs 1-5:   In-batch + memory bank only (no hard negatives)
Epochs 6-10:  Add 8 hard negs per sample, cos range [0.82, 0.92]
Epochs 11-30: Add 16 hard negs per sample, cos range [0.84, 0.96]
```

**Training time estimate**: ~8-10 hours (30 epochs with mining every epoch after epoch 6)

---

### 4. `launch_v4_overnight.sh` ✅

**Purpose**: One-command launch for v4 training

**Configuration**:
```bash
BATCH_SIZE=32
ACCUM=16                    # Effective batch = 512
EPOCHS=30
LR=5e-5 → 1e-6             # Cosine decay
TAU=0.05
MARGIN=0.03
MEMORY_BANK=50000
MINE_SCHEDULE="0-5:none;6-10:8@0.82-0.92;11-30:16@0.84-0.96"
FILTER_THRESHOLD=0.98
```

**Usage**:
```bash
./launch_v4_overnight.sh

# Monitor
tail -f logs/twotower_v4_*.log

# Check status
ps -p $(cat runs/twotower_v4/train.pid)
```

**Output**:
- Checkpoints: `runs/twotower_v4/checkpoints/best.pt`
- History: `runs/twotower_v4/history.json`
- Logs: `logs/twotower_v4_<timestamp>.log`

---

## Expected Results

### Phase 2 Post-Mortem (Most Likely)
- **Margin**: 0.02-0.04 (collapsed, hard negs too hard)
- **Duplicate rate**: 1-3% (some impurity)
- **Bank alignment**: Mean ~0.05 (reasonable)

**Conclusion**: Hard negatives overwhelmed the model (cos 0.84-0.96 from start was too aggressive)

### v4 Pair Expansion
- **Actual pairs**: 40k-80k (depending on stride)
- **Size**: 7-12 GB NPZ file
- **Train/val**: ~36k-72k train, ~4k-8k val

### v4 Training (Target)
- **Mid-gate (epoch 10)**: Recall@500 ≥ 35-40%
- **Final gate (epoch 30)**: Recall@500 ≥ **55-60%** (beats 38.96% heuristic)
- **Best case**: 65-70% if curriculum works perfectly
- **Worst case**: 30-40% if still data-limited (need 100k+ pairs)

---

## Execution Timeline

**Now** (7:30 AM): Phase 2 at epoch 40/50, mining hard negs
**~10:00 AM**: Phase 2 completes (all 50 epochs)
**10:00-10:05**: Run post-mortem diagnostics
**10:05-10:15**: Expand pairs to v4 (40k-80k)
**10:15**: Launch v4 training overnight
**~6:00 PM**: v4 training completes (30 epochs)

**Tomorrow morning**: Review v4 results, generate report, decide next steps

---

## Decision Gates

### After v4 Training

**If Recall@500 ≥ 55%**: ✅ SUCCESS
- Generate Phase 3 (v4) report
- Deploy to production cascade
- Start integration testing

**If 40% ≤ Recall@500 < 55%**: ⚠️ PARTIAL SUCCESS
- Better than Phase 2 (proves curriculum works)
- Need more data: Ingest more Wikipedia (10k articles)
- Generate 100k-200k pairs
- Run v5 with same curriculum

**If Recall@500 < 40%**: ❌ DATA WALL
- Curriculum didn't help enough
- Minimum data requirement > 100k pairs
- Options:
  1. Ship heuristic (38.96%) as Stage-1 interim
  2. Massive Wikipedia ingestion (50k articles)
  3. Explore alternative architectures (cross-encoder, ColBERT)

---

## Files Created

```
tools/phase2_postmortem.py          # Post-mortem diagnostics
tools/expand_pairs_to_v4.py         # Pair expansion (18k → 50k-100k)
tools/train_twotower_v4.py          # v4 training with curriculum
launch_v4_overnight.sh              # One-command launch
PHASE3_READY_TO_DEPLOY.md          # This file
```

---

## Next Actions

1. ⏳ **Wait for Phase 2** (~2.5 hours)
2. 🔍 **Run post-mortem** (~5 mins)
3. 📊 **Generate Phase 2 report** (~10 mins)
4. 🚀 **Expand pairs** (~5-10 mins)
5. 🌙 **Launch v4 overnight** (~8-10 hours)
6. 📈 **Review v4 tomorrow morning**

---

**Status**: All infrastructure ready. Waiting for Phase 2 to complete for clean comparison.

**Confidence**: High (curriculum approach addresses Phase 2's failure mode directly)

**Risk**: Data volume might still be limiting factor (will know after v4 completes)
