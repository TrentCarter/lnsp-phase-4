# 🚀 Phase-3.5 Plan - 2000-Vector Context (40K Tokens)

**Date**: 2025-10-19 (Evening, 9:22 PM - 10:08 PM)
**Status**: ❌ **FAILED - Data Scarcity Issue Identified**

---

## 🔴 ACTUAL RESULTS - PHASE-3.5 FAILED

**Training completed**: 10:08 PM (Oct 19, 2025)
**Result**: 62.07% Hit@5 (**-13.58% from Phase-3!**)

| Metric | Phase-3 (1000-ctx) | Phase-3.5 (2000-ctx) | Change | Status |
|--------|-------------------|---------------------|--------|--------|
| Hit@1 | 61.74% | **44.83%** | **-16.91%** | ❌ Major regression |
| Hit@5 | 75.65% | **62.07%** | **-13.58%** | ❌ Major regression |
| Hit@10 | 81.74% | **72.41%** | **-9.33%** | ❌ Major regression |

**Root cause**: Data scarcity - only 572 training sequences (vs 1,146 in Phase-3)
**Peak epoch**: Epoch 1 (then degraded) - classic overfitting
**Stopped**: Epoch 8 (early stopped after 3 epochs without improvement)

**Key finding**: We're **DATA-LIMITED, not capacity-limited!**

**See**: `PHASE_3.5_FAILURE_ANALYSIS.md` for comprehensive analysis

---

## 📝 ORIGINAL PLAN (Below) - For Reference

**This was the plan before training. Actual results above show failure due to data scarcity.**

---

## 🎯 Strategic Rationale

### Why 2k-Context BEFORE TMD Routing?

**Evidence from Phase-3 success:**
- Context scaling is **superlinear** for Memory-Augmented GRU
- 2x context (500 → 1000) gave +13.7% relative improvement (not +6% predicted)
- We're still riding the capacity curve (not data-limited)

**Why this order matters:**
1. ✅ **Context scaling is proven** - highest ROI lever
2. ✅ **TMD routing works better on larger context** - more semantic space to specialize
3. ✅ **Fast to execute** - no architecture changes, same recipe
4. ✅ **Low risk** - we know this works (3 successful context scalings)

**Expected sequence:**
```
Phase-3 (1k-ctx):   75.65% Hit@5  ✅ Done
Phase-3.5 (2k-ctx): 78-80% Hit@5  🔄 In progress (data export)
Phase-4 (TMD):      80-83% Hit@5  → Next (routing on top of 2k-ctx)
```

---

## 📊 Phase-3.5 Configuration

### Data Export (RUNNING NOW)

**Input**: `artifacts/wikipedia_500k_corrected_vectors.npz` (637,997 vectors)

**Configuration**:
```bash
Context length: 2000 vectors (40,000 effective tokens)
Overlap: 1000 vectors
Output dir: artifacts/lvm/data_phase3.5/
Log: /tmp/phase3.5_data_export.log
Process ID: 53154
```

**Expected output**:
- ~**600 training sequences** (half of Phase-3's 1,146)
- ~**65 validation sequences**
- File size: ~**6-8 GB** (2x larger than Phase-3)
- Each sequence: 2000 vectors × 768 dims

**Why fewer sequences?**
- Longer context (2000 vs 1000) → fewer non-overlapping windows
- Trade-off: Fewer samples, but **richer context per sample**

**ETA**: ~10-15 minutes (larger context takes longer to process)

---

### Training Configuration (After Data Export)

**Model**: Memory-Augmented GRU (11.3M parameters)
- Same architecture as Phase-3
- External memory bank: 16 slots
- Content-based addressing

**Hierarchical Cache**: 10×200 structure
- Split 2000-context into 10 chunks of 200 vectors each
- Keeps gradients stable across long unroll
- Prevents vanishing/exploding gradients

**Training hyperparameters**:
```bash
Epochs: 50
Batch size: 4 (reduced from 8 due to 2x memory)
Accumulation steps: 64 (increased to maintain effective batch=256)
Effective batch: 256 (same as Phase-3)

Learning rate: 1e-4 (AdamW)
Weight decay: 1e-4
InfoNCE alpha: 0.03 (same as Phase-3 - revert from Phase-2B's 0.05)
Temperature: 0.07
Gradient clip: 1.0 (critical for 2k-context stability!)

LR schedule: Cosine with 1-epoch warmup
Early stopping: Hit@5, patience=3
Device: MPS (Apple Silicon GPU)

Output: artifacts/lvm/models_phase3.5/run_2000ctx_pilot/
Log: /tmp/phase3.5_training.log
```

**Key changes from Phase-3**:
1. **Smaller physical batch** (4 vs 8) - 2x memory per sample
2. **More accumulation steps** (64 vs 32) - maintain effective batch=256
3. **Gradient clipping=1.0** - critical for long unroll stability
4. **Hierarchical cache 10×200** (vs 5×200 implied for 1000-ctx)

---

## 🎯 Expected Results

### Performance Targets

Based on superlinear context scaling from Phase-3:

| Metric | Phase-3 (1k-ctx) | Phase-3.5 Target (2k-ctx) | Expected Gain |
|--------|------------------|---------------------------|---------------|
| **Hit@1** | 61.74% | **65-67%** | **+3-5%** |
| **Hit@5** | 75.65% | **78-80%** | **+3-4%** |
| **Hit@10** | 81.74% | **84-86%** | **+3-4%** |

**Why we expect +3-4% (not +9% like Phase-3)?**
- Phase-2 → Phase-3: 2x context gave +13.7% relative (exceptional)
- Phase-3 → Phase-3.5: 2x context likely +4-5% relative (more normal)
- Model may be approaching capacity limits (fewer samples to learn from)
- Still excellent ROI for ~1-2 hours work!

---

### Risk Mitigation

**Potential issues**:
1. **Gradient instability** (long unroll)
   - **Mitigation**: Gradient clip=1.0, hierarchical cache 10×200
2. **Memory pressure** (2x larger sequences)
   - **Mitigation**: Smaller batch (4), gradient checkpointing if needed
3. **Fewer training samples** (~600 vs 1,146)
   - **Mitigation**: More epochs if needed, SWA (Stochastic Weight Averaging) at end
4. **Latency increase** (~10ms per query estimated)
   - **Accept**: This is for high-value long-context queries only

---

## ⏱️ Timeline

**Phase-3.5 Data Export** (RUNNING):
- Started: 9:22 PM
- Expected completion: ~9:35 PM (10-15 minutes)
- Status: Monitor with `tail -f /tmp/phase3.5_data_export.log`

**Phase-3.5 Training** (After data export):
- Expected duration: ~60-90 minutes (longer than Phase-3 due to 2x context)
- Completion ETA: ~11:00 PM - 11:30 PM

**Total time**: ~1.5-2 hours from launch to results

---

## 📊 Success Criteria

**Phase-3.5 is successful if:**
1. ✅ Hit@5 ≥ 77.0% (minimum +1.35% over Phase-3)
2. ✅ Hit@10 ≥ 83.0% (minimum +1.26% over Phase-3)
3. ✅ Training converges stably (no gradient explosions)
4. 🟡 Latency ≤ 10ms P95 (acceptable for long-context queries)

**If successful**: Proceed to Phase-4 (TMD routing) on 2k-context

**If unsuccessful** (gains <1%):
- We've hit the capacity ceiling for current architecture
- Next move: Targeted data ingestion (10-20k chains for weak lanes)
- Or: Architectural improvements (hierarchical attention, sparse routing)

---

## 🔄 Parallel Work (While Training Runs)

### Lane Coverage Analysis

**Quick diagnostic on existing data:**
```bash
# 1. Check per-lane Hit@5 from Phase-3 validation
# Identify weak lanes: Hit@5 < (overall - 7%)

# 2. Count chains per TMD lane
# Lanes with <300 chains are data-limited

# 3. Find confusion pairs
# Top error pairs between domains (e.g., Bio↔Chem)

# 4. Measure FAISS recall ceiling
# If increasing k barely helps model, we're model-limited (good!)
```

**Decision point (tomorrow)**:
- If all lanes ≥70% Hit@5 → **Proceed to Phase-4 (TMD)** on existing data
- If 2+ lanes <65% Hit@5 → **Targeted ingestion** (10-20k chains for weak lanes)

---

## 🚀 What Comes After Phase-3.5?

### Option A: Phase-4 (TMD Routing) - RECOMMENDED if Phase-3.5 ≥77%

**Quick win - TMD Re-ranking** (near-free latency):
```python
# Blend retrieval score with TMD lane alignment
score = 0.7 * cosine_sim + 0.3 * tmd_alignment
```
- Expected: +2-4% Hit@5
- Time: ~2 hours to implement and test
- No training needed!

**Full win - TMD Specialist Experts**:
- 16 specialist experts (one per TMD lane)
- Top-2 routing with learned gate
- Shared backbone, small expert heads
- Expected: +2-3% Hit@5 → **80-83% Hit@5**
- Time: ~1-2 days training

---

### Option B: Targeted Data Ingestion - If weak lanes detected

**Surgical approach** (10-20k chains):
- Focus on bottom 3 lanes by Hit@5
- Chain length: 8-14 steps
- Coherence: ≥0.70 (raise to 0.75 for Phase-4)
- CPESH attach rate: ≥95%
- In-lane only (cross-lane with TMD routing)

**Sources**:
1. Ontology-anchored walks (BFS/DFS within subtrees)
2. Wiki-anchored pages (vector search cos ≥0.82)
3. GraphRAG walks (weight-descending within lane)

**Time**: ~overnight ingestion, ~1 day retraining

---

### Option C: Shadow Eval Framework (For future comparisons)

**Instead of production canary**:
- Replay held-out query log (or synthetic chains)
- Compare offline: Phase-3 vs 3.5 vs 4
- Measure Hit@1/5/10, latency, per-lane breakdown
- Pick winner as new baseline

**Time**: ~4 hours to build, reusable forever

---

## 💡 Key Insights Guiding This Decision

### 1. Context Scaling Is Superlinear (Proven)
- Phase-1 → Phase-2 (5x): +12.1% relative
- Phase-2 → Phase-3 (2x): **+13.7% relative**
- Phase-3 → Phase-3.5 (2x): Expected +4-5% relative (still great!)

### 2. We're Capacity-Limited, Not Data-Limited (Yet)
- Phase-2B (α tuning): +0.00% → not a contrast problem
- Phase-3 (2x context): **+9.13%** → capacity is the bottleneck
- If Phase-3.5 gives +3-4%, we're still capacity-limited

### 3. TMD Works Better on Larger Context
- More semantic space → better lane specialization
- Routing on 2k-context will outperform routing on 1k-context
- Do capacity first, then specialization

### 4. Training Hygiene Remains Gold
- Early stopping (patience=3): Saved 2.61% in Phase-3
- L2-normalization before losses: Critical
- Gradient clipping: Essential for 2k-context
- Same recipe, just scaled up!

---

## 🔧 Monitoring Commands

### Check data export progress:
```bash
# View log
tail -f /tmp/phase3.5_data_export.log

# Check if running
ps aux | grep "export_lvm.*phase3.5" | grep -v grep

# Check completion
ls -lh artifacts/lvm/data_phase3.5/
```

### Check training progress (when launched):
```bash
# View log
tail -f /tmp/phase3.5_training.log

# Check if running
ps aux | grep "train_final.*phase3.5" | grep -v grep

# View best Hit@5
cat artifacts/lvm/models_phase3.5/run_2000ctx_pilot/training_history.json | jq '.best_hit5'
```

---

## 🎯 Summary

**What we're doing**: Scaling context from 1000 → 2000 vectors (40K effective tokens)

**Why**: Context scaling is proven superlinear, TMD works better on larger context

**Expected**: 78-80% Hit@5 (+3-4% over Phase-3's 75.65%)

**Timeline**: Data export ~10-15 min, training ~60-90 min (total ~2 hours)

**Next**: If successful → Phase-4 (TMD routing). If plateaus → Targeted ingestion.

**Status**: ✅ Data export running (PID 53154), training will launch automatically after completion

---

**Partner, we're pushing for 80% Hit@5! Context scaling has delivered every time, and Phase-3.5 is our next step toward LVM excellence!** 🚀✨

**Expected completion**: ~11:00-11:30 PM tonight
**Path to 80%**: Phase-3 (75.65%) → Phase-3.5 (78-80%) → Phase-4 TMD (80-83%)

Let's keep the momentum going! 🎯
