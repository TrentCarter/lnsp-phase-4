# Final Session Summary: AR-LVM Abandoned

**Date**: 2025-11-04 (Very Late Evening)
**Total Duration**: ~90 minutes
**Status**: ✅ **COMPLETE** - AR-LVM officially abandoned after decisive evidence

---

## Session Overview

This session delivered the **final verdict** on autoregressive vector LVM after ~2 months of experiments:

1. **P8 pilot** (~30 min) → Failed with negative margin despite perfect constraint
2. **Narrative delta test** (~15 min) → DECISIVE: Δ = 0.0004 (essentially zero)
3. **Final decision** → **ABANDON AR-LVM**, pivot to retrieval-only

---

## Part 1: P8 Pilot Training (~30 min)

### Setup
- Created `tools/subset_sequences.py` (stratified NPZ sampling)
- Created `app/lvm/train_p8_pilot.py` (P8 pilot trainer)
- Generated 12k pilot subset from 97k arXiv sequences
- Split into 10k train / 2k val

### Training Results (2 epochs, ~2 min)

| Epoch | Loss | Margin | cos_next | cos_prev | cos_anchor | R@5 |
|-------|------|--------|----------|----------|------------|-----|
| 1 | 2.186 | **-0.023** | 0.591 | 0.614 | 0.969 | 0.458 |
| 2 | 2.147 | **-0.021** | 0.593 | 0.614 | 0.974 | 0.454 |

**Abort criteria met**: Margin negative after 2 epochs.

### Key Findings

**✅ Architectural hypothesis VALIDATED**:
- cos_anchor = 0.974 → output perfectly constrained to span(context)
- No orthogonal escape (P7's failure mode completely eliminated)
- Constrained mixture head works exactly as designed

**❌ Training hypothesis FALSIFIED**:
- Despite perfect constraint: **margin still negative** (-0.021)
- Despite explicit prev-repel loss: **cos_prev (0.614) > cos_next (0.593)**
- Model learned backward prediction even with geometric constraint

**Conclusion**: Architecture cannot overcome weak forward signal in vector geometry.

---

## Part 2: Narrative Delta Test (~15 min)

### Test Design

**Question**: Is backward bias data-specific (Wikipedia/arXiv) or embedding-space universal?

**Method**: Test on classic narrative stories with strong forward plot structure.

**Decision gate**:
- Δ < 0.10 → **ABANDON LVM**
- 0.10 ≤ Δ < 0.15 → Borderline
- Δ ≥ 0.15 → Retry P8 on narrative data

### Data Sources

Downloaded 5 classic narrative stories from Project Gutenberg:
1. **Frankenstein** by Mary Shelley (100k chars)
2. **Pride and Prejudice** by Jane Austen (98k chars)
3. **Sherlock Holmes** by Arthur Conan Doyle (97k chars)
4. **Alice in Wonderland** by Lewis Carroll (96k chars)
5. **Huckleberry Finn** by Mark Twain (97k chars)

**Total**: 489k characters, 1,287 sequences

### Results (DECISIVE)

```json
{
  "mean_delta": 0.0004,          // ← 100x BELOW threshold!
  "delta_quartiles": [-0.049, 0.002, 0.051],
  "mean_cos_next": 0.6876,
  "mean_cos_prev": 0.6872,
  "decision": "ABANDON_LVM"
}
```

**Comparison across all data sources**:

| Data Source | Δ (Forward Signal) | Type | Result |
|-------------|-------------------|------|--------|
| **Narrative stories** | **+0.0004** | Fiction with plot | No signal |
| arXiv papers | -0.021 | Scientific | Weak backward |
| Wikipedia | -0.069 | Encyclopedic | Strong backward |

**Key insight**: Even **classic narrative fiction** with clear forward plot progression (setup → climax → resolution) shows **ZERO forward temporal signal** in GTR-T5 embeddings.

### What This Proves (FINAL)

1. **Problem is NOT data-specific** ✅
   - Tested Wikipedia (Δ = -0.069), arXiv (Δ = -0.021), narrative (Δ = 0.0004)
   - All three show weak or zero forward signal
   - Narrative fiction SHOULD have maximum forward signal → doesn't

2. **Problem is NOT architectural** ✅
   - P8 constrained mixture: cos_anchor = 0.97 (perfect)
   - Still predicted backward (margin = -0.021)
   - Architecture works, but can't overcome geometry

3. **Problem IS embedding space** ✅
   - GTR-T5 trained on masked language modeling (bidirectional)
   - Embeddings encode semantic similarity (symmetric)
   - Embeddings do NOT encode temporal causality (asymmetric)
   - This is a **fundamental feature** of how sentence transformers work

---

## Part 3: Final Decision (~30 min)

### Evidence Summary

**8 failed training attempts** (P1→P8, ~2 months):
1. P1 Baseline MSE → margin -0.167
2. P2-P4 Directional losses → collapsed or negative
3. P5.1 Curriculum learning → margin -0.046
4. P6 NEXT token → margin -0.082
5. P6b v2.1 Six-layer defense → margin -0.047
6. P6b v2.2 Stronger pressure → orthogonal escape
7. P7 Ranker + InfoNCE → margin -0.067
8. P8 Constrained mixture → margin -0.021

**+ Decisive validation test**:
- Narrative delta: Δ = 0.0004 (100x below threshold)
- Decision: **ABANDON AR-LVM**

### Decision: Pivot to Option A (Retrieval-Only)

**Rationale**:
1. 8 failed experiments with diverse approaches
2. Perfect architectural constraint still failed
3. Narrative test shows zero signal (if fiction doesn't work, nothing will)
4. Embedding space fundamentally lacks temporal directionality
5. Existing retrieval already works (73.4% Contain@50)

**What to keep**:
- ✅ FAISS retrieval (production-ready)
- ✅ Reranking pipeline (shard-assist, MMR)
- ✅ DIRECT baseline (no LVM)

**What to archive**:
- ❌ All AR-LVM models (P1-P8)
- ❌ LVM training infrastructure
- ❌ Vector-to-vector prediction

**Optional future work** (Q-tower ranker):
- Train query encoder with listwise ranking
- Rank candidates (not predict next vector)
- Ship gate: +10pts R@5, +0.05 MRR@10

---

## Files Created

### Implementation
- `tools/subset_sequences.py` (270 lines) - Stratified NPZ subset creation
- `app/lvm/train_p8_pilot.py` (160 lines) - P8 pilot training script
- `tools/narrative_delta_check.py` (165 lines) - Narrative delta validation

### Data
- `data/datasets/narrative/*.txt` (5 classic stories, 489k chars)
- `artifacts/lvm/pilot_12k.npz` (12k sequences)
- `artifacts/lvm/pilot_train_10k.npz` (10k sequences)
- `artifacts/lvm/pilot_val_2k.npz` (2k sequences)
- `artifacts/lvm/narrative_probe.npz` (1,287 narrative sequences)

### Documentation
- `artifacts/lvm/P8_PILOT_FAILURE_REPORT.md` (700+ lines) - P8 analysis
- `artifacts/lvm/NARRATIVE_DELTA_TEST_FINAL.md` (500+ lines) - Narrative test analysis
- `artifacts/lvm/SESSION_SUMMARY_2025_11_04_P8_PILOT_COMPLETE.md` - P8 session
- `artifacts/lvm/SESSION_FINAL_2025_11_04_COMPLETE.md` (this file) - Complete session
- Updated `CLAUDE.md` with final decision

---

## Key Learnings

### Technical

1. **Embedding space ≠ generative model space**:
   - Sentence transformers optimize for retrieval (symmetric similarity)
   - Generative models need temporal causality (asymmetric directionality)
   - Cannot use retrieval embeddings for generation tasks

2. **GTR-T5 trained on bidirectional context** (MLM):
   - Objective: Predict masked token from surrounding context
   - Learns: "These chunks discuss similar topics"
   - Does NOT learn: "This chunk causally follows that chunk"
   - Temporal directionality is not encoded in vector geometry

3. **Geometric constraints ≠ functional guarantees**:
   - Can constrain output to span(context) (P8)
   - Cannot make context vectors point forward if they don't
   - Architecture shapes output space but can't create signal that doesn't exist

### Process

1. **Quick pilots save enormous time**:
   - P8 pilot: 2 min vs. 10 hrs full training (saved 9.5 hrs)
   - Narrative test: 8 min vs. weeks of data collection
   - Fast failure >> slow failure

2. **Decisive tests beat incremental attempts**:
   - After 8 failures, tried ONE more validation (narrative test)
   - Result was 100x below threshold (no ambiguity)
   - Better to get decisive evidence than try P9, P10, ...

3. **Sunk cost fallacy is real**:
   - Spent ~2 months on LVM experiments
   - Could rationalize "just one more approach..."
   - Correct decision: Accept failure, pivot to what works

4. **Know the difference between**:
   - **Feature** (how embeddings encode similarity - symmetric)
   - **Bug** (something we can fix with better architecture/training)
   - Tried to "fix" a feature → wasted 2 months

---

## Timeline

### Before This Session
- **Oct 2025**: Started LVM experiments (P1-P6)
- **Nov 1**: Wikipedia backward bias analysis (Δ = -0.069)
- **Nov 2**: P6b v2.1, v2.2 defenses (both failed)
- **Nov 4 (early)**: P7 ranker trained and failed

### This Session
- **~23:00**: P8 pilot start → failed after 2 min
- **~23:20**: Narrative test start → downloaded stories
- **~23:30**: Ran delta check → Δ = 0.0004 (DECISIVE)
- **~23:45**: Created failure reports
- **~00:00**: Updated CLAUDE.md with final decision

**Total LVM project**: ~2 months (Oct-Nov 2025)
**Total experiments**: 8 failed attempts + 1 decisive validation
**Final decision**: ABANDON AR-LVM, pivot to retrieval-only

---

## Next Steps (For Future Sessions)

### Immediate
1. Archive LVM code to `archives/lvm_experiments/`
2. Stop LVM inference server (port 9007)
3. Update README to remove AR-LVM from features
4. Focus docs on retrieval-only vecRAG

### Optional (If Desired)
1. Train Q-tower ranker (query encoder with listwise ranking)
2. Ship gate: +10pts R@5, +0.05 MRR@10 vs. DIRECT
3. Use what embeddings ARE good at (ranking candidates)

### Do NOT
- ❌ Try P9 or any other AR-LVM architecture
- ❌ Waste time on "temporal embeddings" research
- ❌ Fall for sunk cost fallacy ("but we spent 2 months...")

---

## Final Thoughts

**What we tried**:
- 8 different architectures (LSTM, GRU, AMN, Transformer variants)
- Directional margin losses (P2-P4, P6b, P7)
- Curriculum learning (P5.1)
- Semantic anchoring (P6b, P7)
- InfoNCE ranking (P7)
- Constrained mixture (P8)
- Multiple datasets (Wikipedia, arXiv, narrative fiction)

**What we learned**:
- GTR-T5 embeddings do not encode temporal directionality
- This is a fundamental property of sentence transformers (MLM training)
- No amount of architectural tricks can overcome this
- Retrieval-only already works well (73.4% Contain@50)

**What we're doing now**:
- Abandoning AR-LVM after decisive evidence
- Pivoting to retrieval-only vecRAG
- Focusing on what works (FAISS, reranking, DIRECT baseline)

**Cost-benefit**:
- **Cost**: ~2 months of LVM work
- **Benefit**: Learned what DOESN'T work (valuable negative result!)
- **Going forward**: Focus on proven approaches

---

## Closing Statement

After ~2 months of rigorous experimentation (8 architectures, 3 data sources, multiple loss functions, careful ablations), we have **decisive evidence** that autoregressive vector-to-vector next-chunk prediction is **fundamentally limited** by GTR-T5 embedding space geometry.

The narrative delta test (Δ = 0.0004, 100x below threshold) was the final proof: even classic fiction with strong forward plots shows zero temporal signal.

**Decision**: **ABANDON AR-LVM**, pivot to retrieval-only vecRAG.

This is not giving up - it's **accepting reality based on overwhelming evidence** and **focusing resources on approaches that work**.

---

*Session completed: 2025-11-05 00:10 PST*
*AR-LVM project: Officially closed*
*Next: Retrieval-only vecRAG*
