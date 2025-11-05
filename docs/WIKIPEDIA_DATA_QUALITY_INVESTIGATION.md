# Wikipedia Data Quality Investigation

**Status**: 🔬 INVESTIGATION PLANNED
**Priority**: 🔴 CRITICAL - Root cause of backward bias
**Date**: November 2, 2025

---

## 🎯 Research Question

**Is English Wikipedia data inherently backward-biased, making next-chunk prediction harder than previous-chunk prediction?**

---

## 📊 Current Evidence

From `tools/diagnose_p6_direction.py` (P6 training data):
```
Forward (ctx[-1] → target_next): 0.3876
Backward (ctx[-1] → target_prev): 0.4569
Δ = -0.0692 (backward is 7% stronger!)
```

This suggests Wikipedia chunks have **inherent backward temporal structure** - later chunks reference previous concepts more than they preview future concepts.

---

## 🔬 Investigation Plan

### 1. Run Comprehensive Analysis

```bash
./.venv/bin/python tools/analyze_wikipedia_temporal_flow.py \
  --articles-npz artifacts/wikipedia_584k_fresh.npz \
  --sequences-npz artifacts/lvm/training_sequences_ctx5_p6_next_token.npz \
  --n-samples 5000 \
  --output-dir artifacts/lvm/wikipedia_temporal_analysis
```

**Output**: `artifacts/lvm/wikipedia_temporal_analysis/REPORT.md`

### 2. Research Questions to Answer

1. **Consistency**: Is Δ < 0 across ALL articles or just some?
2. **Domain effects**: Are some topics (science, history, biography) more backward-biased?
3. **Position effects**: Are early chunks (lead section) more forward, late chunks more backward?
4. **Chunking artifact**: Is this due to arbitrary chunk boundaries or inherent text structure?
5. **Reversibility**: Would reversing chunk order within articles help?

### 3. Analyze Specific Examples

**From worst_examples output**:
- Extract actual text for top 10 most backward-biased sequences
- Manually read and identify why backward > forward
- Look for patterns (references, explanations, definitions)

### 4. Compare to Alternative Data Sources

Test Δ on different data types:
- ✅ Wikipedia articles (current): Δ = -0.069
- 🔬 Scientific papers (arXiv abstracts): Δ = ?
- 🔬 Tutorial/how-to guides: Δ = ?
- 🔬 News articles: Δ = ?
- 🔬 Stories/narratives: Δ = ?

---

## 💡 Hypotheses

### H1: Lead Section Preview (Forward Bias)
**Expectation**: Early chunks (lead section) preview content → forward bias
**Test**: Check Δ for chunk_id < 5 vs chunk_id ≥ 10
**If true**: Early chunks should have Δ > 0, late chunks Δ < 0

### H2: Explanatory Structure (Backward Bias)
**Expectation**: Later sections explain/elaborate on concepts introduced earlier → backward bias
**Test**: Check if Δ becomes more negative as chunk_id increases
**If true**: Wikipedia is fundamentally backward-biased for LVM training

### H3: Reference Pattern (Backward Bias)
**Expectation**: Articles mention key terms (e.g., "Einstein", "quantum mechanics") in lead, then later chunks reference those terms
**Test**: Analyze worst examples for repeated key terms
**If true**: Later chunks have high similarity to earlier chunks due to shared vocabulary

### H4: Chunking Artifact (No Systematic Bias)
**Expectation**: Chunk boundaries are arbitrary, Δ should be near zero with high variance
**Test**: Check per-article Δ distribution (should be wide, centered at 0)
**If true**: Problem is chunking strategy, not Wikipedia structure

---

## 🎯 Decision Tree

### If Δ < -0.05 (Strong Backward Bias)

**Conclusion**: Wikipedia is unsuitable for autoregressive LVM training in forward direction

**Options**:
1. **Reverse chunk order**: Train on reversed sequences (predict prev instead of next)
2. **Use different data**: Scientific papers, tutorials, stories
3. **Synthetic data**: Generate forward-flowing sequences with controlled properties
4. **Accept and compensate**: Train with MUCH stronger directional pressure (may not work)

### If -0.05 ≤ Δ < 0 (Moderate Backward Bias)

**Conclusion**: Wikipedia is usable but challenging, need balanced directional pressure

**Options**:
1. **P6b v2.3**: Current approach should work (balanced pressure, survival gates)
2. **Data augmentation**: Mix Wikipedia with forward-biased data (tutorials, papers)
3. **Weighted sampling**: Oversample forward-biased articles, undersample backward-biased

### If Δ ≥ 0 (Forward Bias or Neutral)

**Conclusion**: Wikipedia is suitable for LVM training, current failures are due to training approach

**Options**:
1. **Reduce directional pressure**: May be overfitting to noise
2. **Simplify training**: Remove directional loss, rely on MSE + data quality

---

## 📈 Expected Findings

Based on preliminary evidence (Δ = -0.069):

**Most likely**: H2 (Explanatory Structure) + H3 (Reference Pattern)
- Wikipedia articles follow general→specific pattern
- Lead sections introduce key concepts
- Later sections elaborate with references back to lead
- This creates backward temporal flow

**Impact on training**:
- P6 architecture (removed identity path) doesn't help because **data itself is backward**
- Directional loss helps but can't fully overcome data bias without collapse
- Need to either change data or accept moderate backward bias

---

## 🚀 Next Steps

### Immediate (Before Next Training Run)
1. ✅ Run `analyze_wikipedia_temporal_flow.py` (5-10 minutes)
2. ✅ Review REPORT.md for Δ value and per-article distribution
3. ✅ Check reversed_order_test results
4. ⚠️ If Δ < -0.05: STOP training on Wikipedia, switch to different data

### Short-term (This Week)
1. Extract and read worst 10 examples (manual qualitative analysis)
2. Test alternative data sources (arXiv papers, tutorials)
3. If Wikipedia unsuitable: Create synthetic forward-flowing dataset
4. Update P6b v2.3 training plan based on findings

### Medium-term (Next Week)
1. If using Wikipedia: Implement data augmentation (mix with forward-biased sources)
2. If switching data: Retrain P6b v2.3 on new data
3. Document data quality requirements for LVM training
4. Add data quality gates to ingestion pipeline

---

## 📝 Documentation Updates

After investigation completes:
1. Update `docs/WIKIPEDIA_BACKWARD_BIAS_ROOT_CAUSE.md` with findings
2. Add data quality requirements to `docs/LVM_TRAINING_CRITICAL_FACTS.md`
3. Update `CLAUDE.md` with data source recommendations
4. Create `docs/LVM_DATA_REQUIREMENTS.md` if needed

---

## ⚠️ Critical Insights

**Why this matters**:
- P1-P6 ALL failed with backward bias
- P6b v2.1 improved margin but still negative
- P6b v2.2 temporarily flipped margin but collapsed (orthogonal escape)
- **If data is backward, training tricks won't fix it**

**What we learned**:
- P6 proved it's not architecture (identity path removed, still backward)
- P6b v2.2 proved directional pressure can't overcome data bias without breaking quality
- **Need to investigate data BEFORE trying more training approaches**

**Stakes**:
- If Δ < -0.05: Months of training experiments were chasing impossible goal
- If Δ ≈ 0: Current approach (P6b v2.3) should work
- Either way: MUST KNOW before investing more training time

---

**Status**: Ready to run analysis
**Command**: `./tools/analyze_wikipedia_temporal_flow.py --articles-npz artifacts/wikipedia_584k_fresh.npz --sequences-npz artifacts/lvm/training_sequences_ctx5_p6_next_token.npz --n-samples 5000`
**ETA**: 5-10 minutes
**Priority**: 🔴 CRITICAL - Run BEFORE starting P6b v2.3 training
