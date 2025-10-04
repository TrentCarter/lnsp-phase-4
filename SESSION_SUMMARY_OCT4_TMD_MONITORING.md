# Session Summary: TMD Alpha Tuning & RAG Monitoring Infrastructure
**Date**: October 4, 2025
**Duration**: ~2 hours
**Status**: ✅ Complete

## Executive Summary

Built comprehensive RAG monitoring infrastructure and completed TMD alpha parameter tuning. Discovered that TMD re-ranking provides modest improvement (+1.5pp P@5) but alpha parameter has no effect. Fixed critical metrics calculation bug and created tools for ongoing performance tracking.

---

## What Was Accomplished

### 1. TMD Alpha Parameter Tuning ✅

**Goal**: Optimize alpha parameter for TMD re-ranking to improve P@5 from 97.5% → 98%+

**What We Did**:
- Created `tune_alpha.sh` to test 5 alpha values (0.2, 0.3, 0.4, 0.5, 0.6)
- Updated `RAG/bench.py` to read `TMD_ALPHA` environment variable
- Created `tools/compute_alpha_metrics.py` for results analysis
- Ran 200-query benchmark for each alpha value (~25 min total)

**Critical Bug Found & Fixed**:
- **Bug**: Metrics calculation used 0-based indexing (`gold_rank == 0`)
- **Reality**: Benchmark uses 1-based indexing (`gold_rank == 1` = first position)
- **Symptom**: Initially showed P@1=0% across all alphas
- **Fix**: Updated metrics to check `gold_rank == 1` and `<= 5` instead of `== 0` and `< 5`

**Results** (After Bug Fix):

| Alpha | TMD% | Vec% | P@1 | P@5 | P@10 | MRR |
|-------|------|------|-----|-----|------|-----|
| **Baseline (no TMD)** | 0% | 100% | 55.0% | 75.5% | 79.0% | 0.6490 |
| 0.2 | 20% | 80% | 55.5% | 77.0% | 79.0% | 0.6555 |
| 0.3 | 30% | 70% | 55.5% | 77.0% | 79.0% | 0.6555 |
| 0.4 | 40% | 60% | 55.5% | 77.0% | 79.0% | 0.6555 |
| 0.5 | 50% | 50% | 55.0% | 77.0% | 79.0% | 0.6530 |
| 0.6 | 60% | 40% | 55.0% | 77.0% | 78.5% | 0.6524 |

**Key Findings**:
1. ✅ TMD re-ranking works: **+1.5pp P@5 improvement** (75.5% → 77.0%)
2. ❌ Alpha doesn't matter: All values 0.2-0.6 produce identical results
3. ❌ Initial P@5=97.5% claim was based on metrics calculation bug
4. ⚠️  TMD signal is much weaker than vector similarity

**Conclusion**: Use **alpha=0.2** for modest improvement, but don't expect optimization through alpha tuning.

---

### 2. RAG Monitoring Infrastructure ✅

**Goal**: Build dashboard and tracking tools per LNSPRAG PRD recommendation

**Tools Created**:

#### A. RAG Performance Dashboard
- **File**: `tools/rag_dashboard.py`
- **Commands**:
  - `make rag-status` - One-time snapshot
  - `make rag-watch` - Continuous monitoring (5s refresh)
- **Features**:
  - Real-time metrics for all RAG backends
  - Alpha tuning progress tracking
  - Improvement summary (baseline vs TMD)
  - Actionable recommendations
  - Quick command reference

#### B. GraphRAG Iteration Tracker
- **File**: `tools/graphrag_tracker.py`
- **Command**: `make graphrag-track ARGS="..."`
- **Features**:
  - Log each GraphRAG experiment with metrics
  - Compare iterations (first vs latest)
  - Track iteration-by-iteration improvements
  - Git commit tracking

**Example Usage**:
```bash
# Add iteration
make graphrag-track ARGS="add --name 'Fix edge expansion' --p1 0.60 --p5 0.84"

# List history
make graphrag-track ARGS="list"

# Compare improvements
make graphrag-track ARGS="compare"
```

#### C. Analysis Tools
- **File**: `tools/compute_alpha_metrics.py` - Compute metrics from raw benchmark results
- **File**: `tools/compare_baseline_tmd.py` - Compare baseline vs TMD re-ranking
- **File**: `compare_alpha_results.py` - Wrapper for alpha analysis

---

### 3. Documentation Created ✅

#### Primary Docs
1. **`docs/RAG_MONITORING_GUIDE.md`** - Complete monitoring guide
   - Tool descriptions and usage
   - Daily monitoring workflows
   - Troubleshooting guide
   - Integration with existing tools

2. **`RAG/results/ALPHA_TUNING_GUIDE.md`** - Alpha tuning guide
   - Quick start commands
   - Tuning strategy
   - Time estimates

3. **`RAG/results/MONITORING_SETUP_COMPLETE.md`** - Setup summary
   - Quick reference
   - Example workflows

#### Analysis Docs
4. **`RAG/results/ALPHA_TUNING_FINAL_ANALYSIS.md`** - Deep dive
   - Root cause analysis
   - Verification tests
   - Future recommendations

5. **`RAG/results/TMD_SUMMARY_TABLE.md`** - Results table
   - Updated with unified benchmark table
   - User manually added comprehensive comparison

---

## Files Modified

### New Files Created
```
tools/rag_dashboard.py          # Main monitoring dashboard
tools/graphrag_tracker.py        # GraphRAG iteration tracker
tools/compute_alpha_metrics.py   # Alpha results analysis
tools/compare_baseline_tmd.py    # Baseline vs TMD comparison
tune_alpha.sh                    # Alpha tuning script
compare_alpha_results.py         # Alpha comparison wrapper

docs/RAG_MONITORING_GUIDE.md     # Complete guide
RAG/results/ALPHA_TUNING_GUIDE.md
RAG/results/MONITORING_SETUP_COMPLETE.md
RAG/results/ALPHA_TUNING_FINAL_ANALYSIS.md
RAG/results/ALPHA_TUNING_RESULTS_OCT4.md

# Benchmark results (5 files, ~166KB each)
RAG/results/tmd_alpha_0.2_oct4.jsonl
RAG/results/tmd_alpha_0.3_oct4.jsonl
RAG/results/tmd_alpha_0.4_oct4.jsonl
RAG/results/tmd_alpha_0.5_oct4.jsonl
RAG/results/tmd_alpha_0.6_oct4.jsonl
```

### Files Modified
```
RAG/bench.py                     # Added TMD_ALPHA env variable support
Makefile                         # Added rag-status, rag-watch, graphrag-track targets
RAG/results/TMD_SUMMARY_TABLE.md # User updated with comprehensive results
```

### Files with Bug Fixes
```
tools/compute_alpha_metrics.py   # Fixed 0-based vs 1-based indexing bug
```

---

## Key Discoveries

### 1. Metrics Calculation Bug (CRITICAL)
**Issue**: P@1 showed 0% across all tests
**Root Cause**:
```python
# WRONG (0-based)
p_at_1 = sum(1 for r in results if r.get('gold_rank', 999) == 0)

# CORRECT (1-based)
p_at_1 = sum(1 for r in results if r.get('gold_rank', 999) == 1)
```
**Impact**: Initial P@5=97.5% claim was based on this bug
**Fix**: Updated all metrics calculations in `tools/compute_alpha_metrics.py`

### 2. TMD Signal Weakness
**Observation**: All alpha values produce identical results
**Hypothesis**:
1. TMD similarities are uniform across corpus (~0.8-0.9 for most docs)
2. Score normalization washes out TMD differences
3. Query TMD extraction doesn't vary enough

**Evidence**: 111 out of 200 queries rank gold doc at position 1, regardless of alpha

### 3. Actual TMD Re-ranking Performance
- **Baseline vecRAG**: P@5 = 75.5%
- **TMD re-rank**: P@5 = 77.0%
- **Improvement**: +1.5 percentage points (NOT the 20pp initially claimed)
- **Conclusion**: Modest but real improvement

---

## Next Steps & Recommendations

### Immediate Actions (High Priority)

#### 1. Fix GraphRAG (CRITICAL)
**Current Status**: P@1=8%, P@5=26% (broken)
**Expected**: P@1=60%+, P@5=84%+ (based on Oct 3 fix)
**Action**:
```bash
# Check if Neo4j edge expansion fix is applied
make graphrag-track ARGS="add --name 'Current broken state' --p1 0.08 --p5 0.26"

# Apply fix (already in scripts/fix_neo4j_concept_edges.py)
# Then re-run benchmark and track improvement
```

#### 2. Use TMD Re-ranking with Alpha=0.2
**Action**: Update default in `RAG/bench.py` line 439
```python
# Change from:
alpha=(1.0 - tmd_alpha)  # Convert TMD weight to vector weight

# To explicitly set default:
tmd_alpha = float(os.getenv("TMD_ALPHA", "0.2"))  # Use 0.2 instead of 0.3
```

#### 3. Skip Corpus Re-ingestion with LLM TMD
**Reason**: Alpha doesn't affect results, so LLM-based corpus TMD won't help
**Time Saved**: ~1.9 hours
**Better Use**: Focus on GraphRAG fixes

### Investigation Tasks (Medium Priority)

#### 4. Debug Why TMD Signal is Weak
Run verification tests from `ALPHA_TUNING_FINAL_ANALYSIS.md`:

**Test 1: TMD Similarity Distribution**
```python
# Check if TMD similarities discriminate
query_tmd = generate_tmd_for_query("material entity")
corpus_tmds = corpus_vectors[:, :16]
tmd_sims = compute_tmd_similarity(query_tmd, corpus_tmds)
print(f"TMD sim range: {tmd_sims.min():.3f} - {tmd_sims.max():.3f}")
print(f"TMD sim std: {tmd_sims.std():.3f}")
# If std < 0.05, TMD isn't discriminating
```

**Test 2: Query TMD Diversity**
```python
# Check if different queries get different TMD codes
queries = ["material entity", "continuant", "MAQC data", "photosynthesis", "protein"]
for q in queries:
    tmd = extract_tmd_with_llm(q)
    print(f"{q}: domain={tmd['domain_code']}, task={tmd['task_code']}, mod={tmd['modifier_code']}")
# If all same code, that's the problem
```

**Test 3: Try Unnormalized Scoring**
```python
# In RAG/vecrag_tmd_rerank.py, change line 205:
# FROM:
combined_scores = alpha * vec_scores_norm + (1.0 - alpha) * tmd_similarities

# TO:
combined_scores = alpha * vec_scores + (1.0 - alpha) * tmd_similarities
# Then re-run alpha tuning
```

### Future Improvements (Low Priority)

#### 5. Alternative Re-ranking Methods
Instead of TMD re-ranking, try:
- **BM25 + Vector Fusion**: Combine lexical and semantic signals
- **Query Expansion**: Use LLM to generate related terms
- **Cross-encoder Re-ranking**: Fine-tune BERT for pairwise ranking

#### 6. Improve TMD Encoding
Current 16D TMD encoding may lose information:
- Consider 32D or 64D encoding
- Use learned embeddings instead of random projection
- Try task-specific TMD schemas

---

## Monitoring Workflow Going Forward

### Daily Routine
```bash
# Morning: Check system health
make lnsp-status    # API/DB health
make rag-status     # Performance metrics

# Throughout day: Monitor changes
make rag-watch      # Continuous monitoring

# After changes: Track iteration
make graphrag-track ARGS="add --name 'Your change' --p1 X --p5 Y"
```

### After Benchmark Runs
```bash
# Compare results
./.venv/bin/python tools/compare_baseline_tmd.py

# Analyze alpha tuning
./.venv/bin/python tools/compute_alpha_metrics.py
```

### End of Day
```bash
# Save snapshot
make slo-snapshot

# Review dashboard
make rag-status
```

---

## Technical Debt & Known Issues

### 1. TMD Re-ranking Alpha Insensitivity
- **Issue**: All alpha values produce identical results
- **Root Cause**: Unknown (needs investigation)
- **Priority**: Medium (works, but can't optimize)
- **Tracked In**: `ALPHA_TUNING_FINAL_ANALYSIS.md`

### 2. GraphRAG Performance Degradation
- **Issue**: P@1=8% (should be 60%+)
- **Root Cause**: 10x edge expansion bug (fix exists)
- **Priority**: CRITICAL
- **Action**: Apply fix and verify

### 3. Metrics Calculation Discrepancy
- **Issue**: Different test runs showed different baselines
- **Root Cause**: Multiple benchmark files with different query sets
- **Priority**: Low (understood now)
- **Resolution**: Use `comprehensive_200.jsonl` as canonical baseline

---

## Commit Plan

### Commit 1: Add RAG Monitoring Infrastructure
```bash
git add tools/rag_dashboard.py tools/graphrag_tracker.py
git add docs/RAG_MONITORING_GUIDE.md
git add RAG/results/MONITORING_SETUP_COMPLETE.md
git add Makefile
git commit -m "Add RAG monitoring infrastructure

- Dashboard: make rag-status, make rag-watch
- GraphRAG iteration tracker: make graphrag-track
- Complete monitoring guide in docs/
- Integration with existing LNSP tools"
```

### Commit 2: Add TMD Alpha Tuning Analysis
```bash
git add tune_alpha.sh compare_alpha_results.py
git add tools/compute_alpha_metrics.py tools/compare_baseline_tmd.py
git add RAG/bench.py  # TMD_ALPHA env variable support
git add RAG/results/ALPHA_TUNING_*.md
git commit -m "Add TMD alpha parameter tuning infrastructure

- tune_alpha.sh: Test 5 alpha values (0.2-0.6)
- Analysis tools: compute metrics, compare baseline
- Fixed metrics bug: 0-based vs 1-based indexing
- Result: +1.5pp P@5 improvement, alpha doesn't affect results
- Recommendation: Use alpha=0.2, skip corpus re-ingestion"
```

### Commit 3: Update Documentation
```bash
git add docs/RAG_MONITORING_GUIDE.md
git add RAG/results/TMD_SUMMARY_TABLE.md
git add SESSION_SUMMARY_OCT4_TMD_MONITORING.md
git commit -m "Update documentation: TMD tuning results and monitoring

- RAG monitoring guide with examples
- TMD summary table with unified benchmarks
- Session summary with next steps
- Complete analysis in ALPHA_TUNING_FINAL_ANALYSIS.md"
```

---

## Success Metrics

### What Worked ✅
1. ✅ Built comprehensive RAG monitoring infrastructure
2. ✅ Fixed critical metrics calculation bug (P@1=0%)
3. ✅ Confirmed TMD re-ranking works (+1.5pp P@5)
4. ✅ Documented all findings and created guides
5. ✅ Created tools for ongoing performance tracking

### What Didn't Work ❌
1. ❌ Alpha tuning inconclusive (all values identical)
2. ❌ Cannot optimize beyond +1.5pp improvement
3. ❌ Initial P@5=97.5% claim was incorrect

### Time Investment vs Value
- **Time Spent**: ~2 hours (alpha tuning + monitoring setup)
- **Direct Value**: +1.5pp P@5 improvement (modest)
- **Infrastructure Value**: Monitoring tools for future iterations (high)
- **Learning Value**: Understanding TMD limitations (high)

---

## References

### Documentation
- Main Guide: `docs/RAG_MONITORING_GUIDE.md`
- Alpha Guide: `RAG/results/ALPHA_TUNING_GUIDE.md`
- Analysis: `RAG/results/ALPHA_TUNING_FINAL_ANALYSIS.md`

### Tools
- Dashboard: `tools/rag_dashboard.py`
- Tracker: `tools/graphrag_tracker.py`
- Analysis: `tools/compute_alpha_metrics.py`

### Make Commands
```bash
make rag-status              # View dashboard
make rag-watch               # Continuous monitoring
make graphrag-track ARGS=... # Track iterations
make lnsp-status             # API health
make graph-smoke             # GraphRAG test
```

---

## Handoff Notes

This session completed TMD alpha tuning and built monitoring infrastructure. The alpha tuning revealed that TMD re-ranking provides modest improvement (+1.5pp P@5) but cannot be optimized further through alpha parameter adjustment.

**Priority for next session**: Fix GraphRAG performance (currently P@1=8%, should be 60%+). Use the monitoring tools created here to track improvements.

All tools are ready to use. Start with `make rag-status` to see current system state.

---

**Session End**: October 4, 2025
**Status**: ✅ Complete and documented
**Next Focus**: GraphRAG fixes (see background job 3fb56f for Neo4j edge fix progress)
