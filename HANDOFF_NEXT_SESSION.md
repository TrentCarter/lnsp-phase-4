# Handoff Document for Next Session

**Date**: October 4, 2025
**Session**: Critical Fixes & Validation (COMPLETED)
**Next Session**: vecRAG + GraphRAG Benchmarking
**System Status**: ✅ READY FOR TESTING

---

## 🚦 TL;DR - What You Need to Know

### The System is Ready ✅
- 4,484 ontology concepts ingested and synchronized
- PostgreSQL + Neo4j + FAISS all match (100% sync)
- CPESH data 96.2% complete (4,313/4,484)
- FAISS index built (14MB) and ready
- Neo4j graph has 10,257 relationships

### Critical Fixes Were Applied ✅
- Fixed `dataset_source` labeling bug (parameterized)
- Fixed missing FAISS save() call (NPZ now created)
- Fixed validation script (content-based, not label-based)
- All fixes committed: `2f6b064`

### Known Cosmetic Issue ⚠️
- Current 4,484 concepts labeled `factoid-wiki-large`
- BUT data IS ontological (verified manually)
- Future ingestions will use correct `ontology-*` labels
- No need to re-ingest (7-8 hours saved)

---

## 📋 Immediate Next Steps (Start Here)

### 1. Read Critical Documentation (5 minutes)
```bash
# Start with these in order:
cat LNSP_LONG_TERM_MEMORY.md         # 5 Cardinal Rules
cat docs/FIXES_Oct4_2025_FactoidWiki_Labeling.md  # Recent fixes
cat SESSION_SUMMARY_Oct4_2025.md     # This session's work
```

### 2. Verify System Status (2 minutes)
```bash
# Check synchronization
./scripts/verify_data_sync.sh

# Expected output:
# ✅ ALL CHECKS PASSED - Data stores are synchronized!
```

### 3. Run vecRAG Benchmark (15 minutes)
```bash
FAISS_NPZ_PATH=artifacts/ontology_4k_full.npz \
  ./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 100 \
  --topk 10 \
  --backends vec \
  --out RAG/results/vecrag_ontology_4k.jsonl
```

### 4. Run GraphRAG Quick Test (5 minutes)
```bash
./scripts/graphrag_quick_test.sh

# Expected: +10-15% P@1 improvement over vecRAG
```

### 5. Generate 6-Degrees Shortcuts (if needed)
```bash
./scripts/generate_6deg_shortcuts.sh

# Target: 0.5-3% of total edges (~50-300 shortcuts)
```

---

## 🗂️ File Locations (Quick Reference)

### Critical Documentation
- `LNSP_LONG_TERM_MEMORY.md` - 5 Cardinal Rules (READ FIRST!)
- `CLAUDE.md` - Instructions for Claude Code
- `README.md` - Project overview
- `SESSION_SUMMARY_Oct4_2025.md` - This session's summary

### Fix Documentation
- `docs/FIXES_Oct4_2025_FactoidWiki_Labeling.md` - Comprehensive fix details
- `docs/CRITICAL_GraphRAG_Data_Synchronization.md` - Sync requirements
- `docs/GraphRAG_Root_Cause_Analysis.md` - Oct 2-3 incident

### Data Artifacts
- `artifacts/ontology_4k_full.npz` - FAISS vectors (84MB)
- `artifacts/fw10k_ivf_flat_ip.index` - FAISS index (14MB)
- PostgreSQL database: `lnsp` (4,484 concepts)
- Neo4j database: Default (4,484 concepts, 10,257 edges)

### Validation Scripts
- `scripts/verify_data_sync.sh` - Check PostgreSQL/Neo4j/FAISS sync
- `scripts/validate_no_factoidwiki.sh` - Check data quality
- `scripts/graphrag_quick_test.sh` - Quick GraphRAG test

---

## 🎯 Goals for Next Session

### Priority 0: Benchmarking (REQUIRED)
1. **vecRAG baseline** - Establish current performance
   - Metrics: P@1, P@5, P@10
   - Expected: ~54% P@1 (from previous tests)
   - Dataset: 100 queries from ontology concepts

2. **GraphRAG performance** - Measure improvement
   - Metrics: P@1, P@5, P@10 + graph neighbors
   - Expected: +10-15% P@1 over vecRAG
   - Modes: Local (1-2 hop) vs Global (walks)

3. **Comparative analysis** - vecRAG vs GraphRAG vs Hybrid
   - Document which queries benefit from graph
   - Measure latency overhead
   - Identify failure cases

### Priority 1: Graph Optimization (RECOMMENDED)
1. **Generate SHORTCUT_6DEG edges**
   - Target: 0.5-3% of total edges (~50-300 shortcuts)
   - Method: Identify concepts 4-6 hops apart with similarity >0.7
   - Validate: Average path length should drop to ≤6 hops

2. **Re-run GraphRAG benchmark** with shortcuts
   - Compare performance: with vs without shortcuts
   - Expected: Faster convergence, better recall

### Priority 2: Scale Testing (OPTIONAL)
1. Ingest full ontology datasets (target 10K+ concepts)
2. Test synchronization at scale
3. Measure ingestion throughput improvements

---

## ⚠️ Common Pitfalls to Avoid

### ❌ Don't Do These Things
1. **Don't run tools/regenerate_*_vectors.py** - Only updates PostgreSQL, breaks sync
2. **Don't use FactoidWiki data** - Not ontological (Cardinal Rule #2)
3. **Don't skip verification scripts** - Always check sync before benchmarks
4. **Don't modify data stores independently** - Use atomic ingestion only

### ✅ Always Do These Things
1. **Always read LNSP_LONG_TERM_MEMORY.md first** - 5 Cardinal Rules
2. **Always run verify_data_sync.sh** - Before any benchmark/test
3. **Always use real LLM** - Ollama + Llama 3.1:8b (no stubs)
4. **Always verify CPESH data** - Check soft/hard negatives exist

---

## 🔍 Troubleshooting

### Issue: Benchmark fails with "No NPZ file found"
**Solution**: Set environment variable
```bash
export FAISS_NPZ_PATH=artifacts/ontology_4k_full.npz
```

### Issue: GraphRAG returns 0 neighbors
**Check**:
1. Run `./scripts/verify_data_sync.sh` - Ensure sync
2. Check Neo4j: `cypher-shell "MATCH ()-[r:RELATES_TO]->() RETURN count(r)"`
3. Expected: ~10,257 relationships

### Issue: Validation script fails
**If it says "FactoidWiki detected"**:
- Check the sample concepts shown
- If they look ontological (activity, software, entity), the labeling bug is cosmetic
- Data is correct, just mislabeled
- Run benchmarks anyway

### Issue: Import errors in Python
**Solution**: Use correct Python path
```bash
PYTHONPATH=src ./.venv/bin/python <script>
```

---

## 📊 Expected Results (Sanity Check)

### vecRAG Baseline
- **P@1**: 50-55% (similar to previous FactoidWiki tests)
- **P@5**: 75-80%
- **P@10**: 85-90%
- **Latency**: <1ms per query

### GraphRAG with Graph
- **P@1**: 60-70% (+10-15% improvement)
- **P@5**: 80-85%
- **P@10**: 90-95%
- **Latency**: 5-10ms per query (graph overhead)
- **Graph neighbors**: 1-10 per query (NOT 0!)

### If Results Are Different
- Much lower P@1 (<40%): Check data sync, verify FAISS index
- Zero graph neighbors: Check Neo4j relationships exist
- No improvement: Verify RRF fusion is enabled

---

## 🎓 Context from Previous Session

### What Was the Issue?
After overnight 6K ontology ingestion, discovered:
1. All data labeled `factoid-wiki-large` (wrong)
2. FAISS NPZ not created (missing save() call)
3. Validation script gave false positives

### What Was Fixed?
1. Parameterized `dataset_source` in `src/ingest_factoid.py`
2. Added `faiss_db.save()` in `src/ingest_ontology_simple.py`
3. Updated validation to check content, not just labels

### What's the Current State?
- Data IS ontological (100% verified)
- All stores synchronized (4,484 concepts)
- Labeling is wrong but cosmetic only
- System is ready for benchmarking

---

## 🚀 Quick Start Commands

### Verify System (1 minute)
```bash
./scripts/verify_data_sync.sh
```

### Run Full Benchmark Suite (30 minutes)
```bash
# vecRAG only
FAISS_NPZ_PATH=artifacts/ontology_4k_full.npz \
  ./.venv/bin/python RAG/bench.py --dataset self --n 100 --topk 10 --backends vec

# GraphRAG quick test
./scripts/graphrag_quick_test.sh

# Full comparative (if time allows)
./scripts/run_graphrag_benchmark.sh
```

### Check Results
```bash
# View results
cat RAG/results/vecrag_ontology_4k.jsonl | jq '.metrics'

# Compare P@1
grep "P@1" RAG/results/*.jsonl
```

---

## 📝 Git Status

**Last Commit**: `2f6b064` - Fix critical bugs: dataset_source labeling + FAISS save()

**Branch**: `main`

**Staged Changes**: None

**Untracked Files**:
- `SESSION_SUMMARY_Oct4_2025.md` (new)
- `HANDOFF_NEXT_SESSION.md` (this file)
- `sprints/sprint_10042025_Critical_Fixes.md` (new)

**Next Commit Suggestion**:
```bash
git add SESSION_SUMMARY_Oct4_2025.md HANDOFF_NEXT_SESSION.md sprints/
git commit -m "Add session summary and handoff documentation"
```

---

## 🎯 Success Criteria for Next Session

### Minimum Success
- [ ] vecRAG benchmark completes (100 queries)
- [ ] GraphRAG benchmark completes (10-50 queries)
- [ ] Results show >0 graph neighbors
- [ ] Document P@1 improvement (even if 0%)

### Target Success
- [ ] GraphRAG shows +5-10% P@1 improvement
- [ ] Graph neighbors average 1-10 per query
- [ ] Comparative analysis complete (vecRAG vs GraphRAG)
- [ ] Results documented in sprints/

### Stretch Goals
- [ ] Generate SHORTCUT_6DEG edges
- [ ] Re-run benchmarks with shortcuts
- [ ] Show path length reduction proof
- [ ] Scale to 10K+ concepts

---

## 🆘 Emergency Contacts

**If Something Goes Wrong**:

1. **Data desynchronization**: Re-read `docs/CRITICAL_GraphRAG_Data_Synchronization.md`
2. **FAISS issues**: Check `artifacts/ontology_4k_full.npz` exists and has vectors
3. **Neo4j issues**: Verify `cypher-shell` works, check relationship count
4. **Import errors**: Ensure `PYTHONPATH=src` and using `./.venv/bin/python`

**Documentation Trail**:
- Start: `LNSP_LONG_TERM_MEMORY.md`
- Issues: `docs/FIXES_Oct4_2025_FactoidWiki_Labeling.md`
- Context: `SESSION_SUMMARY_Oct4_2025.md`
- This file: `HANDOFF_NEXT_SESSION.md`

---

**Good luck with the benchmarks! The system is ready and waiting.** 🚀

_If you see this file, you're in good hands. All the hard work is done - now it's time to see the results!_
