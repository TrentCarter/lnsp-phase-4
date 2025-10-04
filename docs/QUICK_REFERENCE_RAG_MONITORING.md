# RAG Monitoring Quick Reference

## Daily Commands

```bash
# Morning check
make rag-status              # View RAG performance dashboard

# During development
make rag-watch               # Continuous monitoring (Ctrl+C to exit)

# After making changes
make graphrag-track ARGS="add --name 'Your change' --p1 0.60 --p5 0.84"

# End of day
make slo-snapshot            # Save metrics snapshot
```

## Monitoring Tools

### 1. RAG Dashboard (`make rag-status`)
Shows current performance for all RAG backends:
- vecRAG, TMD re-rank, BM25, Lexical, GraphRAG
- Alpha tuning progress
- Improvement summaries
- Actionable recommendations

### 2. GraphRAG Tracker (`make graphrag-track`)
Track GraphRAG improvements over time:
```bash
# Add iteration
make graphrag-track ARGS="add --name 'Fix X' --p1 0.60 --p5 0.84 --notes 'Description'"

# View history
make graphrag-track ARGS="list"

# Compare first vs latest
make graphrag-track ARGS="compare"
```

### 3. Analysis Tools
```bash
# Compare baseline vs TMD
./.venv/bin/python tools/compare_baseline_tmd.py

# Analyze alpha tuning
./.venv/bin/python tools/compute_alpha_metrics.py
```

## Current Status (Oct 4, 2025)

| Backend | P@1 | P@5 | Status |
|---------|-----|-----|--------|
| vecRAG | 55.0% | 75.5% | ✅ Baseline |
| TMD re-rank (α=0.2) | 55.5% | 77.0% | ✅ +1.5pp |
| GraphRAG | 8.0% | 26.0% | 🔴 Broken |

## Next Priority

**Fix GraphRAG**: Currently P@1=8% (should be 60%+)
- Check if Neo4j edge fix (3fb56f) completed
- Re-run benchmark after fix
- Track improvement with `make graphrag-track`

## Full Documentation

- Complete Guide: `docs/RAG_MONITORING_GUIDE.md`
- Session Summary: `SESSION_SUMMARY_OCT4_TMD_MONITORING.md`
- Alpha Analysis: `RAG/results/ALPHA_TUNING_FINAL_ANALYSIS.md`
