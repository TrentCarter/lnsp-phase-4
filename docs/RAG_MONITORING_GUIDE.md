# RAG Performance Monitoring Guide

## Overview

Real-time monitoring and tracking tools for RAG backend performance metrics, designed to help track improvements over iterations and maintain visibility into system health.

## Quick Start

```bash
# View current RAG performance dashboard
make rag-status

# Continuous monitoring (updates every 5s)
make rag-watch

# Track GraphRAG iterations
make graphrag-track ARGS="list"
```

## Tools

### 1. RAG Dashboard (`make rag-status`)

**Purpose**: Centralized view of all RAG backend metrics

**Features**:
- Main benchmark comparison (vecRAG, TMD re-rank, BM25, Lexical, GraphRAG)
- Alpha parameter tuning progress
- Improvement summary (TMD vs baseline)
- Actionable recommendations
- Quick command reference

**Usage**:
```bash
# One-time snapshot
make rag-status

# Continuous monitoring (refreshes every 5s)
make rag-watch

# Python direct invocation
./.venv/bin/python tools/rag_dashboard.py
./.venv/bin/python tools/rag_dashboard.py --watch
```

**Example Output**:
```
====================================================================================================
                                   RAG PERFORMANCE DASHBOARD
                            Last Updated: 2025-10-04 15:30:00
====================================================================================================

MAIN BENCHMARKS (200 queries)
----------------------------------------------------------------------------------------------------
Backend              P@1      P@5     P@10      MRR     nDCG      Latency   Status
----------------------------------------------------------------------------------------------------
vecRAG              91.5%    95.6%    97.0%   0.9356   0.9652      0.05ms   ✅ OK
TMD_rerank          94.5%    97.5%    98.5%   0.9557   0.9775      1.50s    ✅ BEST
BM25                48.5%    72.5%      N/A   0.6050   0.6470      0.94ms   ✅ OK
Lexical             49.0%    71.0%      N/A   0.5860   0.6230      0.38ms   ✅ OK
GraphRAG             8.0%    26.0%      N/A   0.1530   0.2150    434.00ms   🔴 BROKEN
----------------------------------------------------------------------------------------------------

TMD ALPHA PARAMETER TUNING
----------------------------------------------------------------------------------------------------
Alpha    TMD%   Vec%     P@1      P@5     P@10      MRR    Status
----------------------------------------------------------------------------------------------------
0.3       30%    70%    94.5%    97.5%    98.5%   0.9557   🏆 OPTIMAL
----------------------------------------------------------------------------------------------------
Best configuration: alpha=0.3 (P@5=97.5%)

TMD RE-RANKING IMPROVEMENTS
----------------------------------------------------------------------------------------------------
Metric     Baseline     TMD Rerank   Δ Absolute   Δ Relative
----------------------------------------------------------------------------------------------------
P@1          91.5%         94.5%        +3.0pp        +3.3%
P@5          95.6%         97.5%        +2.0pp        +2.1%
P@10         97.0%         98.5%        +1.5pp        +1.5%
MRR        0.9356        0.9557       +0.0201        +2.1%
nDCG       0.9652        0.9775       +0.0123        +1.3%
----------------------------------------------------------------------------------------------------

RECOMMENDATIONS
----------------------------------------------------------------------------------------------------
1. 🔴 CRITICAL: GraphRAG performance degraded (P@1 < 20%). Review Neo4j edge expansion.
2. ✅ All other systems performing well.
----------------------------------------------------------------------------------------------------
```

### 2. GraphRAG Iteration Tracker (`make graphrag-track`)

**Purpose**: Track GraphRAG improvements over time

**Features**:
- Log each GraphRAG experiment with metrics
- Compare iterations to show progress
- Track iteration-by-iteration improvements
- Git commit tracking

**Usage**:
```bash
# Add a new iteration
make graphrag-track ARGS="add --name 'Fix 10x edge expansion' --p1 0.60 --p5 0.84 --notes 'Reduced Neo4j edge expansion from 10x to 3x'"

# List all iterations
make graphrag-track ARGS="list"

# Compare first vs latest
make graphrag-track ARGS="compare"

# Full example with all metrics
make graphrag-track ARGS="add \
  --name 'Optimized local search' \
  --p1 0.65 \
  --p5 0.88 \
  --p10 0.92 \
  --mrr 0.745 \
  --latency 250.5 \
  --notes 'Switched from global to local search, improved relevance' \
  --commit abc123f"
```

**Example Output**:
```
========================================================================================================================
                                           GRAPHRAG ITERATION HISTORY
========================================================================================================================

#    Date         Name                           P@1      P@5     P@10      MRR    Latency
------------------------------------------------------------------------------------------------------------------------
1    2025-10-01   Baseline (broken)              8.0%    26.0%     N/A   0.1530    434.0ms
     Notes: Initial implementation with 10x edge expansion bug
2    2025-10-03   Fix 10x edge expansion        60.0%    84.0%     N/A   0.7120      9.6ms
     Notes: Reduced Neo4j edge expansion from 10x to 3x
3    2025-10-04   Optimized local search        65.0%    88.0%    92.0%  0.7450    250.5ms
     Notes: Switched from global to local search, improved relevance
------------------------------------------------------------------------------------------------------------------------

Total iterations: 3
```

## Monitoring Workflows

### Daily Monitoring

```bash
# Morning: Check overall system health
make rag-status

# Throughout day: Monitor specific backends
make lnsp-status         # API health
make graph-smoke         # GraphRAG endpoints
make slo-snapshot        # Save SLO metrics
```

### After Making Changes

```bash
# 1. Run benchmarks
bash tune_alpha.sh  # If tuning TMD

# 2. Check results
make rag-status

# 3. Track iteration (for GraphRAG changes)
make graphrag-track ARGS="add --name 'Your change' --p1 0.XX --p5 0.XX"

# 4. Compare to previous
make graphrag-track ARGS="compare"
```

### Continuous Monitoring (Development)

```bash
# Terminal 1: Run API
make api PORT=8094

# Terminal 2: Watch metrics
make rag-watch

# Terminal 3: Make changes and test
# ... your development work ...
```

## Integration with Existing Tools

### Relationship to `make lnsp-status`

- **`make lnsp-status`**: Low-level API health check (endpoints, Neo4j, PostgreSQL, FAISS)
- **`make rag-status`**: High-level RAG performance metrics (P@1, P@5, improvements)

Use both together:
```bash
make lnsp-status  # Is the system healthy?
make rag-status   # How well is it performing?
```

### Relationship to SLO Metrics

- **SLO snapshots** (`make slo-snapshot`): Point-in-time performance capture
- **RAG dashboard**: Historical comparison and improvement tracking

SLO workflow:
```bash
# Before change
make slo-snapshot  # Save baseline

# After change
make slo-snapshot  # Save new state
make rag-status    # View improvements
```

## Data Sources

### Result Files

The dashboard reads from:
- `RAG/results/comprehensive_200.jsonl` - Main benchmark results
- `RAG/results/tmd_200_oct4.jsonl` - TMD re-ranking results
- `RAG/results/tmd_alpha_*.jsonl` - Alpha tuning results
- `RAG/results/graphrag_*.jsonl` - GraphRAG results

### Iteration Tracking

GraphRAG iterations stored in:
- `artifacts/graphrag_iterations.jsonl` - Append-only log

## Best Practices

### 1. Track Every Significant Change

```bash
# After fixing a bug
make graphrag-track ARGS="add --name 'Fix: description' --p1 X --p5 Y"

# After optimization
make graphrag-track ARGS="add --name 'Optimize: description' --p1 X --p5 Y"
```

### 2. Include Context in Notes

```bash
make graphrag-track ARGS="add \
  --name 'Reduce edge expansion' \
  --p1 0.60 \
  --p5 0.84 \
  --notes 'Changed MAX_EDGES from 10000 to 1000 in graph_service.py:45'"
```

### 3. Regular Snapshots

```bash
# Daily
make rag-status | tee artifacts/rag_status_$(date +%Y%m%d).txt

# After major changes
make slo-snapshot
```

### 4. Use Watch Mode During Development

```bash
# While iterating on improvements
make rag-watch
```

## Troubleshooting

### Dashboard Shows "N/A" or Missing Data

**Cause**: Result files don't exist yet

**Fix**:
```bash
# Run comprehensive benchmark
FAISS_NPZ_PATH=artifacts/ontology_4k_full.npz \
PYTHONPATH=. \
./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 200 \
  --topk 10 \
  --backends vec,bm25,lex \
  --out RAG/results/comprehensive_200.jsonl
```

### GraphRAG Tracker Shows No Iterations

**Cause**: No iterations added yet

**Fix**:
```bash
# Add current baseline
make graphrag-track ARGS="add --name 'Current baseline' --p1 0.XX --p5 0.XX"
```

### Dashboard Doesn't Update

**Cause**: Using static mode instead of watch mode

**Fix**:
```bash
# Use watch mode for continuous updates
make rag-watch
```

## Advanced Usage

### Custom Comparison Scripts

```python
#!/usr/bin/env python3
import json
from pathlib import Path

# Load results
results_dir = Path("RAG/results")
with open(results_dir / "comprehensive_200.jsonl") as f:
    for line in f:
        data = json.loads(line)
        if 'summary' in data:
            print(f"{data['name']}: P@5 = {data['metrics']['p_at_5']:.3f}")
```

### Automated Reporting

```bash
# Cron job for daily reports
0 9 * * * cd /path/to/lnsp-phase-4 && make rag-status | mail -s "Daily RAG Status" team@example.com
```

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Check RAG Performance
  run: |
    make rag-status
    python tools/check_regression.py  # Custom script to fail if metrics drop
```

## Related Documentation

- [Alpha Tuning Guide](../RAG/results/ALPHA_TUNING_GUIDE.md)
- [TMD Schema](PRDs/TMD-Schema.md)
- [GraphRAG QuickStart](GraphRAG_QuickStart.md)
- [LNSPRAG Status Tool](../tools/lnsprag_status.py)

## Quick Reference

```bash
# Monitoring Commands
make rag-status              # View dashboard (one-time)
make rag-watch               # Continuous monitoring (5s refresh)
make lnsp-status             # API health check
make graph-smoke             # GraphRAG endpoint test
make slo-snapshot            # Save SLO metrics

# Iteration Tracking
make graphrag-track ARGS="list"                    # List all iterations
make graphrag-track ARGS="compare"                 # Compare first vs latest
make graphrag-track ARGS="add --name X --p1 Y"     # Add new iteration

# Benchmarking
bash tune_alpha.sh                                 # Run alpha tuning
./.venv/bin/python compare_alpha_results.py        # Compare alpha results
```

## Example Workflow: Improving GraphRAG

```bash
# 1. Record current baseline
make graphrag-track ARGS="add --name 'Baseline before fix' --p1 0.08 --p5 0.26"

# 2. Make your changes (e.g., fix edge expansion bug)
# ... edit code ...

# 3. Run benchmark
PYTHONPATH=. ./.venv/bin/python RAG/bench.py \
  --dataset self --n 50 --topk 10 --backends graphrag_hybrid \
  --out RAG/results/graphrag_after_fix.jsonl

# 4. Record new results
make graphrag-track ARGS="add --name 'Fixed edge expansion' --p1 0.60 --p5 0.84"

# 5. Compare improvements
make graphrag-track ARGS="compare"

# 6. View overall status
make rag-status
```

Output shows:
```
ITERATION-BY-ITERATION P@5 PROGRESS
----------------------------------------------------------------------------------------------------
1. [2025-10-01] Baseline before fix                      P@5: 26.0% (baseline)
2. [2025-10-04] Fixed edge expansion                     P@5: 84.0% (+58.0pp)
----------------------------------------------------------------------------------------------------
```

Success! 🎉 GraphRAG P@5 improved from 26% to 84% (+58 percentage points).
