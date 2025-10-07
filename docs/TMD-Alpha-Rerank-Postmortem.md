# TMD Alpha Re-ranking: Invariance Fix and Diagnostics

Date: 2025-10-04
Owner: Retrieval/Infra

## Background

We observed that tuning the TMD blend weight (alpha) produced nearly identical retrieval results across a wide range (0.2–0.6). Separately, GraphRAG results reported by the bench harness were misleading due to a stubbed path.

This document summarizes root causes, code changes, how to run the updated pipeline, and the outcomes.

## Original Issues

- **Alpha invariance in TMD re-ranking**
  - Changing TMD alpha (blend weight) did not meaningfully affect top-k rankings.
  - Root causes identified:
    - Vector-channel min–max normalization collapsed when the top-k band had nearly constant scores.
    - Re-ranking pool (`search_k = topk*2`) was too small to permit rearrangement.
    - Potential LLM fallback could yield constant/weak TMD signals.

- **Bench comparison tool didn’t pick up metrics**
  - `compare_alpha_results.py` expects a per-run JSON `{"summary": true, "metrics": {...}}` line. `RAG/bench.py` only wrote per-query JSONL and a Markdown summary, so alpha comparisons weren’t automatic.

- **GraphRAG outputs misleading**
  - The `lightrag_full` path in `RAG/bench.py` is a stub that returns empty indices, yielding artificially low P@k.

## Changes Implemented

- **`RAG/vecrag_tmd_rerank.py`**
  - Robust normalization with fallback via env var `TMD_NORM`:
    - `softmax` (default), `zscore`, `minmax` with fallback to softmax if collapse detected.
    - `TMD_TEMP` controls softmax temperature (default `1.0`).
  - Expanded re-ranking candidate pool:
    - `search_k = min(topk * TMD_SEARCH_MULT, TMD_SEARCH_MAX, corpus_size)`, defaults: `TMD_SEARCH_MULT=5`, `TMD_SEARCH_MAX=200`.
  - Diagnostics (enable with `TMD_DIAG=1`):
    - Per-query JSONL at `RAG/results/tmd_diag_<ts>.jsonl` records `spearman_vec_tmd`, `vec_collapsed`, `changed_positions`, `top_before`/`top_after`, modes used.
    - Footer aggregates: `collapse_count`, `changed_queries`, `changed_ratio`, `llm_zero_tmd`.
  - LLM controls:
    - `TMD_USE_LLM=1|0` toggles LLM-based TMD extraction; counts zero-vector cases.

- **`RAG/bench.py`**
  - Appends a one-line JSON summary per backend to `--out`:
    - `{"summary": true, "backend": name, "metrics": {"p_at_1","p_at_5","p_at_10","mrr","ndcg"}, "latency_ms": {...}}`
  - Gates `lightrag_full` behind `ALLOW_LRAG_FULL=1` and knowledge graph presence; still marked experimental/unmapped.

- **`tools/tmd_alpha_diagnostics.py`** (new)
  - Aggregates diagnostics JSONL to report Spearman correlation distribution, % queries with top-k changes, normalization collapse frequency, and LLM zero counts.

## How to Run

From repo root:

```bash
# LLM (Ollama) setup
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"
export TMD_USE_LLM=1

# Normalization + search
export TMD_NORM=softmax
export TMD_TEMP=1.0
export TMD_SEARCH_MULT=10
export TMD_SEARCH_MAX=200
export TMD_DIAG=1

# Run alpha sweep
./tune_alpha.sh

# Compare metrics (reads JSON summaries now appended by bench)
python compare_alpha_results.py

# Analyze diagnostics
python tools/tmd_alpha_diagnostics.py --files "RAG/results/tmd_diag_*.jsonl" --hist 12 --alpha
```

Optional quick bench:

```bash
export TMD_ALPHA=0.2
python RAG/bench.py --dataset self --n 200 --topk 10 \
  --backends vec,vec_tmd_rerank \
  --out RAG/results/final_alpha_0_2.jsonl
```

## Results

- `compare_alpha_results.py` after the sweep:

```
Alpha    P@1      P@5      P@10     MRR      nDCG
0.2      0.5550   0.7750   0.7950   0.6588   0.6932
0.3      0.5500   0.7750   0.7900   0.6557   0.6898
0.4      0.5500   0.7700   0.7850   0.6532   0.6867
0.5      0.5400   0.7600   0.7750   0.6432   0.6767
0.6      0.5350   0.7500   0.7650   0.6360   0.6687
```

- Diagnostics summaries (`tools/tmd_alpha_diagnostics.py`) — per 200 queries/run:
  - `alpha=0.400`: Changed@topk 21.5%, Vec collapsed 0, LLM zero TMD 0, Spearman mean +0.5061 (min −0.5680, max +1.0000)
  - `alpha=0.500`: Changed@topk 19.0%, Vec collapsed 0, LLM zero TMD 0, Spearman mean +0.5061
  - `alpha=0.600`: Changed@topk 15.5%, Vec collapsed 0, LLM zero TMD 0, Spearman mean +0.5061
  - `alpha=0.700`: Changed@topk 13.0%, Vec collapsed 0, LLM zero TMD 0, Spearman mean +0.5061
  - `alpha=0.800`: Changed@topk 9.0%,  Vec collapsed 0, LLM zero TMD 0, Spearman mean +0.5061

Interpretation:
- Non-zero changed-topk confirms alpha affects rankings now.
- No normalization collapse; LLM extracting TMD successfully.
- Moderate positive Spearman suggests TMD provides complementary (not identical) signal.

## Recommendations

- Use `TMD_ALPHA=0.2` as default on this dataset (best P@1/P@5/MRR observed).
- Keep `TMD_NORM=softmax`, `TMD_SEARCH_MULT=10`, `TMD_SEARCH_MAX=200` as operational defaults.
- Avoid `lightrag_full` in `RAG/bench.py` until mapping is implemented; use `vec_graph_rerank` or the LightRAG runner script.

## Files Modified/Created

- Modified: `RAG/vecrag_tmd_rerank.py`
  - Added `_normalize_scores()`, `_softmax()`, `_spearman()`; env-driven search and diagnostics.
- Modified: `RAG/bench.py`
  - Appends per-backend summary JSON lines; gates `lightrag_full`.
- Added: `tools/tmd_alpha_diagnostics.py`

## Environment Variables

- `TMD_ALPHA` — TMD weight (bench converts to vec weight internally)
- `TMD_NORM` — `softmax|zscore|minmax` (default `softmax`)
- `TMD_TEMP` — softmax temperature (default `1.0`)
- `TMD_SEARCH_MULT` — multiply `topk` for re-rank pool (default `5`)
- `TMD_SEARCH_MAX` — cap on re-rank pool size (default `200`)
- `TMD_DIAG` — `1` to enable per-query diagnostics JSONL
- `TMD_USE_LLM` — `1` to enable LLM extraction (default `1`)
- `ALLOW_LRAG_FULL` — gate `lightrag_full` backend (default `0`)

## Future Work

- Unit tests for normalization modes and collapse handling under `tests/RAG/`.
- Document LightRAG result mapping if/when implemented, then re-enable `lightrag_full` in bench.
