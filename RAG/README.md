# RAG Benchmarks

Evaluate vecRAG as a RAG-only system and compare against alternatives. This harness runs retrieval-only (no LVM, no generation), measures accuracy metrics, and reports latency.

## Backends

- **vec**: vecRAG (FAISS dense, 768D or 784D fused)
- **bm25**: BM25 lexical baseline (strong baseline, requires `rank-bm25`)
- **lex**: Simple token overlap baseline (weak, reference only)
- **lightvec**: LightRAG vector-only (uses same FAISS index for fair comparison)
- **lightrag_full**: LightRAG hybrid mode with graph enhancement (experimental, requires KG in `artifacts/kg/`)

## Datasets

- **CPESH probe set** (if `artifacts/cpesh_cache.jsonl` exists): query = probe text, label = doc_id
- **Self retrieval**: query = concept_text, label = its own doc_id (sanity check)

## Metrics

- **P@1, P@5**: Precision at rank 1 and 5
- **MRR@10**: Mean Reciprocal Rank (rewards top hits)
- **nDCG@10**: Normalized Discounted Cumulative Gain (graded relevance)
- **Mean/P95 latency (ms)**: Performance characteristics

## Quick start

Prereqs:
- Build FAISS index and metadata NPZ (see `sprints/sprint_10012025_S1.md`). Ensure `artifacts/faiss_meta.json` exists and points to the index.
- Install deps: `pip install -r requirements.txt`

Run (auto-detects NPZ via FAISS_NPZ_PATH or common artifacts):

```bash
# CPESH if available, else self-retrieval
python RAG/bench.py --n 1000 --topk 10 --out RAG/results/bench_$(date +%Y%m%d_%H%M).jsonl

# Force dataset
python RAG/bench.py --dataset self --n 1000
python RAG/bench.py --dataset cpesh --n 1000
```

Options:
- `--backends vec,bm25,lex,lightvec,lightrag_full` (comma-separated, default: `vec,bm25`)
- `--n` number of queries (default 500)
- `--topk` evaluation depth (default 10)
- `--dataset` cpesh|self (default: auto)

Env:
- `FAISS_NPZ_PATH` path to NPZ (defaults to `artifacts/fw10k_vectors.npz` or `artifacts/fw9k_vectors.npz`)
- `LNSP_FUSED=1` to enforce 784D fused mode (recommended to match index dim)
- `FAISS_NPROBE` tuning for IVF search

Outputs:
- JSONL per-query results under `RAG/results/` with detailed hits (doc_id, score, rank)
- Markdown summary under `RAG/results/*.md`

## Interpretation Guide

### Expected Performance Baselines

**Self-Retrieval (Sanity Check):**
- **P@1 > 0.95**: System can find exact matches (if lower, check index build)
- **P@5 > 0.98**: Nearly perfect self-retrieval expected
- **Mean latency**: <15ms for 784D IVF-IP with nprobe=16

**CPESH Queries (Real-World Performance):**
- **Lexical (token overlap)**: P@5 ≈ 0.30-0.45 (weak baseline)
- **BM25**: P@5 ≈ 0.50-0.65 (strong lexical baseline)
- **vecRAG (dense)**: P@5 ≈ 0.60-0.75 (target: beat BM25 by 10-15%)
- **LightRAG vector**: P@5 ≈ 0.60-0.75 (similar to vecRAG, same FAISS backend)
- **LightRAG full (hybrid)**: P@5 ≈ 0.70-0.85 (graph boost, slower)

**Latency Expectations:**
- **768D FAISS (IVF-IP, nprobe=16)**: <10ms mean
- **784D fused**: <15ms mean
- **BM25**: <5ms (no embedding overhead)
- **LightRAG full**: 30-50ms (includes graph traversal)

### Performance Targets for vecRAG

1. **Beat BM25**: vecRAG should outperform BM25 by 10-15% on P@5 for CPESH queries
2. **Low latency**: <15ms mean for 784D fused retrieval
3. **High recall**: P@5 > 0.60 on real queries (CPESH probes)
4. **Competitive with LightRAG vector-only**: Within 5% of lightvec baseline

### Red Flags

⚠️ **P@1 < 0.90 on self-retrieval** → Index corruption or dimensionality mismatch
⚠️ **vecRAG worse than BM25** → Embedding quality or FAISS tuning issue
⚠️ **Latency > 30ms for dense** → IVF parameters need tuning (reduce nprobe or nlist)
⚠️ **P@5 < 0.30 on CPESH** → Poor query-corpus alignment, check embedding coverage

## Notes
- For fused 784D indices, queries are built as `[tmd_dense(16), concept_vec(768)]` to match index dimension.
- LightRAG vector-only uses the same FAISS index via `src/adapters/lightrag/vectorstore_faiss.py`.
- BM25 is the primary lexical baseline (stronger than simple token overlap).
- Enhanced JSONL output includes per-hit doc_id, score, and rank for detailed analysis.
