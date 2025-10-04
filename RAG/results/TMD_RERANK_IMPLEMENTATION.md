# TMD-ReRank Implementation Summary

**Date**: 2025-10-04
**Status**: ✅ Implemented and tested
**Backend Name**: `vec_tmd_rerank`

---

## What is TMD-ReRank?

TMD-ReRank is a re-ranking strategy that uses **Taxonomic Metadata (TMD)** to improve vecRAG retrieval by matching semantic metadata between queries and retrieved documents.

### Strategy

1. **Initial FAISS search** using full 784D vectors (768D GTR-T5 + 16D TMD)
2. **Get top-K×2 results** (e.g., top-20 for final top-10)
3. **Generate TMD for query text** using pattern-based extractor (fast, no LLM)
4. **Extract TMD from retrieved vectors** (first 16 dimensions)
5. **Calculate TMD similarity** (cosine similarity between query TMD and result TMDs)
6. **Combine scores**: `final_score = α × vec_score + (1-α) × tmd_score`
7. **Re-rank and return top-K**

---

## TMD Structure (16 Dimensions)

Current implementation uses simplified encoding:

```
Dimension  | Field              | Encoding
-----------|--------------------|-----------------------------------------
[0]        | domain_code        | Normalized 1-16 → 0.0625-1.0
[1]        | task_code          | Normalized 1-27 → 0.037-1.0
[2]        | modifier_code      | Normalized 1-27 → 0.037-1.0
[3:16]     | (unused)           | Zeros (reserved for future use)
```

**Full TMD fields (as designed)**:
- `[0:3]`   - entity_type (one-hot: concept/entity/relation)
- `[3:6]`   - semantic_role (one-hot: subject/predicate/object)
- `[6:9]`   - context_scope (one-hot: local/domain/global)
- `[9:12]`  - temporal_aspect (one-hot: static/dynamic/temporal)
- `[12:16]` - confidence_metrics (4 floats: extraction confidence, relation strength, etc.)

**TODO**: Implement full 16D TMD encoding to use all metadata fields.

---

## Current Performance

### Benchmark Results (50 queries)

| Backend        | P@1   | P@5   | MRR@10 | nDCG@10 | Mean Latency | Speedup    |
|----------------|-------|-------|--------|---------|--------------|------------|
| vec (baseline) | 0.600 | 0.840 | 0.712  | 0.745   | 0.06ms ⚡     | 1.0x       |
| vec_tmd_rerank | 0.600 | 0.840 | 0.712  | 0.745   | 0.14ms       | 0.43x      |

**Key Findings**:
- ✅ **No precision degradation**: TMD re-ranking doesn't hurt accuracy
- ⚠️ **No precision improvement yet**: Simple 3D TMD encoding insufficient
- ⚡ **Still fast**: 0.14ms = 140 microseconds (7,143 queries/second)
- 📊 **2.3x latency overhead** compared to pure vecRAG (acceptable)

---

## Why No Improvement Yet?

### Root Cause: Insufficient TMD Signal

**Current TMD Encoding**: Only uses 3 dimensions
```python
tmd[0] = domain_code / 16.0    # e.g., 0.5625 for domain=9 (Entertainment)
tmd[1] = task_code / 27.0      # e.g., 0.037 for task=1
tmd[2] = modifier_code / 27.0  # e.g., 1.0 for modifier=27
```

**Problems**:
1. **Low dimensionality**: 3D TMD has low discriminative power
2. **Pattern-based extraction**: May not capture semantic intent accurately
3. **Ontology data**: Concept texts are short, domain-specific (hard to pattern-match)
4. **Sparse usage**: Only 3 out of 16 TMD dimensions populated

---

## Next Steps to Improve TMD-ReRank

### 1. Use Full 16D TMD Encoding ✅ HIGH PRIORITY

**Current**: Only domain/task/modifier (3 dims)
**Needed**: Entity type, semantic role, context scope, temporal aspect, confidence (16 dims total)

**Action**: Update TMD generator to populate all 16 dimensions:
```python
tmd[0:3]   = entity_type_onehot      # concept/entity/relation
tmd[3:6]   = semantic_role_onehot    # subject/predicate/object
tmd[6:9]   = context_scope_onehot    # local/domain/global
tmd[9:12]  = temporal_aspect_onehot  # static/dynamic/temporal
tmd[12:16] = confidence_scores       # 4 floats
```

### 2. Use LLM-Based TMD Generation 🔥 RECOMMENDED

**Current**: Pattern-based extraction (`src/tmd_extractor_v2.py`)
**Needed**: LLM-based extraction using Llama 3.1:8b

**Why**:
- Ontology concepts are short, domain-specific (e.g., "oxidoreductase activity")
- Pattern matching fails on technical terminology
- LLM can understand semantic intent better

**Action**: Create LLM-based TMD extractor:
```python
def generate_tmd_with_llm(query_text: str) -> np.ndarray:
    """Use Llama 3.1:8b to extract TMD from query text."""
    prompt = f"""
    Analyze this query and extract metadata:
    Query: {query_text}

    Return JSON:
    {{
      "entity_type": "concept|entity|relation",
      "semantic_role": "subject|predicate|object",
      "context_scope": "local|domain|global",
      "temporal_aspect": "static|dynamic|temporal",
      "confidence": 0.0-1.0
    }}
    """
    response = llama_client.chat(prompt)
    # Parse JSON and convert to 16D vector
    ...
```

### 3. Tune Alpha Weight

**Current**: `alpha=0.7` (70% vector, 30% TMD)
**Recommended**: Grid search `alpha ∈ {0.5, 0.6, 0.7, 0.8, 0.9}`

### 4. Try Different TMD Similarity Metrics

**Current**: Cosine similarity
**Alternatives**:
- Euclidean distance
- Weighted cosine (weight domain > task > modifier)
- Learned similarity (small MLP)

---

## Usage

### Command Line

```bash
export FAISS_NPZ_PATH=artifacts/ontology_4k_full.npz
export PYTHONPATH=.
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 200 \
  --topk 10 \
  --backends vec,vec_tmd_rerank \
  --out RAG/results/tmd_comparison.jsonl
```

### Python API

```python
from RAG.vecrag_tmd_rerank import run_vecrag_tmd_rerank

indices, scores, latencies = run_vecrag_tmd_rerank(
    db=faiss_db,              # FaissDB instance
    queries=query_vectors,    # List[np.ndarray] of 784D vectors
    query_texts=query_strs,   # List[str] for TMD generation
    corpus_vectors=corpus,    # (N, 784) numpy array
    topk=10,
    alpha=0.7                 # 70% vector, 30% TMD
)
```

---

## Code Files

- **Main implementation**: `RAG/vecrag_tmd_rerank.py`
- **Benchmark integration**: `RAG/bench.py` (lines 432-445)
- **TMD extractor**: `src/tmd_extractor_v2.py` (pattern-based, current)
- **Test results**: `RAG/results/summary_1759602435.md`

---

## Comparison to Other Backends

| Backend            | Strategy                           | P@1   | P@5   | Latency  | Status |
|--------------------|-----------------------------------|-------|-------|----------|--------|
| vec                | Pure FAISS 784D                    | 0.600 | 0.840 | 0.06ms ⚡ | ✅ Best|
| vec_tmd_rerank     | vecRAG + TMD re-rank               | 0.600 | 0.840 | 0.14ms   | ✅ OK  |
| vec_graph_rerank   | vecRAG + graph mutual connections  | 0.600 | 0.840 | 9.56ms   | ✅ OK  |
| graphrag_hybrid    | vecRAG + graph traversal (RRF)     | 0.080 | 0.260 | 434ms    | 🔴 Poor|

**Ranking**:
1. **vec**: Fastest, best baseline (use for production)
2. **vec_tmd_rerank**: 2.3x slower, no improvement yet (needs LLM-based TMD)
3. **vec_graph_rerank**: 160x slower, no improvement (dense graph noise)
4. **graphrag_hybrid**: 7,240x slower, -86% precision (broken)

---

## Recommendations

### Short Term (Now)
✅ TMD-ReRank backend is **production-ready** but **not yet beneficial**
- Use `vec` (pure vecRAG) for production deployments
- Continue development on TMD-ReRank to unlock improvements

### Medium Term (Next Sprint)
1. ✅ **Implement full 16D TMD encoding**
2. 🔥 **Add LLM-based TMD generation** (Llama 3.1:8b)
3. 📊 **Re-benchmark** with improved TMD
4. 🎯 **Target**: P@1 > 0.65, P@5 > 0.85 (15% improvement over vecRAG)

### Long Term (Research)
- Investigate learned TMD embeddings (replace hand-crafted features)
- Train small MLP to predict TMD from query text
- Explore multi-stage ranking: vec → TMD → graph

---

## Expected Impact After Improvements

**With full 16D TMD + LLM extraction**:
- P@1: 0.600 → **0.65-0.70** (+8-17%)
- P@5: 0.840 → **0.85-0.90** (+1-7%)
- Latency: 0.14ms → **0.20-0.30ms** (LLM adds overhead)

**Why TMD should help**:
- Ontology queries often have implicit metadata (domain, scope, role)
- Example: "oxidoreductase activity" (domain=biochemistry, scope=molecular, role=function)
- TMD can distinguish between concepts with similar embeddings but different metadata

---

**Generated**: 2025-10-04
**Author**: Claude Code + User
**Backend**: vec_tmd_rerank
**Status**: ✅ Working, needs LLM-based TMD for improvements
