# Quick Start: LightRAG Integration

**Status:** Ready for benchmark integration
**Date:** October 4, 2025
**Next Session:** Pick up here after `/clear`

---

## What Was Done

1. ✅ Stored 2,484 concept vectors in Neo4j (768-dim)
2. ✅ Implemented LightRAG-style retrieval in `RAG/graphrag_lightrag_style.py`
3. ✅ Added 44 random shortcuts (1% of nodes) in <1 second
4. ✅ Documented architecture in `docs/PRDs/PRD_GraphRAG_LightRAG_Architecture.md`

---

## Next Steps (EXECUTE THESE)

### Step 1: Integrate with bench.py

Add this to `RAG/bench.py` around line 400 (after graphrag backends):

```python
# Add after graphrag_backend initialization
if "lightrag" in backends:
    from graphrag_lightrag_style import run_lightrag_style
    HAS_LIGHTRAG = True
else:
    HAS_LIGHTRAG = False

# In eval_backend() function, add this case:
elif name == "lightrag":
    if not HAS_LIGHTRAG:
        raise ValueError("lightrag backend not available")

    # Get query vectors (should already exist in bench.py)
    query_vectors = np.array([q.vec for q in queries])

    I, D, L = run_lightrag_style(
        queries_text=queries_text,
        query_vectors=query_vectors,
        concept_texts=corp.texts,
        topk=topk
    )
```

### Step 2: Run Benchmark

```bash
# Test with 50 queries first
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 FAISS_NUM_THREADS=1
export FAISS_NPZ_PATH=artifacts/ontology_4k_full.npz
export PYTHONPATH=.

./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 50 \
  --topk 10 \
  --backends vec,lightrag \
  --out RAG/results/lightrag_vs_vec_50.jsonl

# If successful, run full benchmark (200 queries)
./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 200 \
  --topk 10 \
  --backends vec,bm25,lex,lightrag \
  --out RAG/results/lightrag_full_200.jsonl
```

### Step 3: Compare Results

```bash
# Check the summary
cat RAG/results/summary_*.md | tail -20

# Expected improvement:
# vecRAG:   P@1=55%, P@5=75.5%
# LightRAG: P@1=60-70% (target), P@5=80-85% (target)
```

---

## File Locations

**Implementation:**
- `RAG/graphrag_lightrag_style.py` - Main retriever class
- `scripts/add_shortcuts_fast.py` - Fast random shortcuts
- `/tmp/add_vectors_to_neo4j.py` - Vector storage script

**Documentation:**
- `docs/PRDs/PRD_GraphRAG_LightRAG_Architecture.md` - Full architecture PRD
- `SESSION_SUMMARY_OCT4_GRAPHRAG_INVESTIGATION.md` - Investigation log
- `RAG/results/GRAPHRAG_DIAGNOSIS_OCT4.md` - Technical diagnosis

**Tests:**
- `/tmp/test_lightrag_style.py` - Manual test script

---

## Key Commands

### Verify Neo4j has vectors
```bash
cypher-shell -u neo4j -p password \
  "MATCH (c:Concept) WHERE c.vector IS NOT NULL RETURN count(c)"
# Should return: 2484
```

### Verify shortcuts exist
```bash
cypher-shell -u neo4j -p password \
  "MATCH ()-[r:SHORTCUT_6DEG]->() RETURN count(r)"
# Should return: 44
```

### Test LightRAG retrieval manually
```bash
PYTHONPATH=. ./.venv/bin/python /tmp/test_lightrag_style.py
# Should print: ✅ LightRAG-style retrieval working!
```

---

## Architecture Recap

**OLD GraphRAG (WRONG):**
```
Query → vecRAG → Top-5 results → Expand → Re-rank
                     ↓
                Wrong 45% of time
```

**NEW LightRAG (CORRECT):**
```
Query → Match to graph concepts → Traverse neighbors → Re-rank
            ↓
        Always correct seed
```

**Key difference:** Seed from QUERY, not from vecRAG results!

---

## Expected Results

| Backend | P@1 (Expected) | P@5 (Expected) | Latency |
|---------|----------------|----------------|---------|
| vecRAG | 55% (baseline) | 75.5% | 0.05ms |
| LightRAG | **60-70%** ✅ | **80-85%** ✅ | 50-100ms |
| Old GraphRAG | 5-10% ❌ | 25-40% ❌ | 518ms |

**Success criteria:** P@1 ≥ 60% (5pp improvement over vecRAG)

---

## Troubleshooting

### Issue: "lightrag backend not available"
**Fix:** Add import and HAS_LIGHTRAG flag to bench.py (see Step 1)

### Issue: Slow query→concept matching
**Fix:** The implementation loads all 2,484 vectors and computes cosine in Python. For production:
1. Add limit to concepts searched (e.g., filter by TMD lane)
2. Use Neo4j vector index (5.0+)
3. Cache frequent query concepts

### Issue: P@1 not improving
**Possible causes:**
1. Query→concept matching not finding good seeds
2. Graph neighbors not relevant
3. Re-ranking not effective

**Debug:**
```python
# Print intermediate steps
retriever = LightRAGStyleRetriever()
query_concepts = retriever._extract_query_concepts(query_text, query_vec, top_k=5)
print("Query concepts:", query_concepts)

neighbors = retriever._get_graph_neighborhood(query_concepts[0])
print("Neighbors:", neighbors[:10])
```

---

## If This Works

### Production Deployment
1. Optimize query→concept matching (Neo4j vector index)
2. Tune parameters:
   - `top_k` for query concepts (currently 3)
   - `hops` for graph traversal (currently 1)
   - `confidence_threshold` for neighbor filtering
3. Combine with TMD re-ranking for best results

### Further Research
1. Multi-hop traversal (2-3 hops with shortcuts)
2. Query expansion with LLM
3. Adaptive parameters based on query type
4. Hybrid: LightRAG for P@1, TMD for P@5

---

## If This Doesn't Work

### Fallback Plan
1. Stick with TMD re-ranking (proven +1.5pp P@5)
2. Improve base vecRAG:
   - Better embeddings (GTR-T5-large, BGE-large)
   - Hard negative mining
   - Query expansion
3. Archive LightRAG for future research

---

## Contact & Questions

**Architecture Questions:** See `docs/PRDs/PRD_GraphRAG_LightRAG_Architecture.md`
**Implementation Questions:** See code comments in `RAG/graphrag_lightrag_style.py`
**Benchmark Questions:** See `RAG/bench.py` and existing backends

**Key Insight to Remember:**
> Don't expand from vecRAG results (wrong 45% of time).
> Extract entities from QUERY and expand from there!
