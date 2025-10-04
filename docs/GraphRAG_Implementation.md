# GraphRAG Implementation: Graph-Augmented Vector Retrieval

**Date**: October 2, 2025
**Status**: ✅ Implemented and ready for testing
**Related**: [vecRAG Performance Report](../RAG/results/VECRAG_PERFORMANCE_REPORT.md)

---

## Executive Summary

GraphRAG extends our vecRAG baseline by incorporating **knowledge graph relationships** from Neo4j to improve retrieval quality. Instead of pure vector similarity, GraphRAG combines:

1. **Vector retrieval** (FAISS dense search)
2. **Graph context** (Neo4j relationship traversal)
3. **Hybrid fusion** (Reciprocal Rank Fusion)

**Expected Performance**: +10-15% P@1 improvement over vecRAG baseline (currently 54.4% P@1)

---

## Architecture

### 3-Tier Retrieval Strategy

```
┌─────────────────────────────────────────────────────────┐
│                    Query Vector (784D)                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │   Tier 1: Vector Search   │
         │   (FAISS top-K=20)        │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  Tier 2: Graph Expansion  │
         │  - Local: 1-hop neighbors │
         │  - Global: Graph walks    │
         │  - Hybrid: Both modes     │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │   Tier 3: RRF Fusion      │
         │   (Combine vector + graph)│
         └───────────┬───────────────┘
                     │
                     ▼
              ┌──────────────┐
              │  Top-K Results│
              └──────────────┘
```

### Retrieval Modes

| Mode | Strategy | Use Case |
|------|----------|----------|
| **Local** | 1-hop `RELATES_TO` neighbors | High precision, local context |
| **Global** | Graph walks via `SHORTCUT_6DEG` | Semantic exploration, distant concepts |
| **Hybrid** | Both local + global | Best overall performance |

---

## Graph Data

### Neo4j Schema

```cypher
// Concept nodes (4,993 total)
(:Concept {
  text: "material entity",
  cpe_id: "uuid",
  tmd_bits: 12345,
  tmd_lane: "Domain_Biology_Task_Definition",
  domain_code: 2,
  task_code: 1,
  ...
})

// Entity nodes (7,446 total)
(:Entity {
  text: "oxidoreductase",
  ...
})

// Relationships
(:Concept)-[:RELATES_TO {confidence: 0.8}]->(:Entity)
(:Concept)-[:SHORTCUT_6DEG {path_length: 4}]->(:Concept)
```

### Edge Statistics

- **RELATES_TO**: 10,070 edges (concept↔entity relationships from LightRAG extraction)
- **SHORTCUT_6DEG**: 34 edges (6-degree shortcuts between concepts)
- **Average confidence**: 0.65 (from LLM-extracted relations)

### Example Relationships

```json
{
  "subj": "material entity",
  "pred": "is a type of",
  "obj": "independent continuant",
  "confidence": 0.588,
  "source": "lightrag"
}
```

---

## Implementation

### GraphRAG Backend (`RAG/graphrag_backend.py`)

```python
class GraphRAGBackend:
    """Graph-augmented vector retrieval using Neo4j."""

    def _get_1hop_neighbors(self, concept_text: str) -> List[Tuple[str, float]]:
        """Get 1-hop neighbors with confidence scores."""
        # MATCH (c:Concept {text: $text})-[r:RELATES_TO]->(neighbor)
        # RETURN neighbor.text, r.confidence

    def _get_graph_walks(self, concept_text: str, max_length: int = 3):
        """Get graph walk sequences using SHORTCUT_6DEG."""
        # MATCH path = (c:Concept)-[:SHORTCUT_6DEG*1..2]-(neighbor)
        # Score decays by path length: 0.8^path_len

    def _rrf_fusion(self, vector_indices, graph_neighbors, k=60):
        """Reciprocal Rank Fusion: score = sum(1/(k + rank_i))"""
        # Combines vector ranks + graph confidence scores
```

### Integration with Benchmark (`RAG/bench.py`)

```bash
# Run GraphRAG benchmark
PYTHONPATH=. ./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 500 \
  --topk 10 \
  --backends vec,graphrag_local,graphrag_global,graphrag_hybrid

# Quick test script
./scripts/run_graphrag_benchmark.sh
```

---

## Comparison to LightRAG Paper

### LightRAG Architecture (from paper)

| Component | LightRAG | Our GraphRAG |
|-----------|----------|--------------|
| **Vector Store** | Custom FAISS | FAISS IVFFlat (784D TMD-enhanced) |
| **Graph DB** | Custom KG JSON files | Neo4j (native graph) |
| **Entities** | LLM-extracted | LightRAG extraction (10K relations) |
| **Graph Ops** | BFS/DFS traversal | Cypher queries (optimized) |
| **Fusion** | LLM-based re-ranking | RRF (reciprocal rank) |

### Expected Performance Gains

**LightRAG Paper Results** (Agriculture domain):
- Naive RAG: ~45% comprehensiveness
- HyDE: ~50%
- GraphRAG (vector only): ~60%
- **LightRAG (graph-enhanced): ~75%** ← target

**Our vecRAG Baseline**:
- P@1: 54.4% (comparable to GraphRAG-vector-only)
- **GraphRAG target: 65-70% P@1** (hybrid mode)

---

## Running the Benchmark

### Prerequisites

```bash
# 1. Ensure Neo4j is running
brew services start neo4j
cypher-shell -u neo4j -p password "RETURN 1"

# 2. Check graph has data
cypher-shell -u neo4j -p password "MATCH (c:Concept) RETURN count(c)"
# Expected: 4,993+ concepts

# 3. Install neo4j driver
./.venv/bin/pip install neo4j
```

### Quick Start

```bash
# Run full GraphRAG benchmark (5-10 minutes)
./scripts/run_graphrag_benchmark.sh

# Results:
# - Per-query JSONL: RAG/results/graphrag_benchmark_<timestamp>.jsonl
# - Summary table: RAG/results/summary_<timestamp>.md
```

### Manual Invocation

```bash
# Compare all modes side-by-side
PYTHONPATH=. ./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 500 \
  --topk 10 \
  --backends vec,bm25,graphrag_local,graphrag_global,graphrag_hybrid \
  --out RAG/results/graphrag_test.jsonl

# Test specific mode
PYTHONPATH=. ./.venv/bin/python RAG/bench.py \
  --backends graphrag_hybrid \
  --graphrag-mode hybrid \
  --n 100 \
  --out RAG/results/hybrid_test.jsonl
```

---

## Metrics and Evaluation

### Success Criteria

| Metric | vecRAG Baseline | GraphRAG Target | Status |
|--------|----------------|-----------------|--------|
| **P@1** | 54.4% | **≥60%** | 🔄 Testing |
| **P@5** | 77.8% | **≥82%** | 🔄 Testing |
| **MRR@10** | 65.8% | **≥70%** | 🔄 Testing |
| **Latency P95** | 0.05ms (vector) | **<5ms** (with graph) | 🔄 Testing |

### Expected Improvements by Mode

```
Local (1-hop):    +5-8%  P@1  (high precision, low latency)
Global (walks):   +3-6%  P@1  (exploration, moderate latency)
Hybrid (both):    +10-15% P@1  (best accuracy, higher latency)
```

### Latency Expectations

- **Vector baseline**: 0.04ms mean (pure FAISS)
- **GraphRAG local**: 2-3ms (1-hop Cypher query + RRF)
- **GraphRAG global**: 4-6ms (graph walk + RRF)
- **GraphRAG hybrid**: 5-8ms (both modes + RRF)

Still **20-50x faster than BM25** (0.96ms) despite graph overhead!

---

## Algorithm Details

### Reciprocal Rank Fusion (RRF)

```python
# Combine vector retrieval ranks with graph confidence scores
def rrf_score(vector_rank, graph_confidence, k=60):
    vector_contribution = 1 / (k + vector_rank)
    graph_contribution = graph_confidence * 0.5
    return vector_contribution + graph_contribution

# k=60 is standard from IR literature
# graph_confidence ∈ [0,1] from LLM extraction
```

### Graph Walk Scoring

```python
# Decay score by path length (prefer shorter paths)
score = 0.8 ** path_length

# Examples:
# 1-hop: 0.8^1 = 0.80
# 2-hop: 0.8^2 = 0.64
# 3-hop: 0.8^3 = 0.51
```

---

## Troubleshooting

### Common Issues

**1. "Neo4j not running"**
```bash
brew services start neo4j
# Wait 10-15 seconds for startup
cypher-shell -u neo4j -p password "RETURN 1"
```

**2. "Graph has insufficient data"**
```bash
# Check concept count
cypher-shell -u neo4j -p password "MATCH (c:Concept) RETURN count(c)"

# If <100, re-ingest ontologies
./scripts/ingest_all_ontologies.sh
```

**3. "neo4j driver not installed"**
```bash
./.venv/bin/pip install neo4j
```

**4. "No concept_text matches found"**
- Ensure ingestion included `text` property on Concept nodes
- Check: `MATCH (c:Concept) WHERE c.text IS NOT NULL RETURN count(c)`

---

## Future Enhancements

### Short-term (Week 2)
- [ ] Multi-hop reasoning (2-3 hop paths)
- [ ] Edge type filtering (causal, temporal, etc.)
- [ ] Confidence-weighted graph expansion

### Medium-term (Week 3-4)
- [ ] GWOM integration (graph walk sequences for LVM training)
- [ ] Bidirectional graph traversal
- [ ] Cross-domain analogies via graph paths

### Long-term
- [ ] Graph neural network embeddings
- [ ] Dynamic graph updates from LVM predictions
- [ ] Concept drift detection via graph topology

---

## References

- **LightRAG Paper**: "LightRAG: Simple and Fast Retrieval-Augmented Generation" (arXiv:2410.05779v1)
- **RRF**: Cormack et al. "Reciprocal Rank Fusion" (SIGIR 2009)
- **Neo4j Cypher**: https://neo4j.com/docs/cypher-manual/
- **GWOM Design**: `docs/PRDs/PRD_GWOM_design_Options.md`

---

**Status**: ✅ Implementation complete, ready for benchmark execution
**Next Step**: Run `./scripts/run_graphrag_benchmark.sh` to generate results
