# RAG Benchmark Analysis - Ontology Data (Oct 4, 2025)

## Test Configuration
- **Dataset**: Self-retrieval (ontology concepts)
- **Corpus Size**: 4,484 concepts (SWO, GO, ConceptNet, DBpedia)
- **Vector Dimension**: 784D (768D GTR-T5 + 16D TMD)
- **FAISS Index**: IVF-Flat-IP (nlist=112, nprobe=16)
- **Test Queries**: 50 samples (initial run)

## Results Summary

### Performance Comparison

| Backend | P@1     | P@5     | MRR@10  | nDCG@10 | Mean Latency | P95 Latency |
|---------|---------|---------|---------|---------|--------------|-------------|
| **BM25** | **0.660 ✅** | 0.840 ✅ | **0.742 ✅** | **0.767 ✅** | 0.92ms | 1.58ms |
| vecRAG  | 0.600   | 0.840 ✅ | 0.712   | 0.745   | **0.06ms ⚡** | **0.06ms ⚡** |
| Lexical | 0.600   | 0.820   | 0.701   | 0.731   | 0.38ms | 0.52ms |

### Key Findings

1. **BM25 Leads on Precision**:
   - P@1: 0.660 (best overall)
   - P@5: 0.840 (tied with vecRAG)
   - MRR@10: 0.742 (highest mean reciprocal rank)
   - nDCG@10: 0.767 (best relevance scoring)

2. **vecRAG Dominates on Speed**:
   - **15x faster than BM25** (0.06ms vs 0.92ms)
   - **6x faster than Lexical** (0.06ms vs 0.38ms)
   - Sub-millisecond latency enables real-time search
   - P@5 matches BM25 (0.840) despite lower P@1

3. **Lexical (Token Overlap) Baseline**:
   - Competitive P@1 (0.600, same as vecRAG)
   - Slightly lower P@5 (0.820 vs 0.840)
   - 6x slower than vecRAG, 2.4x faster than BM25

## Analysis

### Why BM25 Outperforms on Precision

BM25's strong performance on ontology data likely due to:
- **Exact term matching** works well for structured ontology concepts
- **Concept names are distinctive** (e.g., "material entity", "regulator role")
- **Limited lexical variation** in ontology terminology

### Why vecRAG Shines on Speed

vecRAG's latency advantage comes from:
- **Hardware-accelerated FAISS** (MPS on macOS)
- **IVF quantization** reduces search space
- **No tokenization overhead** (pre-computed vectors)

### Trade-off Analysis

**For Production Use**:
- **Real-time search (<10ms)**: vecRAG (0.06ms) ✅
- **Highest precision**: BM25 (P@1=0.660) ✅
- **Hybrid approach**: vecRAG for speed + BM25 re-ranking

**Latency Budget**:
```
vecRAG:   0.06ms  → 16,667 queries/sec
Lexical:  0.38ms  → 2,632 queries/sec
BM25:     0.92ms  → 1,087 queries/sec
```

## Next Steps

1. **Run 200-query benchmark** with GraphRAG to test graph augmentation
2. **Test CPESH queries** (real-world probe questions vs self-retrieval)
3. **Hybrid retrieval**: vecRAG (top-50) → BM25 re-rank (top-10)
4. **GraphRAG comparison**: Expected +10-15% P@1 boost from graph traversal

## Technical Notes

- Thread limits applied to prevent macOS FAISS segfaults:
  ```bash
  OMP_NUM_THREADS=1
  VECLIB_MAXIMUM_THREADS=1
  FAISS_NUM_THREADS=1
  ```
- Self-retrieval tests ability to find source document for each concept
- P@5=0.840 means 84% of queries found gold document in top-5 results
