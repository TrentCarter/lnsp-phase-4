# LNSP Enums (synced with src/enums.py 2025-09-23)

## Pipeline
- PIPELINE_FACTOID_WIKI
- PIPELINE_GENERIC_DOCS

## Lane
- L1_FACTOID      # curated factoids (10k curated set)
- L2_PASSAGE      # passage-level retrieval
- L3_SYNTHESIS    # synthesized answers/summaries
- L4_DEBUG        # debug/development lane

## Embedding
- EMB_MINILM_L6_384      # MiniLM-L6-v2 384D
- EMB_GTR_T5_BASE_768    # GTR-T5-base 768D (default)
- EMB_STELLA_EN_400M_768 # STELLA-EN-400M 768D
- EMB_NV_NEMO_1024       # NVIDIA NeMo 1024D

## VecStore
- VEC_FAISS        # Primary FAISS vector store
- VEC_NANO_DB      # Reserved for LightRAG experiments

## FaissIndex
- FAISS_FLAT       # Flat index for small datasets
- FAISS_IVF_FLAT   # IVF-Flat for 1k-10k (default)
- FAISS_IVF_PQ     # Product Quantization for >10k
- FAISS_HNSW       # HNSW for low-latency

## GraphStore
- GRAPH_NEO4J      # Primary Neo4j graph store
- GRAPH_NETWORKX   # In-memory NetworkX for experiments

## RetrievalMode
- R_SIMPLE         # Dense vector only
- R_HYBRID_LRAG    # LightRAG hybrid (dense + graph)
- R_HYBRID_BM25VEC # BM25 + vector hybrid

## Reranker
- RR_NONE           # No reranking
- RR_COSINE_TOPK    # Cosine similarity rerank
- RR_COSINE_MM      # Cosine similarity with max-marginal

## ArtifactKind
- ART_CHUNKS_JSONL  # Processed chunks
- ART_EMB_NPZ       # Numpy vector arrays
- ART_FAISS_INDEX   # FAISS index files
- ART_LRAG_DB       # LightRAG database
- ART_EVAL_JSONL    # Evaluation results
- ART_REPORT_MD     # Evaluation reports

## Status
- OK    # Success
- WARN  # Warning/partial success
- FAIL  # Failure

## TMD Bit-Packing Layout (uint16)

**16-bit TMD encoding for lane routing:**

```
[15..12] = Domain (4 bits, 0-15)
[11..7]  = Task (5 bits, 0-31)
[6..1]   = Modifier (6 bits, 0-63)
[0]      = Spare (1 bit, reserved)
```

**Lane Index Extraction:**
- `lane_index = (tmd_bits >> 1) & 0x7FFF` (15-bit lane space: 0-32767)

**Example:**
- Domain=2, Task=5, Modifier=10 → `tmd_bits = (2<<12) | (5<<7) | (10<<1) = 8832`
- Lane index = `8832 >> 1 = 4416`

This deterministic mapping ensures consistent lane routing across all components.
