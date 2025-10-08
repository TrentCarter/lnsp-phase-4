# TMD-LS Implementation Guide

**Version**: 1.0.0
**Date**: 2025-10-08
**Status**: In Progress
**Author**: Trent Carter — True Synthesis AI

---

## Overview

This document tracks the implementation of the **TMD-Lane Specialist (TMD-LS)** architecture for the LNSP system. TMD-LS routes semantic work to specialized 1-3B parameter models based on Task-Modifier-Domain (TMD) vectors, achieving 266% throughput improvement over monolithic 7B models.

**Reference Documents**:
- [PRD_TMD-LS.md](PRD_TMD-LS.md) - Architecture specification
- [TMD-Schema.md](TMD-Schema.md) - Domain/Task/Modifier taxonomy
- [llm_prompts_master.json](../../configs/llm_prompts/llm_prompts_master.json) - Prompt configurations

---

## Implementation Status

### ✅ Completed

1. **Multi-Model Infrastructure** (2025-10-07)
   - Deployed 4 specialist models on separate ports
   - Llama 3.1:8b (port 11434) - Complex reasoning baseline
   - TinyLlama 1.1B (port 11435) - Ultra-fast specialist (284 tok/s)
   - Phi3 Mini 3.8B (port 11436) - Precision specialist (119 tok/s)
   - Granite3 MoE 1B (port 11437) - Low-latency specialist (33ms first token)

2. **Performance Benchmarking** (2025-10-07)
   - Created benchmark suite (tools/benchmark_all_models.py, tools/quick_perf_test.py)
   - Validated 3.87x speedup claim (TinyLlama vs Llama 3.1)
   - Measured first token latency (33ms for TinyLlama/Granite3 vs 143ms for Llama 3.1)

3. **Management Tooling** (2025-10-07)
   - Created scripts/manage_tmd_lanes.sh (start/stop/restart/status/test/logs)
   - Updated docs/howto/how_to_access_local_AI.md with multi-port configuration

4. **Prompt Configuration** (2025-10-08)
   - Created configs/llm_prompts/llm_prompts_master.json with comprehensive prompt library
   - 1x TMD Router prompt (domain/task/modifier extraction)
   - 16x Lane Specialist prompts (one per domain: Science, Math, Tech, etc.)
   - 1x Output Smoothing prompt (LVM post-processing)
   - 1x Ontology Ingestion prompt (SWO/GO/ConceptNet/DBpedia)
   - 1x RL Ingestion prompt (CPESH with feedback loops)

### 🚧 In Progress

5. **TMD Router Component**
   - Implement router service that loads prompts from llm_prompts_master.json
   - Add caching for TMD extractions (concept_text -> TMD codes)
   - Integrate with existing src/llm_tmd_extractor.py

6. **Echo Loop Validator**
   - Implement cosine similarity validation (threshold >= 0.82)
   - Add re-queue logic for failed validations
   - Create validation metrics dashboard

### 📋 Pending

7. **Lane Routing Service**
   - Implement lane selection based on domain_code (0-15)
   - Add load balancing across models with same domain assignment
   - Create fallback routing (Llama 3.1) for edge cases

8. **3→2→1 Ensemble Pipeline**
   - Implement propose/verify/refine pattern per lane
   - Add ensemble voting logic
   - Create quality metrics per ensemble stage

9. **784D Vector Fusion**
   - Implement 768D GTR-T5 + 16D TMD concatenation
   - Update FAISS indexing to support 784D vectors
   - Modify vecRAG retrieval to extract TMD from vectors

10. **Integration with LNSP Pipeline**
    - Update P5 (LLM Interrogation) to use lane specialists
    - Update P11 (Vector Storage) to use lane-based FAISS sub-indices
    - Update P13 (Echo Validation) with per-lane adaptive thresholds
    - Update P17 (Inference Output) with lane-based smoothing

---

## Architecture Components

### 1. TMD Router

**Purpose**: Extract Domain/Task/Modifier codes from concept text to route to appropriate lane specialist.

**Implementation**: `src/tmd_router.py`

**Key Functions**:
```python
def route_concept(concept_text: str) -> Dict:
    """
    Extract TMD codes and route to appropriate lane specialist.

    Returns:
        {
            'domain_code': int (0-15),
            'task_code': int (0-31),
            'modifier_code': int (0-63),
            'lane_specialist': str (model name),
            'port': int (model port)
        }
    """

def load_prompts() -> Dict:
    """Load prompts from configs/llm_prompts/llm_prompts_master.json"""

def cache_tmd(concept_text: str, tmd_dict: Dict):
    """Cache TMD extractions to avoid re-computation"""
```

**Dependencies**:
- `src/llm_tmd_extractor.py` (existing TMD extraction logic)
- `configs/llm_prompts/llm_prompts_master.json` (prompt templates)
- `src/llm/local_llama_client.py` (LLM interface)

**Performance Target**:
- Latency: < 1s per concept (including LLM call)
- Cache hit rate: > 80% after warmup

---

### 2. Echo Loop Validator

**Purpose**: Validate generated CPESH structures by comparing embedding similarity between input concept and generated output.

**Implementation**: `src/echo_loop_validator.py`

**Key Functions**:
```python
def validate_cpesh(
    concept_text: str,
    cpesh: Dict,
    threshold: float = 0.82
) -> Dict:
    """
    Validate CPESH structure using Echo Loop.

    Args:
        concept_text: Original concept text
        cpesh: Generated CPESH structure
        threshold: Cosine similarity threshold (default 0.82)

    Returns:
        {
            'valid': bool,
            'cosine_similarity': float,
            'threshold': float,
            'action': 'accept' | 're_queue' | 'escalate'
        }
    """

def compute_concept_embedding(text: str) -> np.ndarray:
    """Generate 768D GTR-T5 embedding for concept"""

def compute_cpesh_embedding(cpesh: Dict) -> np.ndarray:
    """
    Generate embedding for CPESH structure.
    Combines concept + expected answer embeddings.
    """
```

**Dependencies**:
- `src/vectorizer.py` (GTR-T5 embeddings)
- NumPy/SciPy for cosine similarity
- PostgreSQL for storing validation metrics

**Validation Logic**:
- If `cosine >= 0.85`: Accept immediately (high confidence)
- If `0.82 <= cosine < 0.85`: Accept with monitoring
- If `0.70 <= cosine < 0.82`: Re-queue once for improvement
- If `cosine < 0.70`: Escalate to Llama 3.1:8b for re-generation

---

### 3. Lane Routing Service

**Purpose**: Route concepts to appropriate lane specialist based on domain code and load balancing.

**Implementation**: `src/lane_router.py`

**Key Functions**:
```python
def select_lane(domain_code: int, task_code: int, modifier_code: int) -> Dict:
    """
    Select lane specialist model based on TMD codes.

    Returns:
        {
            'model': str (e.g., 'tinyllama:1.1b'),
            'port': int,
            'specialist_prompt_id': str,
            'temperature': float,
            'max_tokens': int
        }
    """

def check_model_availability(port: int) -> bool:
    """Check if model is running and accepting requests"""

def get_load_metrics(port: int) -> Dict:
    """Get current load (queue depth, avg latency)"""
```

**Routing Table** (from llm_prompts_master.json):

| Domain Code | Domain      | Primary Model      | Port  | Fallback Model |
|-------------|-------------|--------------------|-------|----------------|
| 0           | Science     | tinyllama:1.1b     | 11435 | llama3.1:8b    |
| 1           | Mathematics | phi3:mini          | 11436 | llama3.1:8b    |
| 2           | Technology  | granite3-moe:1b    | 11437 | llama3.1:8b    |
| 3           | Engineering | tinyllama:1.1b     | 11435 | llama3.1:8b    |
| 4           | Medicine    | phi3:mini          | 11436 | llama3.1:8b    |
| 5           | Psychology  | granite3-moe:1b    | 11437 | llama3.1:8b    |
| 6           | Philosophy  | llama3.1:8b        | 11434 | llama3.1:8b    |
| 7           | History     | tinyllama:1.1b     | 11435 | llama3.1:8b    |
| 8           | Literature  | granite3-moe:1b    | 11437 | llama3.1:8b    |
| 9           | Art         | phi3:mini          | 11436 | llama3.1:8b    |
| 10          | Economics   | tinyllama:1.1b     | 11435 | llama3.1:8b    |
| 11          | Law         | llama3.1:8b        | 11434 | llama3.1:8b    |
| 12          | Politics    | phi3:mini          | 11436 | llama3.1:8b    |
| 13          | Education   | granite3-moe:1b    | 11437 | llama3.1:8b    |
| 14          | Environment | tinyllama:1.1b     | 11435 | llama3.1:8b    |
| 15          | Software    | phi3:mini          | 11436 | llama3.1:8b    |

**Load Balancing Strategy**:
1. Check primary model availability
2. If primary unavailable or overloaded (queue > 10): use fallback
3. Track per-lane success rate (Echo Loop validation)
4. Dynamically adjust routing if lane success rate < 75%

---

### 4. 3→2→1 Ensemble Pipeline

**Purpose**: Hierarchical ensemble for high-precision lanes.

**Implementation**: `src/ensemble_pipeline.py`

**Pipeline Stages**:

```python
def ensemble_3_2_1(concept_text: str, domain_code: int) -> Dict:
    """
    Execute 3→2→1 ensemble pipeline for high-precision domains.

    Stage 1 (Propose): 3 models generate candidate CPESH structures
    Stage 2 (Verify): 2 models perform schema + semantic validation
    Stage 3 (Refine): 1 model merges and normalizes
    Stage 4 (Validate): Echo Loop >= 0.82

    Returns:
        {
            'final_cpesh': Dict,
            'ensemble_votes': List[Dict],
            'verification_scores': List[float],
            'echo_score': float
        }
    """
```

**When to Use Ensemble**:
- High-precision domains: Law (11), Medicine (4), Philosophy (6)
- When echo_fail_rate > 7% over last 10k samples
- User-requested high-quality mode

**Performance Impact**:
- Latency: ~1.4x slower than single-model
- Accuracy: +1.5-2% improvement (as per PRD)
- Cost: 3x LLM calls for propose stage

---

### 5. 784D Vector Fusion

**Purpose**: Combine 768D GTR-T5 semantic embeddings with 16D TMD metadata for enhanced retrieval.

**Implementation**: `src/vector_fusion.py`

**Key Functions**:
```python
def create_784d_vector(concept_text: str, tmd_codes: Dict) -> np.ndarray:
    """
    Create fused 784D vector.

    Args:
        concept_text: Input concept
        tmd_codes: {domain_code, task_code, modifier_code}

    Returns:
        784D vector: [16D TMD | 768D GTR-T5]
    """

def extract_tmd_from_vector(vector_784d: np.ndarray) -> Dict:
    """Extract TMD codes from first 16 dimensions"""

def extract_semantic_from_vector(vector_784d: np.ndarray) -> np.ndarray:
    """Extract 768D GTR-T5 embedding from last 768 dimensions"""
```

**Integration with FAISS**:
- Update `src/faiss_index.py` to support 784D vectors
- Modify `build-faiss` target in Makefile
- Update vecRAG retrieval to use 784D search

**TMD-ReRank Algorithm** (from TMD-Schema.md):
```python
# 1. Generate TMD for query
query_tmd = generate_tmd_for_query("software engineering")

# 2. Get top-20 from vecRAG (FAISS)
vec_indices, vec_scores = faiss_search(query_vector, k=20)

# 3. Extract TMD from retrieved vectors (first 16 dims)
result_tmds = corpus_vectors[vec_indices, :16]

# 4. Compute TMD similarity
tmd_similarities = cosine_similarity(query_tmd, result_tmds)

# 5. Combine scores
final_scores = 0.7 * vec_scores + 0.3 * tmd_similarities

# 6. Re-rank and return top-10
reranked_indices = argsort(-final_scores)[:10]
```

---

## Integration Points with LNSP Pipeline

### P5 — LLM Interrogation (Concept Extraction)

**Current State**: Uses single LLM (Llama 3.1) for all CPESH extraction

**TMD-LS Integration**:
```python
# Before
cpesh = extract_cpesh_with_llm(concept_text)

# After
tmd = route_concept(concept_text)  # Get domain/task/modifier
lane = select_lane(tmd['domain_code'], tmd['task_code'], tmd['modifier_code'])
cpesh = extract_cpesh_with_lane_specialist(concept_text, lane)
validated = validate_cpesh(concept_text, cpesh)  # Echo Loop
if not validated['valid']:
    cpesh = requeue_or_escalate(concept_text, cpesh, validated)
```

**Expected Impact**:
- Throughput: 266% improvement (1,100 tok/s vs 300 tok/s baseline)
- Quality: Maintained or improved (due to domain specialization)
- Cost: -72% (using 1-3B models instead of 7B)

---

### P11 — Vector Storage (FAISS Indexing)

**Current State**: 768D GTR-T5 vectors in FAISS

**TMD-LS Integration**:
- Update to 784D vectors (768D semantic + 16D TMD)
- Create lane-specific FAISS sub-indices (optional optimization)
- Add TMD-aware re-ranking at query time

**Implementation Steps**:
1. Re-generate all concept vectors with 784D fusion
2. Rebuild FAISS index with 784D dimension
3. Update `src/faiss_retrieval.py` to handle TMD extraction
4. Add TMD re-ranking to vecRAG pipeline

---

### P13 — Echo Validation (Quality Control)

**Current State**: No automated validation of extracted concepts

**TMD-LS Integration**:
- Implement Echo Loop validator (cosine >= 0.82)
- Add per-lane adaptive thresholds
- Track validation metrics by domain/model

**Validation Metrics to Track**:
- Per-lane echo success rate
- Average cosine similarity by domain
- Re-queue rate by model
- Escalation rate (fallback to Llama 3.1)

**Database Schema** (PostgreSQL):
```sql
CREATE TABLE tmd_validation_metrics (
    id SERIAL PRIMARY KEY,
    concept_id UUID REFERENCES cpe_entry(id),
    domain_code INT,
    task_code INT,
    modifier_code INT,
    model_name VARCHAR(50),
    port INT,
    cosine_similarity FLOAT,
    threshold FLOAT,
    validation_result VARCHAR(20), -- 'accept', 're_queue', 'escalate'
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_validation_domain ON tmd_validation_metrics(domain_code);
CREATE INDEX idx_validation_model ON tmd_validation_metrics(model_name);
```

---

### P17 — Inference Output (LVM Output Smoothing)

**Current State**: Raw LVM vector-to-text output (may have grammatical issues)

**TMD-LS Integration**:
```python
# LVM generates output from vector
lvm_output_text = lvm.decode(output_vector)

# Smooth output using context
smoothed_text = smooth_lvm_output(
    output_text=lvm_output_text,
    input_text_chunk_current=current_chunk,
    input_text_chunk_parent=parent_chunk,
    input_text_chunk_child=child_chunk
)

# Validate smoothing preserved semantics
validation = validate_smoothing(lvm_output_text, smoothed_text)
if validation['cosine_similarity'] < 0.82:
    # Reject smoothing, use raw LVM output
    final_output = lvm_output_text
else:
    final_output = smoothed_text
```

**Prompt**: See `output_smoothing` in llm_prompts_master.json

---

## Testing Strategy

### Unit Tests

1. **TMD Router**:
   - Test prompt loading from JSON
   - Test TMD extraction accuracy (compare to known good examples)
   - Test caching behavior

2. **Echo Loop Validator**:
   - Test embedding generation
   - Test cosine similarity computation
   - Test threshold logic (accept/re-queue/escalate)

3. **Lane Router**:
   - Test domain-to-model mapping
   - Test fallback behavior
   - Test load balancing logic

### Integration Tests

1. **End-to-End CPESH Extraction**:
   - Input: Raw concept text
   - Output: Validated CPESH structure
   - Validate: Echo score >= 0.82, correct domain assignment

2. **784D Vector Fusion**:
   - Input: Concept text + TMD codes
   - Output: 784D vector
   - Validate: Can extract TMD from first 16D, semantic similarity preserved

3. **Multi-Lane Throughput**:
   - Input: 1000 concepts across all 16 domains
   - Measure: Total throughput (concepts/sec)
   - Target: >= 266% improvement over baseline (single Llama 3.1)

### Performance Benchmarks

**Baseline (Current System)**:
- Model: Llama 3.1:8b (single instance)
- Throughput: ~300 tok/s
- CPESH extraction: ~2-3 concepts/sec

**Target (TMD-LS)**:
- Throughput: ~1,100 tok/s (6-lane configuration)
- CPESH extraction: ~8-10 concepts/sec
- Echo validation pass rate: >= 93%

**Test Dataset**:
- 1000 concepts from SWO, GO, ConceptNet, DBpedia
- Balanced across 16 domains
- Include known difficult cases (ambiguous concepts, multi-domain)

---

## Deployment Plan

### Phase 1: Router + Echo Loop (Week 1)

**Goal**: Basic TMD routing and validation working

**Deliverables**:
- [ ] `src/tmd_router.py` implemented
- [ ] `src/echo_loop_validator.py` implemented
- [ ] Unit tests passing
- [ ] Integration with existing `src/llm_tmd_extractor.py`

**Success Criteria**:
- Router correctly assigns domain codes (>90% accuracy on test set)
- Echo Loop validates with <5% false positive rate

---

### Phase 2: Lane Routing + Multi-Model (Week 2)

**Goal**: Distribute work across 4 specialist models

**Deliverables**:
- [ ] `src/lane_router.py` implemented
- [ ] Update P5 (LLM Interrogation) to use lane specialists
- [ ] Load balancing logic working
- [ ] Performance benchmarks completed

**Success Criteria**:
- Throughput >= 800 tok/s (2.67x baseline)
- Echo validation pass rate >= 85%
- All 4 models actively processing requests

---

### Phase 3: 784D Vectors + TMD-ReRank (Week 3)

**Goal**: Enhanced retrieval with TMD metadata

**Deliverables**:
- [ ] `src/vector_fusion.py` implemented
- [ ] Update P11 (Vector Storage) to use 784D vectors
- [ ] FAISS re-indexing with 784D completed
- [ ] TMD re-ranking integrated into vecRAG

**Success Criteria**:
- All vectors stored as 784D (768D + 16D)
- TMD re-ranking improves retrieval precision by >= 5%
- Vector extraction/fusion time < 50ms per concept

---

### Phase 4: 3→2→1 Ensemble (Week 4)

**Goal**: High-precision ensemble for critical domains

**Deliverables**:
- [ ] `src/ensemble_pipeline.py` implemented
- [ ] Ensemble routing for Law, Medicine, Philosophy
- [ ] Quality metrics showing improvement

**Success Criteria**:
- Echo validation pass rate >= 95% for ensemble lanes
- Ensemble accuracy +1.5-2% vs single-model
- Ensemble latency < 1.5x single-model

---

### Phase 5: Production Hardening (Week 5)

**Goal**: Monitoring, observability, reliability

**Deliverables**:
- [ ] Validation metrics dashboard
- [ ] Per-lane performance monitoring
- [ ] Automatic fallback on model failure
- [ ] Documentation complete

**Success Criteria**:
- 99% uptime for router service
- < 1s p95 latency for single-model lanes
- < 1.5s p95 latency for ensemble lanes
- All validation metrics tracked in PostgreSQL

---

## Open Questions

1. **TMD Router Training**: Should we fine-tune a dedicated TMD router model, or continue using prompt-based extraction with Llama 3.1?
   - **Option A**: Fine-tune TinyLlama on TMD classification (faster, ~50ms latency)
   - **Option B**: Keep prompt-based with Llama 3.1 (higher quality, ~500ms latency)

2. **Lane-Specific FAISS Indices**: Should we create 16 separate FAISS indices (one per domain) or one unified 784D index?
   - **Option A**: Unified index (simpler, allows cross-domain retrieval)
   - **Option B**: Per-lane indices (faster, better isolation)

3. **Echo Loop Reference**: What should we compare against for Echo Loop validation?
   - **Option A**: Original concept embedding vs CPESH concept embedding
   - **Option B**: Original concept embedding vs (CPESH concept + expected) combined embedding
   - **Option C**: Original concept embedding vs full CPESH structure embedding

4. **Ensemble Trigger**: When should we auto-trigger 3→2→1 ensemble?
   - **Option A**: Fixed domains (Law, Medicine, Philosophy)
   - **Option B**: Dynamic (when echo_fail_rate > 7% over 10k samples)
   - **Option C**: User-requested (high-quality flag in API)

---

## Success Metrics

### Performance Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Throughput (tok/s) | 300 | 1,100 | Multi-lane benchmark |
| CPESH extraction (concepts/sec) | 2-3 | 8-10 | End-to-end pipeline |
| First token latency (ms) | 143 | 33-50 | Per-model measurement |
| Echo validation pass rate | N/A | 93% | Validation metrics DB |
| Cost per 1M tokens | 1.00x | 0.28x | Model size comparison |

### Quality Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| TMD routing accuracy | N/A | 90% | Manual evaluation |
| Echo Loop cosine similarity | N/A | 0.84 | Average across all lanes |
| Ensemble accuracy improvement | N/A | +1.5-2% | A/B comparison |
| Re-queue rate | N/A | <10% | Validation metrics DB |
| Escalation rate (fallback) | N/A | <3% | Validation metrics DB |

### Operational Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Router uptime | 99% | Service monitoring |
| Model availability | >95% per model | Health check API |
| Lane utilization balance | <20% variance | Load metrics |
| Cache hit rate (TMD) | >80% | Router cache stats |

---

## References

- [PRD_TMD-LS.md](PRD_TMD-LS.md) - Full architecture specification
- [TMD-Schema.md](TMD-Schema.md) - Domain/Task/Modifier taxonomy and encoding
- [llm_prompts_master.json](../../configs/llm_prompts/llm_prompts_master.json) - Prompt library
- [how_to_access_local_AI.md](../howto/how_to_access_local_AI.md) - Multi-model setup guide
- [manage_tmd_lanes.sh](../../scripts/manage_tmd_lanes.sh) - Lane management script

---

**Document Version**: 1.0.0
**Last Updated**: 2025-10-08
**Next Review**: After Phase 1 completion
