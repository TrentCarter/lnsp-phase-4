# Summary: P13-P15 Implementation Plan
**Date**: 2025-09-30
**Status**: Ready for Execution

---

## What We've Accomplished

### ✅ Part A: Updated P1-P12 Pipeline Documentation

**File**: `docs/PRDs/lnsp_lrag_tmd_cpe_pipeline.md`

Added comprehensive status table showing:
- **Completed Stages (P1-P12)**: All operational at 5K scale
- **Current Quality Metrics**: 94.9% CPESH coverage, <1ms searches
- **Identified Gaps**: 257 entries missing negatives (5.1%)
- **Performance Validation**: All 15/15 health checks passed

**Key Updates**:
- P5 now explicitly shows "CPESH" (Concept-Probe-Expected-Soft-Hard negatives)
- P10 updated to reflect two-phase entity resolution
- P11 shows Faiss IVF-Flat + PostgreSQL pgvector integration
- P13 marked as "Partial" - needs full systematic run
- P15-P17 marked as blocked pending GWOM data and P13 completion

---

### ✅ Part B: P13 Echo Validation Script

**File**: `src/pipeline/p13_echo_validation.py`

**Complete implementation** with:

1. **Core Functionality**:
   - Batch processing of all CPE entries
   - Cosine similarity computation (probe → concept)
   - Database updates (echo_score, validation_status)
   - Quality gate assessment (≥90% pass = success)

2. **Reporting**:
   - JSON statistics report
   - Failed entries JSONL export
   - Console summary with color-coded gates
   - Per-lane failure analysis
   - Per-domain score breakdown

3. **Usage Examples**:
```bash
# Full validation run (all 4,993 entries)
python -m src.pipeline.p13_echo_validation \
  --update-db \
  --report-out artifacts/p13_echo_report.json

# Test run (100 entries, no DB update)
python -m src.pipeline.p13_echo_validation \
  --limit 100 \
  --no-update-db

# Custom threshold
python -m src.pipeline.p13_echo_validation \
  --threshold 0.85 \
  --update-db
```

4. **Quality Gates**:
   - ✅ Pass: ≥90% entries with cosine ≥ 0.82
   - ⚠️  Review: 80-90% pass rate
   - ❌ Fail: <80% pass rate

---

### ✅ Part C: P15 Latent-Only LVM Complete Plan

**File**: `docs/PRDs/PRD_P15_Latent_LVM_Implementation_Plan.md`

**100+ page comprehensive plan** covering:

#### 1. Architecture Design
- **Latent-Only Vector Model**: No tokens in or out
- **Input**: Text → vecRAG (GTR-T5) → 768D vector
- **Processing**: Mamba-2 SSM (vector-native)
- **Output**: 768D vector → vecRAG V→T (Faiss + Vec2Text) → Text
- **Smoothing**: Llama 3.1:8b for natural language generation

#### 2. GWOM Training Data Generation
Three parallel methods for ordered concept sequences:

| Method | Target | % of Data | Quality |
|--------|--------|-----------|---------|
| **GraphRAG Walks** | 100K chains | 42% | Neo4j weighted random walks |
| **WikiSearch Anchoring** | 100K chains | 38% | Wikipedia link extraction + validation |
| **Ontology Traversal** | 50K chains | 20% | ConceptNet/DBpedia canonical paths |

**Data Format**:
```jsonl
{
  "seq_id": "uuid",
  "method": "graphrag|wikisearch|ontology",
  "concept_chain": ["C1", "C2", "C3", ...],
  "vectors": [[768D], [768D], ...],
  "tmd_vectors": [[16D], [16D], ...],
  "quality_score": 0.84,
  "coherence_scores": [0.89, 0.85, ...]
}
```

#### 3. LVM Model Options

**Option A: Mamba-2 Hybrid (Recommended)**
- 6 layers, d_model=768, d_state=16
- ~15M parameters, ~500MB memory
- Input: [batch, seq_len, 768]
- Output: [batch, 768] (next concept vector)

**Option B: Vector-MoE**
- 8 experts, top-k=2 routing
- Similar size and performance
- Better for specialized domains

#### 4. Training Procedure

**Curriculum Learning** (3 stages):
1. **Stage 1 (Epochs 1-3)**: Clean data only (coherence ≥0.85)
   - Ontology chains prioritized
   - Learn basic sequential patterns

2. **Stage 2 (Epochs 4-7)**: Mixed quality (coherence ≥0.75)
   - Add GraphRAG + Wiki chains
   - Generalization to noisy data

3. **Stage 3 (Epochs 8-10)**: Full dataset (coherence ≥0.70)
   - All GWOM chains
   - Robustness training

**Loss Function**: Cosine similarity maximization
```python
loss = (1 - cosine_similarity(predicted_vec, target_vec)).mean()
```

#### 5. Inference Pipeline

**End-to-End Flow**:
```
User Query (text)
  ↓
GTR-T5 Encoder → 768D query_vec
  ↓
Faiss Context Retrieval → 5 nearest concepts
  ↓
Mamba LVM → 768D output_vec (predicted next concept)
  ↓
Faiss Lookup (if cosine ≥0.85) → concept_text
  ↓ (fallback)
Vec2Text (JXE + IELab ensemble) → reconstructed text
  ↓
Llama 3.1 LLM Smoother → final fluent response
```

**Latency Target**: <2 seconds end-to-end

#### 6. Evaluation Metrics

| Metric | Target | Gate |
|--------|--------|------|
| Next-vector cosine | ≥0.80 | Pass/Fail |
| Retrieval precision@1 | ≥0.70 | Quality |
| Echo loop similarity | ≥0.82 | System health |
| Vec2Text fallback rate | ≤20% | Efficiency |

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1) ← **YOU ARE HERE**
- [x] Review and update P1-P12 documentation
- [x] Create comprehensive P15 implementation plan
- [x] Create P13 echo validation script
- [ ] **Next**: Execute P13 on all 4,993 entries
- [ ] Fix 257 missing CPESH negatives (or accept as noise)
- [ ] Set up GWOM data lake structure

### Phase 2: GWOM Data Generation (Week 2-3)
- [ ] Implement GraphRAG walks (`src/gwom/graphrag_walks.py`)
- [ ] Implement WikiSearch anchoring (`src/gwom/wikisearch_anchor.py`)
- [ ] Implement Ontology traversal (`src/gwom/ontology_traverse.py`)
- [ ] Generate 250K training chains
- [ ] Validate GWOM quality (coherence ≥0.75 average)

### Phase 3: LVM Training (Week 4-5)
- [ ] Implement Mamba-2 architecture (`src/lvm/models.py`)
- [ ] Create training dataset loader (`src/lvm/dataset.py`)
- [ ] Run training loop (10 epochs, 3-stage curriculum)
- [ ] Evaluate on held-out test set
- [ ] Save trained model (~15-100MB)

### Phase 4: Inference Integration (Week 6)
- [ ] Build inference pipeline (`src/lvm/inference.py`)
- [ ] Integrate Vec2Text fallback (JXE + IELab)
- [ ] Add LLM smoothing layer (Llama 3.1)
- [ ] End-to-end testing
- [ ] Deploy API endpoint

### Phase 5: Evaluation & Production (Week 7-8)
- [ ] Benchmark against baselines (retrieval-only, GPT-4)
- [ ] A/B testing with human evaluators
- [ ] Performance optimization (quantization, caching)
- [ ] Production deployment
- [ ] Monitoring dashboards

---

## Key Decisions Made

### 1. Architecture: Mamba-2 over Pure Transformer
**Rationale**:
- Better for long sequences (GWOM chains 5-15 concepts)
- Lower computational cost (linear vs quadratic)
- State-space models proven for sequential prediction
- Hybrid attention for long-range dependencies

### 2. Training Data: GWOM Multi-Source
**Rationale**:
- GraphRAG: Local nuance from our CPESH graph
- WikiSearch: External knowledge grounding
- Ontology: Logical rigor and canonical orderings
- Balanced 42%/38%/20% mix for robustness

### 3. Output: vecRAG V→T with Vec2Text Fallback
**Rationale**:
- Primary: Faiss lookup leverages existing CPESH corpus
- Fallback: Vec2Text generates novel concepts not in database
- Ensemble (JXE + IELab) for quality
- LLM smoothing ensures fluent final output

### 4. Training: Curriculum Learning (3 stages)
**Rationale**:
- Start with clean data (ontology chains)
- Gradually add noise (GraphRAG, Wiki)
- Proven to improve convergence and generalization

### 5. No Token Layer
**Rationale**:
- Eliminates tokenization bottleneck
- Pure vector-to-vector processing
- Matches latent space philosophy
- vecRAG provides T↔V interface

---

## Success Criteria Summary

### Training Success
- ✅ 250K GWOM chains generated (≥0.75 coherence)
- ✅ LVM achieves ≥0.80 next-vector cosine
- ✅ Training completes <12 hours (single GPU)
- ✅ Model size ≤100MB

### Inference Success
- ✅ End-to-end latency <2 seconds
- ✅ Faiss retrieval precision@1 ≥70%
- ✅ Vec2Text fallback ≤20%
- ✅ Human ratings ≥80% "good" or "excellent"

### System Integration
- ✅ P13 validation pass rate ≥90%
- ✅ GWOM data lake operational with rotation
- ✅ Inference API deployed and stress-tested
- ✅ Monitoring dashboards live

---

## Files Created/Updated

### New Files
1. `docs/PRDs/PRD_P15_Latent_LVM_Implementation_Plan.md` (this plan)
2. `src/pipeline/p13_echo_validation.py` (P13 implementation)
3. `docs/PRDs/SUMMARY_P13_P15_Implementation_Plan.md` (this summary)

### Updated Files
1. `docs/PRDs/lnsp_lrag_tmd_cpe_pipeline.md` (added P1-P12 status table)

### Files to Create (Phase 2-4)
1. `src/gwom/graphrag_walks.py` - GraphRAG sequence generator
2. `src/gwom/wikisearch_anchor.py` - Wikipedia anchoring
3. `src/gwom/ontology_traverse.py` - Ontology traversal
4. `src/gwom/generate_all.py` - GWOM orchestrator
5. `src/gwom/vectorize_chains.py` - Pre-compute vectors
6. `src/lvm/models.py` - Mamba-2 architecture
7. `src/lvm/train_latent_mamba.py` - Training loop
8. `src/lvm/inference.py` - Inference pipeline
9. `src/lvm/dataset.py` - GWOM dataset loader

---

## Next Immediate Actions

### 1. Execute P13 (Next 24 hours)
```bash
# Test run first (100 entries)
python -m src.pipeline.p13_echo_validation \
  --limit 100 \
  --no-update-db

# Full run if test passes
python -m src.pipeline.p13_echo_validation \
  --update-db \
  --report-out artifacts/p13_echo_report.json
```

**Expected Outcome**:
- Validation report with pass/fail statistics
- Database updated with echo_score and validation_status
- Identification of low-quality lanes for potential re-interrogation

### 2. Address Missing Negatives (24-48 hours)
Two options for 257 entries missing soft/hard negatives:

**Option A (Quick)**: Accept as noise
- 5.1% gap is acceptable for initial training
- Mark as low-priority curriculum data
- Re-interrogate later if needed

**Option B (Thorough)**: Re-run P5 on failed entries
```bash
# Export failed CPE IDs
psql lnsp -c "COPY (SELECT cpe_id FROM cpe_entry WHERE jsonb_array_length(soft_negatives) = 0) TO '/tmp/failed_cpe_ids.txt';"

# Re-run extraction with better prompts
python -m src.ingest_factoid \
  --reprocess-ids /tmp/failed_cpe_ids.txt \
  --force-regenerate-negatives
```

**Recommendation**: Start with Option A, revisit after P13 results

### 3. Set Up GWOM Infrastructure (Week 1)
```bash
# Create directory structure
mkdir -p artifacts/gwom/{gwom_segments,gwom_index}

# Initialize GWOM active log
touch artifacts/gwom/gwom_active.jsonl

# Create manifest
echo '{"version": "0.1", "created": "'$(date -Iseconds)'", "segments": []}' > artifacts/gwom/gwom_manifest.jsonl

# Initialize index database
sqlite3 artifacts/gwom/gwom_index.db "CREATE TABLE sequences (seq_id TEXT PRIMARY KEY, segment TEXT, method TEXT, length INT, quality_score REAL, created_at TEXT);"
```

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation Status |
|------|------------|--------|------------------|
| P13 pass rate <80% | Medium | High | ⚠️  Will know after P13 run |
| GWOM chains too noisy | Medium | Medium | ✅ Multi-stage curriculum addresses this |
| LVM doesn't learn | Low | Critical | ✅ Start with toy task validation |
| Vec2Text quality poor | Low | Medium | ✅ Ensemble JXE+IELab |
| Training time >12hrs | Medium | Low | ⏳ Can scale to multi-GPU if needed |
| Missing negatives hurt training | Low | Medium | ✅ 94.9% coverage likely sufficient |

---

## Questions for Review

### Architecture Questions
1. ✅ **Mamba-2 vs pure Mamba**: Chose Mamba-2 for hybrid SSM+attention
2. ✅ **Model size**: 10-100M params (start small at 15M)
3. ✅ **Vec2Text fallback threshold**: 0.85 cosine similarity

### Data Questions
1. ⏳ **GWOM method weights**: Use 42%/38%/20% or adjust based on quality?
2. ⏳ **Missing negatives**: Accept 5.1% gap or re-run P5?
3. ✅ **Curriculum stages**: 3 stages with increasing noise

### Training Questions
1. ✅ **Loss function**: Cosine similarity maximization
2. ✅ **Learning rate**: Start with 1e-4, cosine decay
3. ✅ **Batch size**: 64 (adjustable based on GPU memory)

### Deployment Questions
1. ⏳ **Hosting**: Local GPU server or cloud deployment?
2. ⏳ **Latency SLO**: <2s end-to-end acceptable?
3. ⏳ **Monitoring**: Which metrics most critical for production?

---

## Conclusion

We have completed a **comprehensive planning phase** for P13-P17:

✅ **Part A Complete**: P1-P12 status documented with 5K scale validation
✅ **Part B Complete**: P13 echo validation script ready to execute
✅ **Part C Complete**: P15 Latent-Only LVM fully designed with detailed implementation plan

**Ready for execution** starting with P13 validation run.

**Estimated Timeline**: 6-8 weeks to production-ready P15 LVM inference

**Next Milestone**: P13 completion with ≥90% pass rate (Target: 48 hours)

---

**Document Version**: 1.0
**Last Updated**: 2025-09-30
**Status**: Approved for Execution
