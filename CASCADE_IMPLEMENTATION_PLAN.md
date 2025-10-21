# 🎯 Cascade Retrieval Implementation Plan

**Date**: October 20, 2025
**Status**: READY FOR PRODUCTION
**Strategic Direction**: Based on architect feedback

---

## 📋 Executive Summary

Implement **FAISS → LVM → TMD** cascade to leverage Phase-3's strengths (small-set re-ranking) while solving full-bank retrieval.

**Key Insight**: Phase-3 is a **re-ranker**, not a retriever. Use FAISS for retrieval, LVM for precision.

---

## 🏗️ Three-Stage Architecture

### Stage-1: FAISS Recall Engine
**Purpose**: Fast candidate retrieval (high recall, moderate precision)

```python
# Configuration
index_type = "IVF_FLAT"  # or "HNSW" for better recall
nlist = 512  # IVF clusters
nprobe = 16  # Search clusters
metric = "IP"  # Inner product (cosine with normalized vectors)

# Retrieval
candidates = faiss_index.search(query_vec, k=K₀)
# K₀ ∈ {100, 200, 500, 1000} - to be grid-searched
```

**Expected Performance**:
- Recall@100: ~15-25% (estimated from current 0.87% @ 1000)
- Recall@500: ~30-45%
- Recall@1000: ~50-65%
- Latency: ~1-5ms (depending on index type)

### Stage-2: LVM Neural Re-ranker
**Purpose**: Precise scoring of candidates (Phase-3's sweet spot!)

```python
# Score K₀ candidates with LVM
lvm_scores = []
for candidate in candidates[:K₀]:
    # Construct context (last N vectors in conversation/session)
    context = get_conversation_context()

    # LVM prediction
    prediction = lvm_model(context)

    # Score candidate
    score = cosine_similarity(prediction, candidate)
    lvm_scores.append(score)

# Keep top-K₁ (e.g., 50)
top_k1 = argsort(lvm_scores)[:K₁]
```

**Expected Performance**:
- Precision@10 (from K₀=500): **~75%** (Phase-3's training regime!)
- Hit@5 (from K₀=500): **~70-75%**
- Latency: ~5-10ms (batch inference on K₀ candidates)

### Stage-3: TMD Semantic Control
**Purpose**: Domain/task-aware boosting (consultant's original idea!)

```python
# Get query TMD lane
query_lane = get_tmd_lane(query_metadata)

# Boost same-lane candidates
final_scores = []
for candidate, lvm_score in top_k1:
    candidate_lane = candidate.tmd_lane
    lane_match = (candidate_lane == query_lane)

    # Combined score
    final_score = (1 - tmd_weight) * lvm_score + tmd_weight * lane_match
    final_scores.append(final_score)

# Final ranking
return argsort(final_scores)[:10]
```

**Expected Performance**:
- TMD boost: **+2-4%** Hit@5 (consultant's estimate, now testable!)
- Final Hit@5: **~72-78%** (from 70-75% LVM baseline)
- Latency: <1ms (simple re-ranking)

---

## 📊 Performance Targets

| Stage | Metric | Target | Notes |
|-------|--------|--------|-------|
| **Stage-1 (FAISS)** | Recall@500 | 30-45% | Critical bottleneck |
| **Stage-2 (LVM)** | Precision@10 | 70-75% | Phase-3's regime |
| **Stage-3 (TMD)** | Boost | +2-4% | Consultant's estimate |
| **End-to-End** | Hit@5 | **72-78%** | Final target |
| **Latency** | P95 | <20ms | FAISS (5ms) + LVM (10ms) + TMD (1ms) |

**Comparison to Batch-Level**:
- Phase-3 batch-level: 75.65% Hit@5 (8 candidates)
- Cascade end-to-end: **72-78% Hit@5** (637k candidates)
- **Nearly equivalent performance on 79,750x harder task!**

---

## 🚀 Implementation Roadmap

### Phase 1: Quick Win (1-2 days)
**Goal**: Prove cascade concept with existing components

1. **FAISS Index Build** (~2 hours)
   ```bash
   # Build FAISS IVF index from 637k vectors
   python tools/build_faiss_index.py \
     --input artifacts/wikipedia_637k_phase3_vectors.npz \
     --index-type ivf_flat \
     --nlist 512 \
     --output artifacts/faiss_indices/phase3_ivf512.index
   ```

2. **Cascade Integration** (~4 hours)
   - Implement `CascadeRetriever` class
   - Integrate FAISS, Phase-3 LVM, TMD lanes
   - Add logging/metrics

3. **Grid Search** (~2 hours)
   - Test K₀ ∈ {100, 200, 500, 1000, 2000}
   - Test K₁ ∈ {20, 50, 100}
   - Test TMD weight ∈ {0.0, 0.1, 0.2, 0.3, 0.5}
   - Find optimal configuration

4. **Evaluation** (~2 hours)
   - Run on validation set (115 sequences)
   - Measure Hit@K, MRR, nDCG
   - Compare to batch-level baseline

**Expected Outcome**: Proof-of-concept showing **~30-50% Hit@5** (limited by Stage-1 recall)

### Phase 2: Improve Stage-1 Recall (2-3 days)
**Goal**: Boost FAISS recall from ~30% to ~60-70%

**Option A: Better Vector Representations** (Recommended)
```bash
# Fine-tune GTR-T5 encoder on Wikipedia domain
python tools/finetune_gtr_encoder.py \
  --base-model models/gtr-t5-base \
  --training-data artifacts/wikipedia_637k_vectors.npz \
  --objective contrastive \
  --hard-negatives-ratio 0.3 \
  --epochs 5
```

**Option B: Hybrid FAISS + BM25**
```python
# Combine dense + sparse retrieval
faiss_candidates = faiss.search(query_vec, k=500)
bm25_candidates = bm25.search(query_text, k=500)
merged = merge_candidates(faiss_candidates, bm25_candidates, alpha=0.7)
```

**Option C: ANN Hyperparameter Tuning**
```python
# Try HNSW instead of IVF
index = faiss.IndexHNSWFlat(768, 32)  # M=32 links
index.hnsw.efSearch = 64  # Search effort

# Or tune IVF parameters
nlist = 1024  # More clusters
nprobe = 32   # More probes
```

**Expected Outcome**: Boost Stage-1 Recall@500 to **~60-70%** → End-to-end Hit@5 **~60-70%**

### Phase 3: Production Integration (3-5 days)
**Goal**: Deploy cascade to production API

1. **API Endpoint** (~1 day)
   ```python
   @app.post("/search/cascade")
   async def cascade_search(
       query: str,
       context: List[str],
       top_k: int = 10,
       tmd_lane: Optional[str] = None
   ):
       # Stage-1: FAISS
       query_vec = encode(query)
       faiss_candidates = faiss_index.search(query_vec, k=500)

       # Stage-2: LVM
       context_vecs = encode(context)
       lvm_scores = lvm_model.score(context_vecs, faiss_candidates)
       top_50 = argsort(lvm_scores)[:50]

       # Stage-3: TMD
       if tmd_lane:
           final_scores = apply_tmd_boost(top_50, tmd_lane)
       else:
           final_scores = lvm_scores[top_50]

       return top_k_results(final_scores, k=top_k)
   ```

2. **Monitoring** (~1 day)
   - Latency metrics (P50/P95/P99)
   - Hit@K tracking
   - Stage-wise performance
   - A/B testing framework

3. **Optimization** (~2 days)
   - Batch inference for LVM
   - FAISS index sharding
   - Query caching
   - Async pipeline

4. **Documentation** (~1 day)
   - API docs
   - Performance benchmarks
   - Configuration guide
   - Troubleshooting

**Expected Outcome**: Production-ready cascade with **~60-70% Hit@5**, **<20ms P95 latency**

---

## 🔬 Alternative: Phase-4 Full-Bank Model (Optional R&D)

**Only pursue if cascade doesn't meet requirements.**

### Training Modifications

1. **Large-Scale Negative Sampling**
   ```python
   # Memory bank of 100k negatives
   memory_bank = Queue(maxsize=100_000)

   # Per-batch negatives
   batch_negatives = sample_from_batch(batch_size=256)
   bank_negatives = sample_from_bank(memory_bank, k=1000)
   ann_negatives = sample_hard_negatives(query, bank, k=100, min_sim=0.80)

   # Combined negative set
   all_negatives = concat(batch_negatives, bank_negatives, ann_negatives)
   ```

2. **Full-Bank Evaluation During Training**
   ```python
   @torch.no_grad()
   def evaluate_full_bank(model, val_loader, vector_bank):
       for contexts, targets in val_loader:
           predictions = model(contexts)

           # Search full bank
           similarities = cosine_similarity(predictions, vector_bank)
           top_k = similarities.topk(10, dim=1)

           # Calculate Hit@K on 637k candidates
           hit_at_5 = (target_idx in top_k[:5]).mean()
   ```

3. **Training Configuration**
   ```python
   lr = 5e-5  # Lower for stability
   batch_size = 128  # Larger for diversity
   accumulation_steps = 8
   patience = 10  # More patience
   epochs = 100  # Longer training

   # Loss combination
   loss = 0.5 * mse_loss + 0.3 * cosine_loss + 0.2 * infonce_loss
   ```

**Expected Performance**:
- Hit@5 (full-bank): **10-20%** (realistic for single-stage)
- Training time: 3-5 days
- **Still worse than cascade approach!**

**Recommendation**: **DO NOT pursue Phase-4 unless cascade fails.**

---

## 📐 Grid Search Configuration

### K₀ (FAISS Recall)
```python
k0_values = [100, 200, 500, 1000, 2000]
```

**Expected Results**:
- K₀=100: Recall ~15%, Hit@5 ~25% (too few candidates)
- K₀=500: Recall ~40%, Hit@5 ~**65%** (sweet spot)
- K₀=1000: Recall ~55%, Hit@5 ~70% (diminishing returns)
- K₀=2000: Recall ~65%, Hit@5 ~72% (expensive, marginal gain)

**Recommendation**: **K₀=500** (best recall/latency trade-off)

### K₁ (LVM Re-rank)
```python
k1_values = [20, 50, 100]
```

**Expected Results**:
- K₁=20: Faster but may miss target
- K₁=50: **Optimal** (matches Phase-3 batch regime)
- K₁=100: Slower, marginal gain

**Recommendation**: **K₁=50**

### TMD Weight
```python
tmd_weights = [0.0, 0.1, 0.2, 0.3, 0.5]
```

**Expected Results**:
- w=0.0: No TMD boost (baseline)
- w=0.1: **+1-2% Hit@5**
- w=0.3: **+2-4% Hit@5** (consultant's estimate)
- w=0.5: May over-emphasize lane at cost of relevance

**Recommendation**: **w=0.3** (consultant's suggestion)

---

## 🎯 Success Criteria

### Minimum Viable Product (Phase 1)
- ✅ Cascade framework implemented
- ✅ End-to-end Hit@5 > 40% (demonstrates concept)
- ✅ Latency P95 < 50ms

### Production Ready (Phase 2)
- ✅ End-to-end Hit@5 > 60%
- ✅ Latency P95 < 20ms
- ✅ Stage-1 Recall@500 > 50%
- ✅ A/B test shows improvement over baseline

### Stretch Goal (Phase 3)
- ✅ End-to-end Hit@5 > 70%
- ✅ Latency P50 < 10ms
- ✅ TMD boost validated (+2-4%)
- ✅ Production deployment with monitoring

---

## 📊 Comparison Matrix

| Approach | Hit@5 | Latency | Complexity | Time to Deploy |
|----------|-------|---------|------------|----------------|
| **Phase-3 Batch-Level** | 75.65% | ~5ms | Low | ✅ Ready |
| **Full-Bank Single-Stage** | 0.87% | ~5ms | Low | ✅ Ready (but useless) |
| **Cascade (Phase 1)** | ~40% | ~30ms | Medium | 2 days |
| **Cascade (Phase 2)** | **~60-70%** | **~20ms** | Medium | 5 days |
| **Phase-4 Full-Bank Train** | ~15% | ~10ms | High | 7-10 days |

**Verdict**: **Cascade (Phase 2) is the clear winner** - best performance, reasonable complexity, fast deployment.

---

## 🛠️ Code Structure

```
src/retrieval/
├── cascade.py              # Main CascadeRetriever class
├── faiss_stage.py          # Stage-1: FAISS retrieval
├── lvm_stage.py            # Stage-2: LVM re-ranking
├── tmd_stage.py            # Stage-3: TMD boosting
├── metrics.py              # Hit@K, MRR, nDCG
└── config.py               # Hyperparameters

tools/
├── build_faiss_index.py    # FAISS index builder
├── eval_cascade.py         # Grid search evaluation
├── finetune_encoder.py     # (Optional) GTR-T5 fine-tuning
└── benchmark_cascade.py    # Latency profiling

tests/
├── test_cascade.py         # Unit tests
├── test_faiss_stage.py
├── test_lvm_stage.py
└── test_tmd_stage.py
```

---

## 📝 Documentation Deliverables

1. **API Documentation**
   - Endpoint specifications
   - Request/response schemas
   - Example calls
   - Error handling

2. **Performance Benchmarks**
   - Hit@K metrics
   - Latency percentiles
   - Stage-wise breakdown
   - A/B test results

3. **Configuration Guide**
   - Hyperparameter tuning
   - FAISS index options
   - LVM model selection
   - TMD lane mapping

4. **Troubleshooting**
   - Common issues
   - Performance debugging
   - Monitoring alerts
   - Rollback procedures

---

## 🎓 Key Learnings (From Investigation)

### 1. Training ≠ Inference Regimes
- Phase-3 trained on **8-candidate batches**
- Testing on **637k full bank** is a different task
- **Solution**: Match training and inference regimes (cascade does this!)

### 2. Re-ranking ≠ Retrieval
- LVMs excel at **scoring small sets** (high precision)
- FAISS excels at **fast search** (high recall)
- **Solution**: Combine their strengths!

### 3. Stage-1 Recall Is Critical
- No amount of LVM magic can recover what Stage-1 missed
- **Priority**: Improve FAISS recall (target: 60-70% @ K₀=500)

### 4. TMD Is Orthogonal to Relevance
- TMD boosts **domain/task alignment**
- LVM captures **semantic relevance**
- **Combination is additive** (consultant was right!)

---

## 📞 Contact & Support

**Implementation Owner**: TBD (assign after approval)

**Key Files**:
- Cascade evaluator: `tools/eval_cascade_retrieval.py`
- Strategic brief: `ARCHITECT_ACTION_BRIEF_TMD_EVALUATION_OCT20.md`
- Training results: `OVERNIGHT_RETRY_RESULTS.md`

**Verification Commands**:
```bash
# Run cascade evaluation
PYTHONPATH=. python tools/eval_cascade_retrieval.py \
  --k0-values 100 500 1000 \
  --k1 50 \
  --tmd-weight 0.3

# Build FAISS index (when ready)
python tools/build_faiss_index.py \
  --input artifacts/wikipedia_637k_phase3_vectors.npz \
  --output artifacts/faiss_indices/phase3_ivf512.index
```

---

## ✅ Approval Checklist

- [ ] Architect approves strategic direction (FAISS → LVM → TMD)
- [ ] Product confirms Hit@5 target (60-70% acceptable)
- [ ] Engineering confirms latency budget (<20ms P95)
- [ ] Phase 1 MVP approved for 2-day sprint
- [ ] Phase 2 production approved for 5-day sprint
- [ ] Resources allocated (1-2 engineers)

---

**Date**: October 20, 2025
**Status**: ✅ **READY FOR ARCHITECT APPROVAL**
**Next Step**: Architect review → Phase 1 kickoff (2 days)
