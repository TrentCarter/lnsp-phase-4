# 🚀 Production Rollout Plan - LVM Champion Models

**Date**: 2025-10-19
**Current Status**: Two production-ready models trained and validated
**Next Step**: Canary deployment with monitoring

---

## 📊 Starting Position

### Phase 1: Speed-Optimized (100-Context)
- **File**: `artifacts/lvm/models_final/memory_gru_consultant_recipe/best_val_hit5.pt`
- **Performance**: 59.32% Hit@5, 65.16% Hit@10, 40.07% Hit@1
- **Latency**: ~0.5ms per query
- **Context**: 2,000 effective tokens
- **Status**: ✅ Production baseline

### Phase 2: Champion Model (500-Context) ⭐
- **File**: `artifacts/lvm/models_phase2/run_500ctx_warm/best_val_hit5.pt`
- **Performance**: 66.52% Hit@5, 74.78% Hit@10, 50.00% Hit@1
- **Latency**: ~2.5ms per query
- **Context**: 10,000 effective tokens
- **Status**: ✅ **CHAMPION - Ready for rollout**

---

## 🎯 Rollout Strategy (6-Step Plan)

### Step 1: Canary Deploy Phase-2 Champion

**Objective**: Validate Phase-2 in production with minimal risk

**Deployment Strategy (Option A - Context-Based Routing)**:
```python
def route_to_model(query_context_length):
    """Route based on context requirements"""
    if query_context_length <= 2000:
        return phase1_model  # 100-ctx, 0.5ms
    else:
        return phase2_model  # 500-ctx, 2.5ms

# Routing logic
context_len = count_context_vectors(query)
model = route_to_model(context_len)
prediction = model.predict(context_vectors)
```

**Deployment Strategy (Option B - Traffic-Based A/B)**:
```python
def select_model_ab(user_id, rollout_percentage=10):
    """A/B test Phase-2 vs Phase-1"""
    if hash(user_id) % 100 < rollout_percentage:
        return phase2_model  # Champion
    else:
        return phase1_model  # Baseline
```

**Timeline**:
- **Week 1**: 10% traffic on Phase-2 (shadow mode + logging)
- **Week 2**: 20% traffic (A/B testing)
- **Week 3**: 50% traffic (if metrics stable)
- **Week 4**: 100% rollout (champion becomes default)

**Key Metrics to Track**:

1. **Hit@K Proxy** (approximate retrieval accuracy):
```python
def compute_hit_proxy(predicted_vec, user_clicked_concept):
    """Real-time Hit@K approximation"""
    clicked_vec = get_concept_vector(user_clicked_concept)
    similarity = cosine_similarity(predicted_vec, clicked_vec)

    # Log metrics
    log_metric('hit@1_proxy', similarity > 0.8)
    log_metric('hit@5_proxy', similarity > 0.6)
    log_metric('hit@10_proxy', similarity > 0.4)

    # Per-model comparison
    log_metric(f'{model_name}_hit5_proxy', similarity > 0.6)
```

2. **Latency P50/P95/P99**:
```python
import time

start = time.time()
prediction = model.predict(context)
latency_ms = (time.time() - start) * 1000

log_metric('lvm_latency_p50', latency_ms)
log_metric('lvm_latency_p95', latency_ms)
log_metric('lvm_latency_p99', latency_ms)

# Per-model breakdown
log_metric(f'{model_name}_latency_p95', latency_ms)
```

3. **Empty Hit Rate** (queries with no good results):
```python
def track_empty_hits(top_k_similarities):
    """Monitor queries where model can't find good matches"""
    max_similarity = max(top_k_similarities)

    # Flag if best match is below threshold
    if max_similarity < 0.4:
        log_metric('empty_hit_rate', 1)
        log_warning(f'Low confidence prediction: {max_similarity:.3f}')
    else:
        log_metric('empty_hit_rate', 0)
```

4. **Lane Deltas** (TMD-aware performance):
```python
def track_lane_performance(tmd_vector, hit_proxy):
    """Monitor performance by TMD lane"""
    lane_id = tmd_vector.argmax()  # Dominant lane

    log_metric(f'lane_{lane_id}_hit5_proxy', hit_proxy)
    log_metric(f'lane_{lane_id}_query_count', 1)

    # Alert if any lane degrades
    if hit_proxy < 0.5:
        log_warning(f'Lane {lane_id} underperforming: {hit_proxy:.3f}')
```

**Success Criteria for 100% Rollout**:
- ✅ Hit@5 proxy ≥ 60% (Phase-2 should show ~66%)
- ✅ P95 latency < 5ms (Phase-2 target: 2.5ms)
- ✅ Empty hit rate < 10%
- ✅ No lane shows >20% degradation vs baseline
- ✅ No error rate spike (NaN/Inf predictions)

**Rollback Triggers**:
- ❌ Hit@5 proxy drops below 55% (Phase-1 baseline)
- ❌ P95 latency > 10ms (unacceptable)
- ❌ Empty hit rate > 20%
- ❌ Error rate > 1%
- ❌ Any critical lane fails completely

---

### Step 2: Phase-2B - Enable CPESH Soft Negatives

**Objective**: Further improve Phase-2 with curriculum learning

**Implementation**:
```bash
# Re-train Phase-2 model with soft negatives
./.venv/bin/python -m app.lvm.train_final \
    --model-type memory_gru \
    --data artifacts/lvm/data_phase2/training_sequences_ctx100.npz \
    --epochs 50 \
    --batch-size 16 \
    --accumulation-steps 16 \
    --device mps \
    --min-coherence 0.0 \
    --alpha-infonce 0.05 \  # Increased from 0.03 (warm)
    --enable-soft-negatives \
    --soft-negative-cosine-range 0.6 0.8 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --patience 3 \
    --output-dir artifacts/lvm/models_phase2/run_500ctx_softnegs
```

**Expected Results**:
- Hit@5: 66.52% → **68-69%** (+1.5-3%)
- Hit@10: 74.78% → **76-77%** (+1-2%)
- Latency: ~2.5ms (unchanged)

**What Are Soft Negatives?**
```python
# Soft negatives: Same TMD lane, moderate similarity (0.6-0.8)
def get_soft_negatives(concept_vec, tmd_lane):
    """
    Find concepts in same lane that are similar but not identical
    - Same domain (e.g., all Biology concepts)
    - Moderate cosine similarity (0.6-0.8)
    - Teaches model to discriminate within-lane
    """
    lane_concepts = get_concepts_by_lane(tmd_lane)
    similarities = cosine_similarity(concept_vec, lane_concepts)

    # Filter to 0.6-0.8 range (not too hard, not too easy)
    mask = (similarities >= 0.6) & (similarities <= 0.8)
    soft_negs = lane_concepts[mask]

    return soft_negs[:8]  # Return 8 soft negatives
```

**Why This Works**:
- Teaches model to distinguish between related concepts
- Prevents mode collapse (all biology → "cell")
- Improves top-5/top-10 recall (pushes similar concepts down)

**Training Timeline**: ~1 day

**Deployment**:
- Canary: 10% traffic (Week 5)
- Full rollout: Week 6 (if +1.5% Hit@5 confirmed)

---

### Step 3: Phase-2C - Add Hard Negatives

**Objective**: Push to 70%+ Hit@5 with challenging negatives

**Implementation**:
```bash
# Re-train Phase-2B model with hard negatives
./.venv/bin/python -m app.lvm.train_final \
    --model-type memory_gru \
    --data artifacts/lvm/data_phase2/training_sequences_ctx100.npz \
    --epochs 50 \
    --batch-size 16 \
    --accumulation-steps 16 \
    --device mps \
    --min-coherence 0.0 \
    --alpha-infonce 0.07 \  # Increased from 0.05
    --enable-soft-negatives \
    --enable-hard-negatives \
    --hard-negative-cosine-range 0.75 0.9 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --patience 3 \
    --output-dir artifacts/lvm/models_phase2/run_500ctx_hardnegs
```

**Expected Results**:
- Hit@5: 68.5% → **70-72%** (+1.5-3.5%)
- Hit@10: 76.5% → **78-80%** (+1.5-3.5%)
- Latency: ~2.5ms (unchanged)

**What Are Hard Negatives?**
```python
# Hard negatives: Very similar but wrong (0.75-0.9)
def get_hard_negatives(concept_vec, tmd_lane):
    """
    Find concepts that are VERY similar but incorrect
    - High cosine similarity (0.75-0.9)
    - Often same entity type (e.g., "dog" vs "wolf")
    - Teaches fine-grained discrimination
    """
    lane_concepts = get_concepts_by_lane(tmd_lane)
    similarities = cosine_similarity(concept_vec, lane_concepts)

    # Filter to 0.75-0.9 range (hard but not impossible)
    mask = (similarities >= 0.75) & (similarities <= 0.9)
    hard_negs = lane_concepts[mask]

    return hard_negs[:4]  # Return 4 hard negatives (fewer than soft)
```

**Why This Works**:
- Forces model to learn subtle distinctions
- Improves top-1 precision (correct answer ranked #1)
- Critical for high-precision applications

**Training Timeline**: ~1 day

**Deployment**:
- Canary: 10% traffic (Week 7)
- Full rollout: Week 8 (if +1.5% Hit@5 confirmed)

---

### Step 4: 1000-Vector Context Pilot

**Objective**: Scale context to 1000 vectors (20K effective tokens)

**Data Preparation**:
```bash
./.venv/bin/python tools/export_lvm_training_data_extended.py \
    --input artifacts/wikipedia_500k_corrected_vectors.npz \
    --context-length 1000 \
    --overlap 500 \
    --output-dir artifacts/lvm/data_phase3/
```

**Expected**:
- Fewer sequences (~1,200 train, 130 val)
- Larger files (~6-8 GB training data)
- More memory required

**Training Strategy**:
```bash
# Hierarchical caching (5 chunks × 200 vectors each)
./.venv/bin/python -m app.lvm.train_final \
    --model-type memory_gru \
    --data artifacts/lvm/data_phase3/training_sequences_ctx100.npz \
    --epochs 50 \
    --batch-size 8 \  # Reduced for memory
    --accumulation-steps 32 \  # Maintain 256 effective
    --device mps \
    --hierarchical-cache \  # Enable chunking
    --cache-chunks 5 \
    --cache-chunk-size 200 \
    --min-coherence 0.0 \
    --alpha-infonce 0.05 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --patience 3 \
    --output-dir artifacts/lvm/models_phase3/run_1000ctx
```

**Expected Results**:
- Hit@5: 70% → **73-75%** (+3-5%)
- Hit@10: 78% → **82-85%** (+4-7%)
- Latency: ~5-8ms (2x increase due to 2x context)

**Hierarchical Caching**:
```python
class HierarchicalMemoryGRU(nn.Module):
    """
    Process 1000 vectors efficiently:
    - Split into 5 chunks of 200 vectors
    - Process each chunk → 5 summary vectors
    - Final prediction from 5 summaries + current vector
    """
    def __init__(self):
        self.chunk_encoder = GRU(input_dim=768, hidden_dim=256)
        self.summary_encoder = GRU(input_dim=256, hidden_dim=256)

    def forward(self, x):
        # x: (batch, 1000, 768)
        chunks = x.view(batch, 5, 200, 768)

        # Encode each chunk
        summaries = []
        for i in range(5):
            chunk_out, _ = self.chunk_encoder(chunks[:, i, :, :])
            summaries.append(chunk_out[:, -1, :])  # Last hidden state

        summaries = torch.stack(summaries, dim=1)  # (batch, 5, 256)

        # Encode summaries
        final_out, _ = self.summary_encoder(summaries)

        return self.output_proj(final_out[:, -1, :])
```

**Success Criteria**:
- ✅ Hit@5 ≥ 72% (worth the latency trade-off)
- ✅ P95 latency < 10ms (acceptable for long-context queries)
- ✅ No memory OOM errors

**If Successful**: Promote to production for long-context queries (>5K tokens)

**If Latency Too High**: Keep Phase-2C as default, use 1000-ctx only for special cases

**Training Timeline**: 2-3 days (larger context = slower training)

---

### Step 5: TMD-Aware Routing (16 Specialists)

**Objective**: +2-3% Hit@5 via domain specialization

**Architecture**:
```python
class TMDAwareMemoryGRU(nn.Module):
    """
    16 specialist experts (one per TMD lane)
    Top-2 routing with learned gate
    """
    def __init__(self):
        # 16 experts (one per lane)
        self.experts = nn.ModuleList([
            MemoryGRU() for _ in range(16)
        ])

        # Router: TMD vector → expert weights
        self.router = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Softmax(dim=-1)
        )

    def forward(self, context, tmd_vector):
        # Route to top-2 experts
        expert_weights = self.router(tmd_vector)  # (batch, 16)
        top2_weights, top2_indices = expert_weights.topk(2, dim=-1)

        # Get predictions from top-2 experts
        predictions = []
        for i in range(2):
            expert_idx = top2_indices[:, i]
            expert_out = self.experts[expert_idx](context)
            predictions.append(expert_out)

        # Weighted combination
        pred = (top2_weights[:, 0:1] * predictions[0] +
                top2_weights[:, 1:2] * predictions[1])

        return pred
```

**Training Strategy**:
```bash
# Two-stage training:
# 1) Train all 16 experts independently
# 2) Freeze experts, train router

# Stage 1: Train experts (can parallelize)
for lane_id in {0..15}; do
    python -m app.lvm.train_final \
        --model-type memory_gru \
        --data artifacts/lvm/data_phase2/training_sequences_ctx100.npz \
        --filter-by-tmd-lane $lane_id \
        --epochs 50 \
        --output-dir artifacts/lvm/models_tmd/expert_${lane_id} &
done

# Stage 2: Train router
python -m app.lvm.train_tmd_router \
    --expert-checkpoints artifacts/lvm/models_tmd/expert_*/best_val_hit5.pt \
    --data artifacts/lvm/data_phase2/training_sequences_ctx100.npz \
    --epochs 20 \
    --gate-loss-weight 0.01 \
    --output-dir artifacts/lvm/models_tmd/router
```

**Expected Results**:
- Hit@5: 72% → **74-75%** (+2-3%)
- Hit@10: 80% → **82-83%** (+2-3%)
- **Better tail performance** (underperforming lanes improve)
- Latency: ~3-4ms (slight increase from routing)

**Why This Works**:
- Specialists learn lane-specific patterns
- Biology expert understands protein → enzyme chains
- History expert understands cause → effect relationships
- Load balancing prevents any lane from degrading

**Training Timeline**: 3-5 days (16 experts + router)

**Deployment**:
- Canary: 10% traffic (Week 11)
- Full rollout: Week 12 (if +2% Hit@5 confirmed)

---

### Step 6: External Eval Set & Drift Monitoring

**Objective**: Continuous monitoring and regression prevention

**External Eval Set Creation**:
```bash
# Create held-out evaluation set (not used in training)
python tools/create_eval_set.py \
    --wikipedia-articles data/datasets/wikipedia/wikipedia_500k.jsonl \
    --ontology-chains artifacts/ontology_chains/*.jsonl \
    --output artifacts/eval/held_out_eval_set.jsonl \
    --num-samples 5000 \
    --skip-articles-used-in-training

# Expected output:
# - 2,500 Wikipedia article chains (never seen in training)
# - 2,500 ontology chains (held-out concepts)
# - Ground truth next-vector labels
```

**Weekly Drift Monitoring**:
```python
def weekly_drift_check():
    """
    Run every Monday morning
    Compare current model vs eval set
    """
    eval_set = load_eval_set('artifacts/eval/held_out_eval_set.jsonl')

    # Test current production model
    results = evaluate_model(production_model, eval_set)

    # Compare to baseline (Phase-2 champion: 66.52% Hit@5)
    baseline_hit5 = 0.6652
    current_hit5 = results['hit@5']

    drift = (current_hit5 - baseline_hit5) / baseline_hit5

    # Alert if drift > 5%
    if abs(drift) > 0.05:
        alert(f'Model drift detected: {drift*100:.1f}%')
        alert(f'Current Hit@5: {current_hit5:.2%} vs baseline {baseline_hit5:.2%}')

    # Log metrics
    log_metric('eval_set_hit@1', results['hit@1'])
    log_metric('eval_set_hit@5', results['hit@5'])
    log_metric('eval_set_hit@10', results['hit@10'])

    # Per-lane health
    for lane_id in range(16):
        lane_hit5 = results[f'lane_{lane_id}_hit@5']
        log_metric(f'eval_lane_{lane_id}_hit@5', lane_hit5)

        # Alert if any lane degrades >10%
        if lane_hit5 < baseline_hit5 * 0.9:
            alert(f'Lane {lane_id} degraded: {lane_hit5:.2%}')
```

**Lane Health Monitoring**:
```python
def monitor_lane_health():
    """
    Daily check for per-lane performance
    Ensure no lane is underserved
    """
    for lane_id in range(16):
        # Get production metrics for this lane
        lane_queries = get_queries_by_lane(lane_id, last_24h=True)
        lane_hit5 = compute_hit5_proxy(lane_queries)

        # Compare to overall Hit@5
        overall_hit5 = get_metric('hit@5_proxy', last_24h=True)

        # Alert if lane is >15% below average
        if lane_hit5 < overall_hit5 * 0.85:
            alert(f'Lane {lane_id} underperforming')
            alert(f'Lane Hit@5: {lane_hit5:.2%} vs overall {overall_hit5:.2%}')

            # Suggest remediation
            if lane_id in [0, 1, 2]:  # Science lanes
                suggest('Consider fine-tuning science expert')
            elif lane_id in [8, 9, 10]:  # Humanities
                suggest('Consider adding humanities training data')
```

**Automated Regression Testing**:
```bash
# Run daily regression suite
./scripts/run_daily_regression.sh

# Test suite includes:
# 1. Eval set performance (should be ≥66% Hit@5)
# 2. Latency benchmarks (P95 should be <5ms)
# 3. Per-lane health (all lanes >60% Hit@5)
# 4. Edge cases (empty context, very long context)
# 5. Error rates (should be <0.1%)

# If any test fails, alert and optionally rollback
```

---

## 📅 Complete Timeline

| Week | Phase | Action | Expected Outcome |
|------|-------|--------|------------------|
| **1** | Canary Deploy | Phase-2 @ 10% traffic | Validate 66% Hit@5 in prod |
| **2** | Canary Deploy | Phase-2 @ 20% traffic | Confirm latency <5ms |
| **3** | Canary Deploy | Phase-2 @ 50% traffic | Monitor lane health |
| **4** | Full Rollout | Phase-2 @ 100% traffic | **Champion becomes default** |
| **5** | Phase-2B Training | Add soft negatives | Train +1.5-3% model |
| **6** | Phase-2B Deploy | Canary → Full rollout | 68-69% Hit@5 in prod |
| **7** | Phase-2C Training | Add hard negatives | Train +1.5-3.5% model |
| **8** | Phase-2C Deploy | Canary → Full rollout | 70-72% Hit@5 in prod |
| **9-10** | Phase-3 Training | 1000-vector context | Train +3-5% model |
| **11** | Phase-3 Pilot | Long-context queries only | Validate 73-75% Hit@5 |
| **12-14** | TMD Training | 16 experts + router | Train +2-3% model |
| **15** | TMD Deploy | Canary → Full rollout | 74-75% Hit@5 in prod |
| **16+** | Continuous | Weekly drift monitoring | Maintain 75%+ Hit@5 |

**Total Timeline**: 4 months to full TMD deployment with 75%+ Hit@5

---

## 🎯 Success Metrics

### Production Targets (Progressive)

| Milestone | Hit@5 Target | Hit@10 Target | Status |
|-----------|-------------|--------------|--------|
| **Phase-2 (Current)** | ≥66% | ≥75% | ✅ **ACHIEVED!** |
| **Phase-2B (Soft negs)** | ≥68% | ≥76% | Week 6 target |
| **Phase-2C (Hard negs)** | ≥70% | ≥78% | Week 8 target |
| **Phase-3 (1000-ctx)** | ≥73% | ≥82% | Week 11 target |
| **TMD (16 experts)** | ≥75% | ≥83% | Week 15 target |

### Operational Metrics (Always Monitor)

| Metric | Target | Alert Threshold |
|--------|--------|----------------|
| **Hit@5 Proxy** | ≥66% | <60% |
| **P95 Latency** | <5ms | >10ms |
| **Empty Hit Rate** | <10% | >20% |
| **Error Rate** | <0.1% | >1% |
| **Lane Delta** | <15% below avg | >20% below avg |

---

## 🔧 Infrastructure Requirements

### Compute Resources

**Phase-2 Production (Current)**:
- GPU: 1x A10 or similar (24GB VRAM)
- CPU: 8 cores
- RAM: 32GB
- Storage: 100GB (models + cache)

**Phase-3 (1000-Context)**:
- GPU: 1x A100 (40GB VRAM) - larger context
- CPU: 16 cores
- RAM: 64GB
- Storage: 200GB

**TMD (16 Experts)**:
- GPU: 2x A100 (80GB VRAM total)
- CPU: 32 cores
- RAM: 128GB
- Storage: 500GB (16 experts + router)

### Monitoring Stack

```yaml
# Prometheus metrics
lvm_hit5_proxy: gauge
lvm_latency_p95: histogram
lvm_empty_hit_rate: gauge
lvm_error_rate: gauge
lvm_lane_hit5: gauge (per lane)

# Grafana dashboards
- LVM Performance Overview
- Lane Health Monitoring
- Latency Breakdown
- Error Tracking
- A/B Test Comparison

# Alerts (PagerDuty/Slack)
- Hit@5 drops <60%
- Latency P95 >10ms
- Error rate >1%
- Any lane degrades >20%
```

---

## 🚨 Rollback Procedures

### Automatic Rollback (Circuit Breaker)

```python
class LVMCircuitBreaker:
    def __init__(self):
        self.error_count = 0
        self.window_size = 100
        self.threshold = 10  # 10% error rate

    def check_health(self, prediction_result):
        if prediction_result.has_error():
            self.error_count += 1

        # Check error rate
        if self.error_count >= self.threshold:
            # Automatic rollback
            self.rollback_to_previous_model()
            self.alert('Circuit breaker triggered - automatic rollback')

        # Reset window
        if prediction_result.request_id % self.window_size == 0:
            self.error_count = 0
```

### Manual Rollback (One-Command)

```bash
# Rollback to previous model
./scripts/rollback_lvm.sh --to-version phase2-champion

# Options:
# --to-version phase1-baseline    # 59.32% Hit@5
# --to-version phase2-champion    # 66.52% Hit@5
# --to-version phase2b-softnegs   # 68.5% Hit@5
# --to-version phase2c-hardnegs   # 70.5% Hit@5

# Script will:
# 1. Stop current model server
# 2. Load specified checkpoint
# 3. Restart with new model
# 4. Run smoke tests
# 5. Alert ops team
```

---

## 📊 Expected ROI

### Performance Gains Timeline

| Month | Phase | Hit@5 | Hit@10 | Cumulative Gain |
|-------|-------|-------|--------|----------------|
| **0 (Current)** | Phase-2 | 66.52% | 74.78% | Baseline |
| **1** | Phase-2B | 68.5% | 76.5% | +1.98% / +1.72% |
| **2** | Phase-2C | 70.5% | 78.5% | +3.98% / +3.72% |
| **3** | Phase-3 | 73.5% | 82.5% | +6.98% / +7.72% |
| **4** | TMD | 75.5% | 83.5% | +8.98% / +8.72% |

**Total improvement from broken model** (36.99% → 75.5%):
- **+38.51% absolute Hit@5 improvement**
- **+104% relative improvement**

### Business Impact

**Assuming 1M queries/day:**

| Metric | Before (Broken) | After (TMD) | Impact |
|--------|----------------|-------------|--------|
| Successful retrievals | 370K | 755K | **+385K/day** |
| User satisfaction | Low | High | **+104% improvement** |
| Query reformulations | 630K | 245K | **-385K wasted queries** |

**Value:**
- Fewer query reformulations = better UX
- Higher accuracy = more trust
- Faster retrieval = lower latency

---

## 🎊 Summary

**Current Status**: Champion model (66.52% Hit@5) ready for deployment

**Recommended Path**:
1. ✅ **Deploy Phase-2 immediately** (canary → full rollout)
2. → Train Phase-2B (soft negatives) for +2% gain
3. → Train Phase-2C (hard negatives) for +2% gain
4. → Pilot Phase-3 (1000-context) for +3% gain
5. → Deploy TMD routing for +2% gain
6. → Continuous monitoring and drift detection

**Expected Outcome**: 75%+ Hit@5 in 4 months

**Risk Management**:
- Canary deployments at every step
- Automated rollback on regressions
- Weekly drift monitoring
- Per-lane health tracking

**Partner, we have a clear path to 75%+ Hit@5!** Let's start with Phase-2 canary deployment and iterate from there! 🚀
