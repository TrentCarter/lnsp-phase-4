# 48-Hour Rollout Runbook — Release v0

**Date:** October 28, 2025
**Owner:** Retrieval Platform
**Stack:** GTR-T5 768D + FAISS IVF-Flat + AMN/GRU + Vector Reranker

---

## Pre-Flight Checklist (T-24h)

### 1. Artifacts Freeze
- [ ] `artifacts/releases/retriever_v0/` contains:
  - `p_flat_ip.faiss` (truth index)
  - `p_ivf.faiss` (serving index, nprobe=8)
  - `metrics.json` (pre-ship evaluation)
  - `VERSION` (release metadata)
  - `AMN_v0.md`, `GRU_v0.md` (model cards)
- [ ] Git tag created: `v0-retriever`
- [ ] Checksum validation: `sha256sum artifacts/releases/retriever_v0/*.faiss > checksums.txt`

### 2. Smoke Tests (MUST PASS)
```bash
# Test 1: IVF vs FLAT agreement (≥95% overlap @ K=10)
python tools/ivf_vs_flat_check.py --n 100 --k 10 \
  --flat artifacts/releases/retriever_v0/p_flat_ip.faiss \
  --ivf artifacts/releases/retriever_v0/p_ivf.faiss

# Test 2: L2-norm invariants (all queries and passages normalized)
python tools/check_l2_norms.py --index artifacts/releases/retriever_v0/p_ivf.faiss

# Test 3: Reranker lift (if enabled, expect +3-5pp on R@5)
python tools/compare_rerank.py \
  --hits artifacts/eval/hits50_baseline.jsonl \
  --reranked artifacts/eval/hits50_reranked.jsonl
```

### 3. Config Pins (Production)
- [ ] **Embeddings:** `GTR-T5-768D` (L2-norm enforced at encode time)
- [ ] **FAISS serving:** IVF-Flat, `nprobe=8` (start conservative)
- [ ] **Reranker:** MLP 2-layer, features = `[cosine, margin, local-ctx, diversity]`
- [ ] **Feature flags:**
  - `RETRIEVER_IMPL=amn_v0` (fallback: `gru_v0`)
  - `RERANKER=on` (or `off` for A/B test)

### 4. Observability Setup
- [ ] Heartbeat JSON endpoint: `/health` (ingest + search + rerank)
- [ ] Latency histograms: P50, P95, P99 for:
  - Encode (Q)
  - Search (Q → Top50)
  - Rerank (Top50 → reordered)
- [ ] Error budget dashboard: P95 ≤ 8ms retrieval, +2ms rerank
- [ ] Alerts configured (see Monitoring section below)

### 5. Rollback Plan
- [ ] Previous version artifacts backed up: `artifacts/releases/v-1/`
- [ ] Rollback script tested: `scripts/rollback_retriever.sh`
- [ ] Canary kill-switch ready: single env var flip

---

## Rollout Timeline (T0 = Go-Live)

### T0: Canary Launch (5% Traffic)

**Actions:**
```bash
# Enable AMN_v0 + reranker for 5% traffic
export RETRIEVER_IMPL=amn_v0
export RERANKER=on
export CANARY_PERCENT=5

# Start canary deployment
kubectl apply -f k8s/retriever-v0-canary.yaml
# OR: update feature flag in control plane
```

**Watch:**
- **QPS:** Should match baseline 5% split
- **Error rate:** ≤ baseline + 0.2%
- **P95 latency:** ≤ 10ms end-to-end (8ms retrieval + 2ms rerank)
- **Cache hit ratio:** Expect drop (cold start), stabilizes in 15 min
- **Top-50 duplication:** <3% near-dups with cosine >0.98

**Decision Point (T+30min):**
- ✅ All metrics green → proceed
- ⚠️ P95 latency spike → tune `nprobe` down (8 → 4) OR disable reranker
- ❌ Error rate >0.5% → **ROLLBACK** (see Emergency Rollback section)

---

### T+2h: Ramp to 25%

**Actions:**
```bash
export CANARY_PERCENT=25
kubectl apply -f k8s/retriever-v0-canary.yaml
```

**Watch:**
- Same metrics as T0, plus:
- **Retrieval quality proxy:** % queries with ≥1 result above `cosine ≥ 0.65` (calibrate threshold from nightly eval)
- **Reranker acceptance rate:** How often rank-1 changes (expect 20-40%)

**Decision Point (T+3h):**
- ✅ Error rate ≤ baseline + 0.2%, P95 ≤ baseline + 15% → proceed to 50%
- ⚠️ Latency creep → run nprobe sweep (see Nprobe Tuning below)
- ❌ Quality degradation (>5pp drop in proxy metric) → hold at 25%, investigate

---

### T+6h: Ramp to 50%

**Actions:**
```bash
export CANARY_PERCENT=50
kubectl apply -f k8s/retriever-v0-canary.yaml

# Trigger nightly eval snapshot (even if midday)
python tools/run_clean_eval.py --out artifacts/eval/nightly_$(date +%Y%m%d).json
```

**Watch:**
- **Nightly eval metrics:** R@5, MRR, Contain@50 (compare to pre-ship baseline)
- **Index health:** IVF list imbalance (log histogram of cluster sizes)
- **Memory footprint:** RAM usage per worker (expect stable)

**Decision Point (T+8h):**
- ✅ Eval metrics stable (∆R@5 < ±2pp) → plan 100% rollout
- ⚠️ R@5 drop 2-5pp → enable reranker if off, or tune nprobe up (8 → 12)
- ❌ R@5 drop >5pp → **HOLD at 50%**, escalate to team

---

### T+24h: Full Rollout (100%)

**Actions:**
```bash
export CANARY_PERCENT=100
kubectl apply -f k8s/retriever-v0-production.yaml

# Archive canary config
mv k8s/retriever-v0-canary.yaml k8s/archive/retriever-v0-canary-$(date +%Y%m%d).yaml
```

**Watch (continuous for 24h):**
- All metrics from previous stages
- **Data drift:** Weekly rebuild of FLAT truth + re-eval (append to `artifacts/eval/history.jsonl`)
- **Corpus growth:** Track `index.ntotal` daily (adjust nlist if >10% growth)

**Validation (T+48h):**
```bash
# Run comprehensive post-rollout eval
python tools/run_clean_eval.py --out artifacts/eval/post_rollout_v0.json

# Compare to pre-ship baseline
python tools/compare_metrics.py \
  --before artifacts/eval/metrics.json \
  --after artifacts/eval/post_rollout_v0.json
```

---

## Emergency Rollback (1 Command)

**Trigger conditions:**
- Error rate >1% for >10 minutes
- P95 latency >20ms for >10 minutes
- R@5 drop >10pp (validated on nightly eval)

**Rollback command:**
```bash
# Method 1: Feature flag flip (instant, no redeploy)
export RETRIEVER_IMPL=gru_v0  # Fallback to previous model
export RERANKER=off           # Disable reranker

# Method 2: Revert to previous release (full rollback)
bash scripts/rollback_retriever.sh v-1

# Method 3: Kubernetes rollback
kubectl rollout undo deployment/retriever-service
```

**Validation:**
```bash
# Check rollback succeeded
curl http://localhost:8080/health | jq '.version'
# Expected: "v-1" or previous version string

# Verify metrics return to baseline
python tools/check_rollback.py --baseline artifacts/eval/baseline_metrics.json
```

---

## Nprobe Tuning (10-Minute Pass)

**When to tune:**
- P95 latency exceeds 8ms consistently
- OR R@10 drops >3pp vs FLAT truth

**Procedure:**
```bash
# Run nprobe sweep on 500 queries
python tools/nprobe_sweep.py \
  --index artifacts/releases/retriever_v0/p_ivf.faiss \
  --queries artifacts/eval/eval_queries.npy \
  --flat artifacts/releases/retriever_v0/p_flat_ip.faiss \
  --nprobe-values 4,8,12,16 \
  --n-queries 500 \
  --out artifacts/tuning/nprobe_sweep_$(date +%Y%m%d).json

# Results will show:
# nprobe=4:  P95=3.2ms, ΔR@10=-1.2pp (fast, slight accuracy loss)
# nprobe=8:  P95=4.8ms, ΔR@10=-0.3pp (balanced, current setting)
# nprobe=12: P95=7.1ms, ΔR@10=-0.1pp (accurate, slower)
# nprobe=16: P95=10.5ms, ΔR@10=0.0pp (exact, too slow)

# Pick smallest nprobe with ΔR@10 ≥ -0.5pp AND P95 ≤ target
```

**Decision:**
- Latency-critical: Use `nprobe=4` (accept -1.2pp accuracy loss)
- Balanced: Use `nprobe=8` (current default)
- Accuracy-critical: Use `nprobe=12` (if latency budget allows)

**Apply:**
```bash
# Update serving config
export FAISS_NPROBE=8  # or 4, 12, 16
kubectl restart deployment/retriever-service

# Document decision
echo "nprobe=8 (P95=4.8ms, ΔR@10=-0.3pp)" >> docs/PROD/nprobe_tuning.md
```

---

## Monitoring & Alerts

### Key Metrics Dashboard

**Retrieval Performance:**
- **P50/P95/P99 latency:** Encode, Search, Rerank (separate histograms)
- **Throughput:** QPS by endpoint (search, encode)
- **Quality proxy:** % queries with ≥1 result above `cosine ≥ τ` (calibrate τ from eval)

**Ranking Quality:**
- **Reranker acceptance rate:** % queries where rank-1 changes
- **∆MRR (sampled):** Compare reranked vs baseline on 1% traffic sample
- **Diversity:** Mean cosine between top-5 results (detect duplication)

**Index Health:**
- **IVF list imbalance:** Histogram of cluster sizes (detect hotspots)
- **Add/remove delta:** Vectors added/removed per day (corpus growth)
- **Memory footprint:** RAM per worker (detect leaks)

**Quality (Nightly, Clean Split):**
- **R@5, MRR, Contain@50:** Pre + post rerank
- **Per-article stats:** Identify problematic article types
- **Trend plot:** 7-day rolling average

### Alert Configuration

**Critical (Page on-call):**
- P95 latency >20ms for 10 minutes → investigate load/hardware
- Error rate >1% for 10 minutes → potential data corruption or bug
- R@5 drop >10pp (nightly eval) → data drift or model degradation

**Warning (Slack notification):**
- P95 latency 10-20ms for 30 minutes → tune nprobe or scale up
- R@5 drop 5-10pp → enable reranker or retrain
- Contain@50 <80% → retrieval quality degradation, rebuild index

**Info (Log only):**
- Reranker acceptance rate <10% → reranker not helping, consider disabling
- Memory >20% above baseline → potential leak, restart workers

---

## Data Drift & Auto-Checks

### Weekly Maintenance
```bash
# Rebuild FLAT truth index
python tools/rebuild_flat_truth.py \
  --corpus artifacts/corpus/current.npz \
  --out artifacts/faiss/p_flat_ip_$(date +%Y%m%d).faiss

# Re-run evaluation
python tools/run_clean_eval.py \
  --index artifacts/faiss/p_flat_ip_$(date +%Y%m%d).faiss \
  --out artifacts/eval/weekly_$(date +%Y%m%d).json

# Append to history
cat artifacts/eval/weekly_$(date +%Y%m%d).json >> artifacts/eval/history.jsonl
```

### Automated Alerts
- **R@5 drop >3pp vs 7-day median:** Alert → investigate corpus changes
- **Contain@50 <0.80:** Alert → retrieval degradation, check index
- **P95 >12ms for 1 hour:** Alert → scale up or tune nprobe

---

## Risk Register & Mitigations

### Risk #1: Containment Dips with Corpus Growth
**Symptom:** Contain@50 drops from 82% → 75% over weeks
**Root Cause:** More articles → harder to find relevant chunks
**Mitigation:**
- Bump same-article filter weight in reranker
- Increase `nprobe` (8 → 12) to search more clusters
- Rebuild IVF with larger `nlist` (adjust √N as corpus grows)
- Consider per-lane sharding (separate index per article collection)

### Risk #2: Reranker Regressions
**Symptom:** ∆R@5 drops from +5pp → +1pp
**Root Cause:** Feature distribution drift (new article types)
**Mitigation:**
- Feature-flag off reranker immediately (1-line change)
- Retrain reranker on updated corpus (weekly cadence)
- A/B test reranker on/off (10% slice, compare lift)

### Risk #3: Cosine Mismatch (L2-Norm Forgotten)
**Symptom:** Top-K results are nonsensical (low cosine scores)
**Root Cause:** Forgot to L2-normalize Q or P before indexing/serving
**Mitigation:**
- Automated preflight assert: `assert abs(np.linalg.norm(vec) - 1.0) < 1e-5`
- Unit tests for encode pipeline (check norm invariants)
- Monitor: Log mean norm of encoded queries (should be 1.0 ± 1e-5)

### Risk #4: IVF Cluster Imbalance
**Symptom:** P95 latency spikes on certain queries
**Root Cause:** Some clusters have 10x more vectors (hotspots)
**Mitigation:**
- Retrain IVF with better initialization (k-means++ with multiple restarts)
- Monitor cluster size distribution (log histogram daily)
- Consider OPQ/PQ for large-N lanes (only if memory-constrained)

---

## Post-Ship Backlog

### Week 1: A/B Test Reranker
- **Goal:** Measure true reranker lift on prod traffic (not just eval)
- **Setup:** 10% traffic with reranker off, 90% on
- **Metrics:** Compare R@5, MRR, click-through rate (if available)

### Week 2: Latency Optimization
- **Goal:** Reduce P95 from 8ms → 5ms
- **Options:**
  - Try `nprobe=4` (if accuracy loss acceptable)
  - Batch encode queries (amortize overhead)
  - OPQ/PQ for memory-tight lanes (only if needed)

### Week 3: Explainability
- **Goal:** Help debug bad results
- **Output:** For each query, log:
  - Top-K cosines before rerank
  - Top-K margins (delta vs rank-1)
  - Article-context stats (how many results from same article)
  - Reranker feature values (for top-5)

### Future: Two-Tower Revisit
- **When:** Only if baseline stack hits limits (Contain@50 <75%)
- **Hypothesis:** Reframe to doc-level objective, broader article diversity
- **Approach:** Different backbone (non-Mamba), larger near-miss mining
- **Gate:** Must show >85% containment on article-disjoint eval before ship

---

## Contact & Escalation

**Owner:** Retrieval Platform Team
**On-Call:** [Slack #retrieval-oncall]
**Escalation Path:**
1. Check runbook (this doc)
2. Run diagnostics: `python tools/diagnose_retriever.py`
3. If critical (error rate >1%), rollback immediately
4. Post-mortem: `docs/PROD/postmortems/YYYYMMDD_incident.md`

---

**Document Version:** 1.0
**Last Updated:** October 28, 2025
**Next Review:** November 28, 2025 (monthly cadence)
