# Go-Live Checklist — Release v0 (Quick Reference)

**Date:** October 28, 2025
**Owner:** Retrieval Platform
**Target:** Ship GTR-T5 + FAISS IVF-Flat + AMN/GRU + Vector Reranker

---

## ✅ Pre-Flight (T-24h)

### Artifacts Freeze
- [ ] `artifacts/releases/retriever_v0/` contains all required files
- [ ] Git tag: `v0-retriever` created
- [ ] Checksums validated: `sha256sum artifacts/releases/retriever_v0/*.faiss`

### Smoke Tests (MUST PASS)
```bash
# Test 1: IVF vs FLAT agreement (≥95% @ K=10)
python tools/ivf_vs_flat_check.py --n 100 --k 10 \
  --flat artifacts/releases/retriever_v0/p_flat_ip.faiss \
  --ivf artifacts/releases/retriever_v0/p_ivf.faiss

# Test 2: Reranker lift (+3-5pp expected)
python tools/compare_rerank.py \
  --hits artifacts/eval/hits50_baseline.jsonl \
  --reranked artifacts/eval/hits50_reranked.jsonl
```

### Config Pins
- [ ] **Embeddings:** `GTR-T5-768D` (L2-norm enforced)
- [ ] **FAISS:** IVF-Flat, `nprobe=8` (start)
- [ ] **Reranker:** On (MLP 2-layer)
- [ ] **Flags:** `RETRIEVER_IMPL=amn_v0`, `RERANKER=on`

### Observability
- [ ] Health endpoint: `/health` working
- [ ] Latency histograms: P50/P95/P99 configured
- [ ] Error budget: P95 ≤ 8ms retrieval + 2ms rerank
- [ ] Alerts: Configured (see runbook)

### Rollback Ready
- [ ] Previous version backed up: `artifacts/releases/v-1/`
- [ ] Rollback script tested: `scripts/rollback_retriever.sh`
- [ ] Kill-switch: Single env var flip ready

---

## 🚀 Rollout Timeline

### T0: Canary (5%)
```bash
export RETRIEVER_IMPL=amn_v0 RERANKER=on CANARY_PERCENT=5
kubectl apply -f k8s/retriever-v0-canary.yaml
```
**Watch:** QPS, error rate ≤ baseline+0.2%, P95 ≤ 10ms
**Decision (T+30min):** Green → proceed | Spike → tune nprobe | Errors → **ROLLBACK**

### T+2h: Ramp 25%
```bash
export CANARY_PERCENT=25
kubectl apply -f k8s/retriever-v0-canary.yaml
```
**Watch:** Quality proxy, reranker acceptance (20-40%)
**Decision (T+3h):** Stable → proceed to 50% | Latency creep → nprobe sweep

### T+6h: Ramp 50%
```bash
export CANARY_PERCENT=50
python tools/run_clean_eval.py --out artifacts/eval/nightly_$(date +%Y%m%d).json
```
**Watch:** Nightly eval (R@5, MRR, Contain@50)
**Decision (T+8h):** ∆R@5 < ±2pp → plan 100% | Drop 2-5pp → tune | Drop >5pp → **HOLD**

### T+24h: Full (100%)
```bash
export CANARY_PERCENT=100
kubectl apply -f k8s/retriever-v0-production.yaml
```
**Validation (T+48h):**
```bash
python tools/run_clean_eval.py --out artifacts/eval/post_rollout_v0.json
python tools/compare_metrics.py --before artifacts/eval/metrics.json --after artifacts/eval/post_rollout_v0.json
```

---

## 🔥 Emergency Rollback (1 Command)

**Triggers:** Error rate >1% OR P95 >20ms OR R@5 drop >10pp (10+ min)

```bash
# Method 1: Feature flag flip (instant)
export RETRIEVER_IMPL=gru_v0 RERANKER=off

# Method 2: Full rollback
bash scripts/rollback_retriever.sh v-1

# Method 3: Kubernetes
kubectl rollout undo deployment/retriever-service
```

**Validate:**
```bash
curl http://localhost:8080/health | jq '.version'
python tools/check_rollback.py --baseline artifacts/eval/baseline_metrics.json
```

---

## ⚙️ Nprobe Tuning (10-Minute Pass)

**When:** P95 >8ms OR R@10 drops >3pp

```bash
python tools/nprobe_sweep.py \
  --index artifacts/releases/retriever_v0/p_ivf.faiss \
  --flat artifacts/releases/retriever_v0/p_flat_ip.faiss \
  --nprobe-values 4,8,12,16 \
  --n-queries 500 \
  --out artifacts/tuning/nprobe_sweep.json
```

**Results:**
- `nprobe=4`: P95=3.2ms, ΔR@10=-1.2pp (fast, slight loss)
- `nprobe=8`: P95=4.8ms, ΔR@10=-0.3pp (balanced, default)
- `nprobe=12`: P95=7.1ms, ΔR@10=-0.1pp (accurate, slower)

**Pick:** Smallest nprobe with ΔR@10 ≥ -0.5pp AND P95 ≤ target

---

## 📊 Key Metrics Dashboard

### Retrieval Performance
- **Latency:** P50/P95/P99 (Encode, Search, Rerank)
- **Throughput:** QPS by endpoint
- **Quality Proxy:** % queries with result ≥ cosine threshold

### Ranking Quality
- **Reranker Acceptance:** % queries where rank-1 changes (expect 20-40%)
- **∆MRR (sampled):** 1% traffic comparison
- **Diversity:** Mean cosine between top-5 (detect duplication)

### Index Health
- **IVF Imbalance:** Cluster size histogram (detect hotspots)
- **Corpus Growth:** Vectors added/removed daily
- **Memory:** RAM per worker (detect leaks)

### Quality (Nightly)
- **R@5, MRR, Contain@50:** Pre + post rerank
- **Trend:** 7-day rolling average

---

## 🚨 Alerts

**Critical (Page):**
- P95 >20ms for 10 min
- Error rate >1% for 10 min
- R@5 drop >10pp (nightly)

**Warning (Slack):**
- P95 10-20ms for 30 min
- R@5 drop 5-10pp
- Contain@50 <80%

**Info (Log):**
- Reranker acceptance <10% (not helping)
- Memory >20% above baseline (leak?)

---

## 📅 Weekly Maintenance

```bash
# Rebuild FLAT truth
python tools/rebuild_flat_truth.py --corpus artifacts/corpus/current.npz \
  --out artifacts/faiss/p_flat_ip_$(date +%Y%m%d).faiss

# Re-eval
python tools/run_clean_eval.py \
  --index artifacts/faiss/p_flat_ip_$(date +%Y%m%d).faiss \
  --out artifacts/eval/weekly_$(date +%Y%m%d).json

# Append history
cat artifacts/eval/weekly_$(date +%Y%m%d).json >> artifacts/eval/history.jsonl
```

---

## 🛡️ Risk Mitigations

| Risk | Symptom | Mitigation |
|------|---------|------------|
| Containment dips | Contain@50: 82% → 75% | Bump same-article filter, increase nprobe, per-lane sharding |
| Reranker regression | ∆R@5: +5pp → +1pp | Flag off reranker, retrain on new corpus, A/B test |
| L2-norm forgotten | Nonsensical results | Automated preflight asserts, unit tests, monitor mean norm |
| IVF imbalance | P95 latency spikes | Retrain IVF (k-means++), monitor cluster sizes, consider OPQ/PQ |

---

## 📋 Post-Ship Backlog

**Week 1:** A/B test reranker (10% off, 90% on, measure lift)
**Week 2:** Latency optimization (nprobe=4, batch encoding, OPQ/PQ)
**Week 3:** Explainability (log top-K cosines, margins, article-ctx stats)
**Future:** Two-tower revisit (only if Contain@50 <75%, different backbone)

---

## 📞 Contact & Escalation

**Owner:** Retrieval Platform Team
**On-Call:** [Slack #retrieval-oncall]

**Escalation:**
1. Check runbook: `docs/PROD/Rollout_48hr_Runbook.md`
2. Run diagnostics: `python tools/diagnose_retriever.py`
3. If critical (error rate >1%), **rollback immediately**
4. Post-mortem: `docs/PROD/postmortems/YYYYMMDD_incident.md`

---

## 🎯 Ship Gates (Final Check)

- [ ] **R@5 ≥ 0.30** OR **MRR ≥ 0.20** (clean eval)
- [ ] **Contain@50 ≥ 0.82** (preferred)
- [ ] **P95 ≤ 8ms** (retrieval) + **≤2ms** (rerank)
- [ ] All smoke tests pass
- [ ] Rollback tested and ready
- [ ] Team on-call and briefed

---

**Document Version:** 1.0
**Last Updated:** October 28, 2025
**Full Runbook:** `docs/PROD/Rollout_48hr_Runbook.md`
