# PLMS Operational Guardrails

Production operations guide for PLMS Tier 1 deployment.

**Version**: V1 Tier 1
**Date**: 2025-11-06
**Status**: Production Ready

---

## 🚨 Critical Ops Guardrails

### 1. Alerting Configuration

**Budget Runway Alerts**

| Alert Level | Threshold | Action | Response Time |
|-------------|-----------|--------|---------------|
| ⚠️ Warning  | T-15 minutes to depletion | Slack notification | 15 min |
| 🔴 Critical | T-5 minutes to depletion | Page on-call engineer | 5 min |
| 🟣 Projected Overrun | >20% over budget | Email project owner + PMO | 1 hour |

**Alert Destinations**:
- Warning: `#plms-alerts` Slack channel
- Critical: PagerDuty incident (escalate to on-call)
- Overrun: Email to `project-owner@company.com` + `pmo@company.com`

**Implementation**:
```python
# services/plms/alerting.py
def check_budget_runway(project_id: int):
    runway = get_budget_runway(project_id)
    minutes_left = runway["minutes_to_depletion"]

    if minutes_left <= 5:
        pagerduty.trigger("plms-budget-critical", severity="critical", ...)
    elif minutes_left <= 15:
        slack.send("#plms-alerts", f"⚠️ Project {project_id} budget T-{minutes_left}m")

    if runway["projected_overrun_usd"] > runway["budget_usd"] * 0.20:
        email.send([project_owner, pmo], subject="Budget overrun projected", ...)
```

**Monitoring Frequency**:
- Budget runway: Every 60 seconds during active execution
- Idle projects: Every 10 minutes

---

### 2. Scheduler Lane Caps

**Initial Configuration** (tune after 1 week):

| Lane ID | Lane Name | Concurrency Cap | Rationale |
|---------|-----------|-----------------|-----------|
| 4200 | Code-API | 4 | Lightweight, can parallelize |
| 4201 | Code-Test | 4 | Fast execution, CPU-bound |
| 4202 | Code-Docs | 3 | Medium weight |
| 5100 | Data-Schema | 2 | Expensive, heavy DB ops |
| 5101 | Data-Ingest | 3 | I/O-bound, moderate parallelism |
| 5102 | Data-Transform | 2 | Memory-intensive |
| 6100 | Model-Train | 2 | GPU-bound, high memory |
| 6101 | Model-Eval | 3 | Inference-only, lighter than training |

**Location**: `services/plms/portfolio_scheduler.py:LANE_CONCURRENCY_LIMITS`

**Tuning Process**:
1. **Week 1**: Start with conservative caps (above values)
2. **Week 2**: Monitor utilization via `/api/portfolio/status`
3. **Adjust**: Increase caps for lanes with:
   - Consistent 100% utilization
   - Low error rates
   - Short task durations
4. **Decrease caps** for lanes with:
   - High error rates (>5%)
   - Resource contention (OOM, timeouts)
   - Slow queue processing

**Monitoring Query** (daily review):
```sql
SELECT lane_id,
       AVG(utilization) as avg_util,
       MAX(queue_length) as max_queue,
       AVG(task_duration_ms) as avg_duration
FROM portfolio_metrics
WHERE date >= NOW() - INTERVAL '7 days'
GROUP BY lane_id
ORDER BY avg_util DESC;
```

**Alerting**:
- If any lane shows >95% utilization for >2 hours: Slack warning
- If queue length >20 for >30 minutes: Consider increasing cap

---

### 3. Calibration Data Policy

**Inclusion Criteria** (strictly enforced):

✅ **Include** in calibration dataset:
- `run_kind IN ('baseline', 'hotfix')`
- `validation_pass = TRUE`
- `write_sandbox = FALSE`

❌ **Exclude** from calibration dataset:
- `run_kind = 'rehearsal'` (not production-representative)
- `run_kind = 'replay'` (deterministic, not novel data)
- `validation_pass = FALSE` (failed KPI gates)
- `write_sandbox = TRUE` (sandbox runs, not real execution)

**Implementation**:
```python
# services/plms/calibration.py
def should_include_run_for_calibration(run_kind: str, validation_pass: bool, write_sandbox: bool) -> bool:
    """
    Determine if a run should be included in Bayesian calibration dataset.

    Args:
        run_kind: 'baseline' | 'rehearsal' | 'replay' | 'hotfix'
        validation_pass: Did all KPIs pass?
        write_sandbox: Was this a sandbox run?

    Returns:
        True if run should be included in calibration
    """
    # Only production-representative runs
    if run_kind not in {'baseline', 'hotfix'}:
        return False

    # Only successful validation
    if not validation_pass:
        return False

    # Only real executions (no sandbox)
    if write_sandbox:
        return False

    return True
```

**Validation** (nightly invariant check):
```sql
-- Alert if any rehearsal/replay runs are in calibration dataset
SELECT COUNT(*) as violation_count
FROM estimate_versions
WHERE run_kind NOT IN ('baseline', 'hotfix')
   OR validation_pass = FALSE
   OR write_sandbox = TRUE;
```

Expected: `violation_count = 0` (alert if >0)

---

### 4. Idempotency Configuration

**Redis Setup** (production):
```bash
# Use Redis cluster for high availability
REDIS_URL=redis://redis-cluster.company.com:6379/0

# TTL: 24 hours (balance between safety and cache bloat)
IDEMPOTENCY_TTL_SECONDS=86400

# Enable in production (disable only for local dev)
PLMS_USE_REDIS=true
```

**Monitoring**:
```bash
# Check cache hit rate (should be 5-15% for normal retries)
redis-cli INFO stats | grep keyspace_hits

# Check memory usage
redis-cli INFO memory | grep used_memory_human
```

**Alerting**:
- Redis unavailable: Critical page (fallback to in-memory, NOT production-safe)
- Memory usage >80%: Adjust TTL or scale Redis
- Hit rate >50%: Investigate potential infinite retry loops

---

### 5. KPI Validation Gates

**Quality SLO Gates** (per lane):

| Lane | KPI | Threshold | Operator | Blocks Completion? |
|------|-----|-----------|----------|--------------------|
| Code-API | test_pass_rate | 0.90 | >= | Yes |
| Code-API | linter_pass | true | == | Yes |
| Code-Test | test_pass_rate | 0.95 | >= | Yes |
| Code-Docs | readability | 10.0 | <= | Yes |
| Data-Schema | schema_diff | 0 | == | Yes |
| Data-Schema | row_count_delta | 0.05 | <= | Yes |
| Data-Ingest | row_count_delta | 0.10 | <= | Yes |

**Echo Baseline** (all lanes):
- Threshold: 0.82 cosine similarity
- Operator: >=
- Blocks completion: No (advisory only for Tier 1)

**Enforcement**:
```python
# services/plms/kpi_emit.py
# Exit code 1 if any KPI fails
all_pass = (passed_kpis == total_kpis) and echo_pass
sys.exit(0 if all_pass else 1)
```

**Override Process** (emergency only):
1. Project owner submits override request
2. Engineering manager reviews KPI logs
3. If justified (e.g., test infra issue): Manual override flag in database
4. All overrides logged for audit

```sql
-- Record override (emergency only)
UPDATE project_runs
SET validation_pass_override = TRUE,
    override_reason = 'Test infra outage - manual validation passed',
    overridden_by = 'eng-manager@company.com',
    overridden_at = NOW()
WHERE run_id = 'abc-123';
```

---

### 6. Provider Snapshot Tracking

**Deterministic Replay Requirements**:

Every run MUST capture:
- Provider matrix (model versions)
- Capability snapshot (available features)
- PRD SHA256 hash
- Environment fingerprint (git commit, Python version, OS)

**Validation** (pre-flight check):
```python
# Before PAS submission
def validate_replay_passport(run_id: str) -> bool:
    passport = load_passport(run_id)

    required_fields = [
        "provider_matrix", "capabilities", "prd_sha",
        "env.git_commit", "env.python_version"
    ]

    missing = [f for f in required_fields if not get_nested(passport, f)]

    if missing:
        raise ValueError(f"Incomplete passport: missing {missing}")

    return True
```

**Alerting**:
- If passport file missing: Block run submission
- If snapshot incomplete: Critical alert

---

### 7. Nightly Invariant Checks

**Automated Checks** (cron at 02:00 ET):

```bash
# Cron entry (Ubuntu)
0 2 * * * /usr/bin/env DB_PATH=/srv/registry.db /srv/app/scripts/check_plms_invariants.py --db $DB_PATH >> /var/log/plms_invariants.log 2>&1
```

**Checks Performed** (`scripts/check_plms_invariants.py`):

1. ✅ All runs have replay passports
2. ✅ KPI formulas are valid JSON
3. ✅ Calibration dataset excludes rehearsal/replay/sandbox/failed runs
4. ✅ Lane overrides have valid lane_id references
5. ✅ Provider snapshots match actual router state
6. ✅ Budget runway calculations are consistent

**Exit Behavior**:
- Exit 0: All checks passed
- Exit 1: Violations detected → Page on-call + email report

**Example Output**:
```
=== PLMS Invariants Check (2025-11-06 02:00:15) ===
✓ Replay passports: 1,287/1,287 complete
✓ KPI formulas: 542/542 valid
✗ Calibration dataset: 3 rehearsal runs included (VIOLATION!)
✓ Lane overrides: 42/42 valid
✓ Provider snapshots: 1,287/1,287 consistent
✗ Budget runway: 2 projects with negative runway (VIOLATION!)

=== VIOLATIONS DETECTED ===
Exit code: 1
Paging on-call engineer...
```

---

### 8. Performance SLOs

**API Latency Targets**:

| Endpoint | P50 | P95 | P99 | Action if Exceeded |
|----------|-----|-----|-----|--------------------|
| POST /start | 50ms | 150ms | 300ms | Optimize DB writes |
| POST /simulate | 200ms | 500ms | 1s | Scale PAS workers |
| GET /metrics | 20ms | 50ms | 100ms | Add DB indices |
| GET /portfolio/status | 10ms | 30ms | 50ms | Cache scheduler state |

**Throughput Targets**:
- Concurrent projects: 10+ without degradation
- Tasks per minute: 100+ (per lane cap)

**Monitoring**:
```python
# Prometheus metrics
from prometheus_client import Histogram

api_latency = Histogram('plms_api_latency_seconds',
                        'API endpoint latency',
                        ['endpoint', 'method'])

@router.post("/start")
@api_latency.labels(endpoint='/start', method='POST').time()
def start_project(...):
    ...
```

**Alerting**:
- P95 latency >2x target for >5 minutes: Warning
- P95 latency >3x target for >2 minutes: Critical

---

## 🔧 Tuning After Week 1

**Data Collection** (days 1-7):
- Monitor lane utilization via `/api/portfolio/status`
- Track KPI pass rates per lane
- Measure calibration error (MAE) per lane
- Log alert frequency and false positive rate

**Adjustment Process** (day 8):

1. **Lane Caps**: Increase for lanes with >90% utilization, decrease for >10% error rates
2. **KPI Thresholds**: Relax if false positive rate >5%, tighten if quality issues reported
3. **Alerting**: Adjust T-15/T-5 thresholds based on response time distributions
4. **Calibration α**: Tune Bayesian learning rate if MAE not converging

**Example Adjustment**:
```python
# Before (Week 1)
LANE_CONCURRENCY_LIMITS[5100] = 2  # Data-Schema

# After (Week 2)
# Observed: 100% utilization, 0.2% error rate, 30s avg task duration
# Decision: Increase cap to 3
LANE_CONCURRENCY_LIMITS[5100] = 3
```

---

## 📊 Dashboards & Observability

**Grafana Panels** (recommended):

1. **Budget Runway Gauge** (real-time)
   - Current budget, spent, burn rate
   - Minutes to depletion
   - Projected overrun

2. **Lane Utilization Heatmap** (1-minute granularity)
   - Rows: Lane IDs
   - Columns: Time buckets
   - Color: % utilization (0-100%)

3. **KPI Pass Rate** (per lane, daily)
   - Stacked bar chart: Pass/Fail counts
   - Trend line for false positive detection

4. **Calibration Drift** (weekly)
   - MAE sparklines per lane
   - Convergence indicator (decreasing MAE = good)

5. **Idempotency Cache Stats** (hourly)
   - Hit rate
   - Memory usage
   - Eviction rate

**Data Sources**:
- Prometheus: Latency, throughput, cache stats
- PostgreSQL: KPI receipts, lane metrics, budget data
- Redis: Cache hit/miss counts

---

## 🚧 Known Limitations (Tier 1)

1. **No cross-project prioritization**: Single project monopolizes lanes (TODO: Tier 2 fair-share)
2. **No preemption**: Long-running tasks block queue (TODO: Tier 2 task preemption)
3. **No dynamic lane caps**: Manual tuning required (TODO: Tier 3 autoscaling)
4. **Echo advisory only**: Not a hard gate (TODO: Tier 2 echo enforcement)

---

## 📞 Escalation Path

| Issue | Severity | Response Time | Contact |
|-------|----------|---------------|---------|
| Budget critical (T-5) | P0 | 5 min | On-call engineer (PagerDuty) |
| Redis unavailable | P0 | 5 min | On-call engineer |
| KPI gate blocking prod deploy | P1 | 15 min | Engineering manager |
| Calibration drift >30% | P2 | 1 hour | MLOps team |
| Lane cap adjustment request | P3 | 1 day | Platform team |

**On-Call Rotation**: `ops/oncall_schedule.md`

---

## 🔒 Security Notes

1. **Idempotency keys**: UUIDs only (prevent key guessing)
2. **RBAC scopes**: Enforce least-privilege (projects.view, projects.start, projects.simulate)
3. **Audit logging**: All `/start` calls logged with user_id
4. **Sensitive data**: Provider credentials stored in secrets vault (not in passports)

---

## ✅ Pre-Production Checklist

- [ ] Redis cluster running with 3+ nodes
- [ ] PagerDuty integration configured
- [ ] Slack webhook set up (`#plms-alerts`)
- [ ] Grafana dashboards deployed
- [ ] Cron job scheduled (`check_plms_invariants.py`)
- [ ] Lane caps tuned for initial load
- [ ] KPI thresholds validated with test runs
- [ ] On-call schedule published
- [ ] Rollback migration tested on staging
- [ ] All 3 demos passing (idempotency, canary, KPI-gated)

---

**Last Updated**: 2025-11-06
**Next Review**: 2025-11-13 (post-Week-1 tuning)
