# PLMS Tier 1 Quick Start Guide

**Version**: V1 Tier 1 Production-Hardened
**Date**: 2025-11-06
**Status**: ✅ Demo Ready

---

## 🎯 What You Can Demo Today

Three crisp demos (<2 minutes each) that showcase production-hardened PLMS features:

### Demo 1: Idempotent Start
**Value**: Multi-pod safe, operator-proof retries

```bash
export PLMS_API_BASE_URL=http://localhost:6100
./tests/demos/demo1_idempotent_start.sh
```

**What it shows**:
- First request: `Idempotent-Replay: false`
- Second request (same key): `Idempotent-Replay: true`
- Same `run_id` returned both times
- HMI shows single run card

### Demo 2: Representative Canary
**Value**: Forecast and derisk before committing budget

```bash
export PLMS_API_BASE_URL=http://localhost:6100
./tests/demos/demo2_representative_canary.sh
```

**What it shows**:
- 1% rehearsal with guaranteed strata coverage (coverage = 1.0)
- Extrapolated estimates with 90% credible intervals
- Risk factors identified
- HMI overlays: risk heatmap + budget runway

### Demo 3: KPI-Gated Completion
**Value**: Real KPIs gate "Done", not just echo

```bash
./tests/demos/demo3_kpi_gated_completion.sh
```

**What it shows**:
- Poor readability → KPI fails → exit code 1
- HMI shows red banner with logs link
- After fix → KPI passes → exit code 0
- HMI shows green banner

---

## 🚀 Quick Setup (10 Minutes)

### 1. Start Redis
```bash
docker run -d -p 6379:6379 redis:alpine
export REDIS_URL=redis://localhost:6379/0
```

### 2. Apply Database Migrations
```bash
# SQLite (local dev)
sqlite3 artifacts/registry/registry.db < migrations/2025_11_06_plms_v1_tier1.sql
sqlite3 artifacts/registry/registry.db < migrations/2025_11_06_plms_tier1_patch2_kpi_receipts.sql

# PostgreSQL (production)
psql lnsp < migrations/2025_11_06_plms_v1_tier1.sql
psql lnsp < migrations/2025_11_06_plms_tier1_patch2_kpi_receipts.sql
```

### 3. Start PLMS API Server
```bash
export REDIS_URL=redis://localhost:6379/0
./.venv/bin/uvicorn services.plms.api.projects:router --port 6100 --reload
```

### 4. Verify Installation
```bash
# Check API health
curl http://localhost:6100/api/projects/42/metrics | jq '.'

# Check portfolio scheduler
curl http://localhost:6100/api/portfolio/status | jq '.'

# Check Redis connection
./.venv/bin/python -c "from services.plms.idempotency import get_cache; cache = get_cache(); print('✓ Redis connected')"
```

---

## 📁 What Was Added (Option A + C)

### New Files

**Services**:
- `services/plms/api/portfolio.py` - Portfolio status endpoint
- `services/plms/kpi_emit.py` - KPI receipts emitter (CLI tool)

**Scripts**:
- `scripts/pas_task_template_example.sh` - PAS task template with KPI emission
- `tests/demos/demo1_idempotent_start.sh` - Idempotency demo
- `tests/demos/demo2_representative_canary.sh` - Canary demo
- `tests/demos/demo3_kpi_gated_completion.sh` - KPI-gated demo

**Migrations**:
- `migrations/2025_11_06_plms_v1_tier1_rollback.sql` - Rollback script (emergency use)

**Documentation**:
- `docs/PLMS_OPS_GUARDRAILS.md` - Ops playbook (alerting, caps, policies)
- `docs/PLMS_QUICKSTART.md` - This guide

### Modified Files

**API**:
- `services/plms/api/projects.py` - Added `Idempotent-Replay` header to fresh responses

---

## 🎬 Running All Three Demos

```bash
# Ensure Redis is running
docker ps | grep redis

# Ensure API server is running
curl -s http://localhost:6100/api/portfolio/status >/dev/null && echo "✓ API ready"

# Run demos sequentially
./tests/demos/demo1_idempotent_start.sh
./tests/demos/demo2_representative_canary.sh
./tests/demos/demo3_kpi_gated_completion.sh

# All demos should exit 0 (success)
```

---

## 🔧 PAS Integration

Add this to your PAS task templates:

```bash
#!/bin/bash
# Example: Code-API task template

# ... your task execution code ...

# Post-task: Emit KPI receipt
python -m services.plms.kpi_emit \
    --task-id "$TASK_ID" \
    --lane "$LANE_ID" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --output "$ARTIFACTS_DIR/kpi_receipt.json"

# Exit with KPI validation status
exit $?
```

**See**: `scripts/pas_task_template_example.sh` for complete example

---

## 📊 API Endpoints Reference

### Project Management

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/projects/{id}/start` | POST | Start execution (requires `Idempotency-Key` header) |
| `/api/projects/{id}/simulate` | POST | Rehearsal mode (`?rehearsal_pct=0.01`) |
| `/api/projects/{id}/metrics` | GET | Get estimates (`?with_ci=1` for credible intervals) |
| `/api/projects/{id}/lane-overrides` | GET | Active learning feedback |
| `/api/projects/{id}/budget-runway` | GET | Budget depletion time |
| `/api/projects/{id}/risk-heatmap` | GET | Lane × phase risk scores |

### Portfolio Management

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/portfolio/status` | GET | Queue status, lane utilization, fair-share |
| `/api/portfolio/lanes` | GET | Configured lane concurrency limits |

---

## 🛡️ Production Checklist

Before deploying to production:

- [ ] Redis cluster running (3+ nodes for HA)
- [ ] Database migrations applied
- [ ] PagerDuty integration configured
- [ ] Slack webhook set up (`#plms-alerts`)
- [ ] Cron job scheduled (`check_plms_invariants.py` at 02:00 ET)
- [ ] Lane caps tuned (see `PLMS_OPS_GUARDRAILS.md`)
- [ ] All 3 demos passing
- [ ] Rollback migration tested on staging

**See**: `docs/PLMS_OPS_GUARDRAILS.md` for complete ops playbook

---

## 🎯 Next Steps (Option B Tonight)

**Nightly Invariants Checker** (cron job):
```bash
# Edit crontab
crontab -e

# Add entry (runs at 02:00 ET daily)
0 2 * * * /usr/bin/env DB_PATH=/srv/registry.db /srv/app/scripts/check_plms_invariants.py --db $DB_PATH >> /var/log/plms_invariants.log 2>&1
```

---

## 📚 Documentation Index

- **Quick Start**: `docs/PLMS_QUICKSTART.md` (this file)
- **Ops Guardrails**: `docs/PLMS_OPS_GUARDRAILS.md` (alerting, caps, calibration policy)
- **PRD**: `docs/PRDs/PRD_Project_Lifecycle_Management_System_PLMS.md` (complete spec)
- **HMI Integration**: `docs/HMI_JSON_CONTRACTS_PLMS.md` (frontend guide)
- **Test Vectors**: `tests/api/plms_test_vectors.sh` (10 API test cases)

---

## 🐛 Troubleshooting

### Redis Connection Failed
```bash
# Check if Redis is running
redis-cli ping

# Start Redis if needed
docker run -d -p 6379:6379 redis:alpine

# Verify connection
export REDIS_URL=redis://localhost:6379/0
./.venv/bin/python -c "from services.plms.idempotency import get_cache; get_cache()"
```

### Demo Scripts Failing
```bash
# Check API server is running
curl http://localhost:6100/api/portfolio/status

# Check required dependencies
./.venv/bin/pip install -q redis fastapi uvicorn

# Run with debug output
bash -x ./tests/demos/demo1_idempotent_start.sh
```

### KPI Emitter Errors
```bash
# Verify module can be imported
./.venv/bin/python -c "from services.plms.kpi_emit import emit_receipt; print('✓')"

# Check artifacts directory exists
ls -la /tmp/plms_demo3/artifacts/t9999/

# Run with verbose output
./.venv/bin/python -m services.plms.kpi_emit --help
```

---

## 🎉 Success Criteria

You're ready to demo when:
1. ✅ All 3 demos pass
2. ✅ Redis idempotency header shows "true" on duplicate requests
3. ✅ Canary sampling shows `strata_coverage = 1.0`
4. ✅ KPI-gated demo blocks completion on failure
5. ✅ Portfolio status endpoint returns queue + utilization metrics

---

**Questions?** See `docs/PLMS_OPS_GUARDRAILS.md` or contact the platform team.

**Last Updated**: 2025-11-06
