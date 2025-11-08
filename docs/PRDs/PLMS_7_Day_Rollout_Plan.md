# PLMS Tier 1 - 7-Day Rollout Plan

**Version**: 1.0
**Dates**: November 8-14, 2025 (ET)
**Status**: Ready for Execution
**Owner**: Operations + Engineering + PM

---

## 🎯 Rollout Overview

**Goal**: Deploy PLMS Tier 1 to production with zero-downtime, progressive validation, and automated monitoring.

**Phases**:
- Day 0 (Fri Nov 7): Prep & freeze
- Days 1-2 (Weekend): Infrastructure deployment (low-risk)
- Days 3-5 (Mon-Wed): Feature enablement & calibration
- Days 6-7 (Thu-Fri): Stress testing & tuning

**Success Criteria**:
- ✅ All 3 demos pass on prod
- ✅ Two concurrent projects complete with KPI gates enforced
- ✅ Invariants pass 3 consecutive nights
- ✅ Grafana/alerts operational
- ✅ Retro completed with tuning recommendations

---

## Day 0 — Prep & Freeze (Fri Nov 7, 20:00–22:00 ET)

**Owner**: Operations
**Duration**: 2 hours
**Change Risk**: None (prep only)

### Tasks

1. **Tag Release**
   ```bash
   git tag -a plms-v1.0.0 -m "PLMS Tier 1 production release"
   git push origin plms-v1.0.0
   ```

2. **Backup Critical Data**
   ```bash
   # Registry database
   sqlite3 artifacts/registry/registry.db ".backup /backups/registry_$(date +%Y%m%d_%H%M%S).db"

   # PLMS logs
   tar -czf /backups/plms_logs_$(date +%Y%m%d_%H%M%S).tar.gz /var/log/plms/

   # Verify backups
   ls -lh /backups/registry_* /backups/plms_logs_*
   ```

3. **Verify Access & Credentials**
   ```bash
   # Slack webhook
   curl -X POST $SLACK_WEBHOOK_URL \
     -H 'Content-Type: application/json' \
     -d '{"text": "[PLMS] Day 0 prep test from prod"}'

   # SMTP relay (optional)
   echo "Test" | mail -s "[PLMS] Email test" $ALERT_EMAIL

   # PagerDuty (optional)
   export PD_ROUTING_KEY=your_routing_key
   python -m services.plms.alert_pd "[PLMS][test] PagerDuty integration check"
   ```

4. **Dry Run Invariants Checker**
   ```bash
   # Run on staging first
   ./scripts/check_plms_invariants.py \
     --db artifacts/registry/registry.db \
     --strict \
     --json | jq

   # Expected: status="OK", violations=0
   ```

### Success Criteria
- ✅ Git tag `plms-v1.0.0` exists
- ✅ Database backup verified (file size > 0)
- ✅ Slack webhook returns 200 OK
- ✅ Invariants checker returns OK status

### Rollback Plan
- N/A (no changes to production)

---

## Day 1 — Deploy Option B (Sat Nov 8, 10:00–12:00 ET)

**Owner**: Operations
**Duration**: 2 hours
**Change Risk**: Low (monitoring only)

### Tasks

1. **Deploy systemd Timer**
   ```bash
   # Copy unit files
   sudo cp deployment/systemd/plms-invariants.service /etc/systemd/system/
   sudo cp deployment/systemd/plms-invariants.timer /etc/systemd/system/

   # Create log directory
   sudo mkdir -p /var/log/plms
   sudo chown $(whoami):$(whoami) /var/log/plms

   # Enable timer (runs daily at 02:00 ET)
   sudo systemctl daemon-reload
   sudo systemctl enable plms-invariants.timer
   sudo systemctl start plms-invariants.timer

   # Verify
   systemctl list-timers | grep plms-invariants
   # Expected: "Next: Tomorrow 02:00:00 ET"
   ```

2. **Deploy Ops API**
   ```bash
   # Start API server (if not already running)
   ./.venv/bin/uvicorn services.plms.api.projects:router \
     --host 0.0.0.0 \
     --port 6100 \
     --reload &

   # Health check
   curl http://localhost:6100/api/ops/healthz | jq
   # Expected: {"status": "ok", "redis": true, "db": true, "filesystem": true}
   ```

3. **Synthetic Failure Test**
   ```bash
   # Temporarily corrupt a check (e.g., insert invalid run_kind)
   sqlite3 artifacts/registry/registry.db \
     "INSERT INTO project_runs (project_id, run_kind) VALUES (999, 'INVALID_KIND');"

   # Run checker
   ./scripts/run_invariants.sh

   # Expected: Slack message with "❌ FAIL" + violation details

   # Cleanup
   sqlite3 artifacts/registry/registry.db \
     "DELETE FROM project_runs WHERE project_id = 999;"
   ```

4. **Verify Alerting Channels**
   - ✅ Slack: Received failure notification within 30 seconds
   - ✅ Email: (if configured) Received failure email
   - ✅ PagerDuty: (if configured) Incident created

### Success Criteria
- ✅ systemd timer shows next run at 02:00 ET
- ✅ `/api/ops/healthz` returns 200 OK
- ✅ Synthetic failure triggers Slack alert
- ✅ Manual run after cleanup returns OK

### Rollback Plan
```bash
# Disable timer
sudo systemctl disable --now plms-invariants.timer

# Stop API
pkill -f "uvicorn services.plms.api.projects"

# Restore backup (if needed)
cp /backups/registry_YYYYMMDD_HHMMSS.db artifacts/registry/registry.db
```

---

## Day 2 — Demo Hardening (Sun Nov 9, 13:00–15:00 ET)

**Owner**: Engineering
**Duration**: 2 hours
**Change Risk**: Low (testing only)

### Tasks

1. **Run Demo 1: Idempotent Start**
   ```bash
   cd tests/demos
   bash demo_1_idempotent_start.sh

   # Expected output:
   # ✓ First call: status=submitted, run_id=xyz
   # ✓ Second call (same key): Idempotent-Replay=true, run_id=xyz (unchanged)
   # ✓ Third call (new key): status=submitted, run_id=abc (new)
   ```

2. **Run Demo 2: Representative Canary**
   ```bash
   bash demo_2_representative_canary.sh

   # Expected output:
   # ✓ Rehearsal with 1%: strata_coverage=1.0 (all strata represented)
   # ✓ Projected tokens: ~15,000 (100x of 150)
   # ✓ Risk score: <0.2 (acceptable)
   ```

3. **Run Demo 3: KPI-Gated Completion**
   ```bash
   bash demo_3_kpi_gated_completion.sh

   # Expected output:
   # Phase 1: Run with failing tests (exit 1)
   # ✓ KPI validator catches failure (test_pass_rate < 0.95)
   # Phase 2: Fix + rerun (exit 0)
   # ✓ KPI validator passes (test_pass_rate = 1.0)
   # ✓ Calibration accepts run (updates priors)
   ```

4. **Document Results**
   ```bash
   # Capture outputs
   bash demo_1_idempotent_start.sh > /tmp/demo1_output.txt 2>&1
   bash demo_2_representative_canary.sh > /tmp/demo2_output.txt 2>&1
   bash demo_3_kpi_gated_completion.sh > /tmp/demo3_output.txt 2>&1

   # Save to documentation
   cat /tmp/demo*_output.txt > docs/PRDs/PLMS_Demo_Results_$(date +%Y%m%d).txt
   ```

### Success Criteria
- ✅ All 3 demo scripts exit with code 0
- ✅ No manual interventions required
- ✅ Demo output matches expected behavior
- ✅ Results documented in `/tmp/demo*_output.txt`

### Rollback Plan
- N/A (testing only, no production changes)

---

## Day 3 — Portfolio Scheduler (Mon Nov 10, 09:00–11:00 ET)

**Owner**: Engineering + Operations
**Duration**: 2 hours
**Change Risk**: Medium (resource allocation changes)

### Tasks

1. **Enable Lane Caps**
   ```bash
   # Update lane capacity configuration
   sqlite3 artifacts/registry/registry.db <<EOF
   UPDATE lane_configs SET max_concurrent = 4 WHERE lane_name = 'Code-API';
   UPDATE lane_configs SET max_concurrent = 2 WHERE lane_name = 'Data-Schema';
   UPDATE lane_configs SET max_concurrent = 6 WHERE lane_name = 'Narrative';
   UPDATE lane_configs SET max_concurrent = 3 WHERE lane_name = 'Viz-Charts';
   EOF

   # Verify
   sqlite3 artifacts/registry/registry.db \
     "SELECT lane_name, max_concurrent FROM lane_configs;"
   ```

2. **Start Two Concurrent Projects**
   ```bash
   # Project A: Code-API lane (small)
   curl -X POST http://localhost:6100/api/projects/101/start \
     -H "Idempotency-Key: $(uuidgen)" \
     -H "Content-Type: application/json" \
     -d '{"run_kind": "baseline"}' | jq

   # Project B: Data-Schema lane (small)
   curl -X POST http://localhost:6100/api/projects/102/start \
     -H "Idempotency-Key: $(uuidgen)" \
     -H "Content-Type: application/json" \
     -d '{"run_kind": "baseline"}' | jq
   ```

3. **Monitor Fair-Share Scheduling**
   ```bash
   # Watch portfolio status (refresh every 5 seconds)
   watch -n 5 "curl -s http://localhost:6100/api/portfolio/status | jq"

   # Expected:
   # - Both projects show steady progress
   # - No starvation (lane_utilization balanced)
   # - Queue depth reasonable (<10)
   ```

4. **Capture Scheduling Metrics**
   ```bash
   # Get portfolio status snapshot
   curl http://localhost:6100/api/portfolio/status | jq > /tmp/portfolio_status_day3.json

   # Check for starvation
   jq '.lanes[] | select(.utilization < 0.2)' /tmp/portfolio_status_day3.json
   # Expected: empty (no lanes starved)
   ```

### Success Criteria
- ✅ Both projects complete successfully
- ✅ No lane starvation (all lanes >20% utilization)
- ✅ Queue depth stays reasonable (<10)
- ✅ Lane caps enforced (no lane exceeds max_concurrent)

### Rollback Plan
```bash
# Disable lane caps (set to infinity)
sqlite3 artifacts/registry/registry.db <<EOF
UPDATE lane_configs SET max_concurrent = 999 WHERE lane_name IN ('Code-API', 'Data-Schema', 'Narrative', 'Viz-Charts');
EOF

# Kill running projects (if needed)
curl -X POST http://localhost:6100/api/projects/101/cancel
curl -X POST http://localhost:6100/api/projects/102/cancel
```

---

## Day 4 — Grafana + Alerts (Tue Nov 11, 10:00–12:00 ET)

**Owner**: Operations
**Duration**: 2 hours
**Change Risk**: Low (monitoring only)

### Tasks

1. **Import Grafana Dashboard**
   ```bash
   # Via Grafana UI (preferred)
   # 1. Go to http://localhost:3000/dashboards
   # 2. Click "Import"
   # 3. Upload grafana/plms_invariants_dashboard.json
   # 4. Select data source: "PLMS Ops API"

   # Or via API
   curl -X POST http://localhost:3000/api/dashboards/import \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $GRAFANA_API_KEY" \
     -d @grafana/plms_invariants_dashboard.json
   ```

2. **Configure Alert Rules**
   ```bash
   # Alert 1: Runway warning (T-15 minutes remaining)
   # Condition: budget_runway_minutes < 15
   # Severity: warning
   # Notification: Slack #plms-ops

   # Alert 2: Runway critical (T-5 minutes remaining)
   # Condition: budget_runway_minutes < 5
   # Severity: critical
   # Notification: Slack #plms-ops + PagerDuty

   # Alert 3: Invariants failure
   # Condition: invariants_status != "OK"
   # Severity: critical
   # Notification: Slack #plms-ops + Email

   # Alert 4: CI width spike (per lane)
   # Condition: ci_width_pct > 50
   # Severity: warning
   # Notification: Slack #plms-ops
   ```

3. **Test Alert Firing**
   ```bash
   # Trigger synthetic invariants failure (same as Day 1)
   sqlite3 artifacts/registry/registry.db \
     "INSERT INTO project_runs (project_id, run_kind) VALUES (999, 'INVALID_KIND');"

   # Run checker
   ./scripts/run_invariants.sh

   # Expected:
   # - Grafana panel shows "FAIL" status
   # - Slack alert fires within 1 minute
   # - PagerDuty incident created (if configured)

   # Cleanup
   sqlite3 artifacts/registry/registry.db \
     "DELETE FROM project_runs WHERE project_id = 999;"
   ```

4. **Verify Dashboard Panels**
   - ✅ Panel 1: Current Status (✓/✗)
   - ✅ Panel 2: Total Violations (0)
   - ✅ Panel 3: Success Rate (100% gauge)
   - ✅ Panel 4: Last Check Time (< 1 hour ago)
   - ✅ Panel 5: Status History (time series, last 30 days)
   - ✅ Panel 6: Violations by Type (stacked bars)
   - ✅ Panel 7: Mean Time to Fix (days)
   - ✅ Panel 8: Critical Checks (table)

### Success Criteria
- ✅ Grafana dashboard imported and rendering
- ✅ All 8 panels show data
- ✅ Test alert fires on synthetic failure
- ✅ Slack + email + PagerDuty (optional) alerts working

### Rollback Plan
```bash
# Delete dashboard (via Grafana UI or API)
curl -X DELETE http://localhost:3000/api/dashboards/uid/plms-invariants \
  -H "Authorization: Bearer $GRAFANA_API_KEY"

# Disable alert rules (temporarily)
curl -X PATCH http://localhost:3000/api/alert-notifications/1 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -d '{"enabled": false}'
```

---

## Day 5 — Calibration Gate (Wed Nov 12, 14:00–16:00 ET)

**Owner**: Engineering
**Duration**: 2 hours
**Change Risk**: Medium (affects estimation quality)

### Tasks

1. **Enable Calibration Policy**
   ```python
   # In services/plms/calibration.py
   # Policy: Include only (run_kind ∈ {baseline, hotfix}) ∧ validation_pass ∧ !write_sandbox

   def should_update_priors(run_metadata: dict) -> bool:
       """Determine if a run should update calibration priors."""
       allowed_kinds = {'baseline', 'hotfix'}

       return (
           run_metadata['run_kind'] in allowed_kinds
           and run_metadata['validation_pass'] is True
           and run_metadata.get('write_sandbox', False) is False
       )
   ```

2. **Seed Priors with Baseline Runs**
   ```bash
   # Run 2 small baseline projects to seed calibration data

   # Baseline 1: Code-API lane
   curl -X POST http://localhost:6100/api/projects/201/start \
     -H "Idempotency-Key: $(uuidgen)" \
     -H "Content-Type: application/json" \
     -d '{
       "run_kind": "baseline",
       "budget_tokens": 10000,
       "budget_duration_minutes": 30
     }' | jq

   # Wait for completion
   sleep 300

   # Baseline 2: Data-Schema lane
   curl -X POST http://localhost:6100/api/projects/202/start \
     -H "Idempotency-Key: $(uuidgen)" \
     -H "Content-Type: application/json" \
     -d '{
       "run_kind": "baseline",
       "budget_tokens": 8000,
       "budget_duration_minutes": 20
     }' | jq
   ```

3. **Verify Prior Updates**
   ```bash
   # Check estimate_versions table for new priors
   sqlite3 artifacts/registry/registry.db \
     "SELECT lane_name, phase_name, mean_tokens, std_tokens, confidence FROM estimate_versions ORDER BY created_at DESC LIMIT 10;"

   # Expected: 2 new rows (one per baseline run)
   ```

4. **Test Credible Intervals**
   ```bash
   # Get estimates with CIs for a new project
   curl "http://localhost:6100/api/projects/203/metrics?with_ci=1" | jq

   # Expected output:
   # {
   #   "tokens_mean": 9000,
   #   "tokens_ci_lower": 7200,  # 90% credible interval
   #   "tokens_ci_upper": 10800,
   #   "duration_mean": 25,
   #   "duration_ci_lower": 20,
   #   "duration_ci_upper": 30,
   #   ...
   # }
   ```

### Success Criteria
- ✅ Calibration policy enabled (code updated)
- ✅ 2 baseline runs completed successfully
- ✅ `estimate_versions` table has new priors (2+ rows)
- ✅ `/metrics?with_ci=1` returns non-empty credible intervals

### Rollback Plan
```bash
# Disable calibration updates (temporary)
sqlite3 artifacts/registry/registry.db \
  "UPDATE system_config SET value = 'false' WHERE key = 'calibration_enabled';"

# Revert to default priors (if needed)
sqlite3 artifacts/registry/registry.db \
  "DELETE FROM estimate_versions WHERE created_at > '2025-11-12 14:00:00';"
```

---

## Day 6 — Dual-Project Stress Test (Thu Nov 13, 13:00–16:00 ET)

**Owner**: PM + Engineering + Operations
**Duration**: 3 hours
**Change Risk**: High (full production load)

### Tasks

1. **Launch Two Real Projects**
   ```bash
   # Project Alpha: Code-API lane (medium complexity)
   curl -X POST http://localhost:6100/api/projects/301/start \
     -H "Idempotency-Key: $(uuidgen)" \
     -H "Content-Type: application/json" \
     -d '{
       "run_kind": "baseline",
       "budget_tokens": 50000,
       "budget_duration_minutes": 120
     }' | jq > /tmp/project_alpha_start.json

   # Project Beta: Data-Schema lane (medium complexity)
   curl -X POST http://localhost:6100/api/projects/302/start \
     -H "Idempotency-Key: $(uuidgen)" \
     -H "Content-Type: application/json" \
     -d '{
       "run_kind": "baseline",
       "budget_tokens": 40000,
       "budget_duration_minutes": 90
     }' | jq > /tmp/project_beta_start.json
   ```

2. **Monitor During Execution**
   ```bash
   # Terminal 1: Watch portfolio status
   watch -n 10 "curl -s http://localhost:6100/api/portfolio/status | jq"

   # Terminal 2: Watch budget runway (Project Alpha)
   watch -n 30 "curl -s http://localhost:6100/api/projects/301/budget-runway | jq"

   # Terminal 3: Watch budget runway (Project Beta)
   watch -n 30 "curl -s http://localhost:6100/api/projects/302/budget-runway | jq"

   # Terminal 4: Watch risk heatmap
   watch -n 60 "curl -s http://localhost:6100/api/projects/301/risk-heatmap | jq"
   ```

3. **Collect Success Metrics**
   ```bash
   # After both projects complete (2-3 hours)

   # Project Alpha metrics
   curl http://localhost:6100/api/projects/301/metrics?with_ci=1 > /tmp/alpha_final_metrics.json

   # Project Beta metrics
   curl http://localhost:6100/api/projects/302/metrics?with_ci=1 > /tmp/beta_final_metrics.json

   # Calculate MAE
   python3 <<EOF
   import json

   for project_id in [301, 302]:
       with open(f'/tmp/project_{project_id}_metrics.json') as f:
           data = json.load(f)

       mae_tokens = abs(data['tokens_actual'] - data['tokens_mean']) / data['tokens_actual']
       mae_duration = abs(data['duration_actual'] - data['duration_mean']) / data['duration_actual']

       print(f"Project {project_id}:")
       print(f"  Token MAE: {mae_tokens*100:.1f}%")
       print(f"  Duration MAE: {mae_duration*100:.1f}%")
       print(f"  CI contains actual: {data['tokens_ci_lower'] <= data['tokens_actual'] <= data['tokens_ci_upper']}")
   EOF
   ```

### Success Criteria (Both Projects)
- ✅ **KPI Gates Hold**: No "green echo, red tests" slips
  - Code-API: test_pass_rate ≥ 0.95, build_success = true
  - Data-Schema: schema_diff_count = 0, migration_success = true

- ✅ **Estimation MAE ≤ 30%**:
  - Token usage: |actual - mean| / actual ≤ 0.30
  - Duration: |actual - mean| / actual ≤ 0.30

- ✅ **CI Contains Actuals**:
  - tokens_ci_lower ≤ tokens_actual ≤ tokens_ci_upper
  - duration_ci_lower ≤ duration_actual ≤ duration_ci_upper

- ✅ **Budget Warnings Visible**:
  - T-15 warning triggered (Slack notification)
  - T-5 critical triggered (Slack + PagerDuty)

- ✅ **No Starvation**:
  - Both projects make steady progress
  - Fair-share scheduling balanced

### Rollback Plan
```bash
# Cancel both projects (if needed)
curl -X POST http://localhost:6100/api/projects/301/cancel
curl -X POST http://localhost:6100/api/projects/302/cancel

# Pause new project submissions
sqlite3 artifacts/registry/registry.db \
  "UPDATE system_config SET value = 'false' WHERE key = 'accept_new_projects';"

# Alert team
python -m services.plms.alert_pd "[PLMS][prod] Stress test rollback - projects cancelled"
```

---

## Day 7 — Retro & Tuning (Fri Nov 14, 11:00–12:30 ET)

**Owner**: PM (scribe) + Engineering + Operations
**Duration**: 1.5 hours
**Change Risk**: Low (documentation + config tuning)

### Tasks

1. **Review Metrics (Last 5 Runs)**
   ```bash
   # Pull last 5 completed runs
   sqlite3 artifacts/registry/registry.db <<EOF
   SELECT
     project_id,
     lane_name,
     tokens_actual,
     tokens_mean,
     ABS(tokens_actual - tokens_mean) / tokens_actual AS mae_pct,
     (tokens_ci_upper - tokens_ci_lower) / tokens_mean AS ci_width_pct,
     validation_pass
   FROM project_runs
   WHERE status = 'completed'
   ORDER BY completed_at DESC
   LIMIT 5;
   EOF
   ```

2. **Calculate Aggregate Stats**
   ```bash
   # Average MAE by lane
   sqlite3 artifacts/registry/registry.db <<EOF
   SELECT
     lane_name,
     AVG(ABS(tokens_actual - tokens_mean) / tokens_actual) AS avg_mae,
     AVG((tokens_ci_upper - tokens_ci_lower) / tokens_mean) AS avg_ci_width,
     SUM(CASE WHEN validation_pass THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS kpi_pass_rate
   FROM project_runs
   WHERE status = 'completed'
     AND run_kind IN ('baseline', 'hotfix')
   GROUP BY lane_name;
   EOF
   ```

3. **Tune Configuration (If Needed)**
   ```bash
   # Example: Increase Code-API lane cap if utilization consistently >90%
   sqlite3 artifacts/registry/registry.db \
     "UPDATE lane_configs SET max_concurrent = 6 WHERE lane_name = 'Code-API';"

   # Example: Increase calibration learning rate (α) if MAE >30%
   sqlite3 artifacts/registry/registry.db \
     "UPDATE system_config SET value = '0.3' WHERE key = 'calibration_alpha';"

   # Example: Tighten KPI threshold if failure rate <5%
   # (Manually update services/plms/kpi_validators.py)
   ```

4. **Publish Rollout Report**
   ```bash
   # Create report document
   cat > docs/PRDs/PLMS_Rollout_Report_v1.0.md <<'EOF'
   # PLMS Tier 1 Rollout Report

   **Date**: November 14, 2025
   **Version**: 1.0
   **Status**: ✅ Production

   ## Executive Summary
   - **Duration**: 7 days (Nov 8-14)
   - **Projects Completed**: 6 (2 seed + 2 stress + 2 rehearsal)
   - **Incidents**: 0
   - **Uptime**: 100%

   ## Key Metrics
   | Lane        | Avg MAE | Avg CI Width | KPI Pass Rate |
   |-------------|---------|--------------|---------------|
   | Code-API    | 24.2%   | 38.5%        | 100%          |
   | Data-Schema | 18.7%   | 32.1%        | 100%          |
   | Narrative   | N/A     | N/A          | N/A           |

   ## Tuning Changes
   - Code-API lane cap: 4 → 6 (utilization >90%)
   - Calibration α: 0.2 → 0.3 (faster learning)
   - KPI test threshold: 0.95 → 0.98 (tighter quality gate)

   ## Sign-Off
   - Engineering: [Name] (Approved)
   - Operations: [Name] (Approved)
   - PM: [Name] (Approved)
   EOF

   # Commit to repo
   git add docs/PRDs/PLMS_Rollout_Report_v1.0.md
   git commit -m "docs: PLMS Tier 1 rollout report"
   git push origin main
   ```

### Success Criteria
- ✅ Metrics reviewed for last 5 runs
- ✅ Aggregate stats calculated by lane
- ✅ Tuning changes documented (if any)
- ✅ Rollout report published with sign-off

### Rollback Plan
- N/A (documentation only)

---

## 📋 Daily Standup Template

**Time**: 09:30 ET
**Duration**: 15 minutes
**Channel**: #plms-rollout

### Agenda
1. **Yesterday**: What was completed?
2. **Today**: What's the plan?
3. **Blockers**: Any issues or risks?
4. **Metrics**: Quick snapshot (MAE, CI width, incidents)

### Example (Day 4)
```
Yesterday: ✅ Portfolio scheduler deployed, 2 projects completed
Today: 🎯 Import Grafana dashboard, configure alerts
Blockers: None
Metrics: MAE=22%, CI width=35%, incidents=0
```

---

## 🚨 Escalation & Comms

### Communication Channels
- **Slack #plms-rollout**: Daily updates, Q&A
- **Email**: Critical incidents only (to: plms-ops@company.com)
- **PagerDuty**: Invariants failures, budget depletion (optional)

### Escalation Path
1. **Ops issue** (timer failure, API down) → Operations lead
2. **Estimation issue** (MAE >50%, CI empty) → Engineering lead
3. **KPI gate failure** (validation_pass=false) → PM + Engineering
4. **Incident** (data loss, corruption) → All leads + Slack #plms-rollout

### Change Window
- **Weekdays (Mon-Thu)**: 09:00–16:00 ET only
- **Weekends (Sat-Sun)**: Low-risk tasks only (monitoring, testing)
- **Freeze**: No changes during standup (09:30–09:45 ET)

---

## 🛡️ Guardrails (Live During Rollout)

### Hard Stops
1. **Budget Overrun**: Auto-pause if projected >25% over budget (unless approved by PM)
2. **Calibration Quarantine**: Exclude runs with KPI fail OR sandbox=true from priors
3. **Canary Coverage**: If strata_coverage <1.0, auto-bump rehearsal_pct to 5% (cap)

### Monitoring Thresholds
| Metric                  | Warning | Critical | Action                                |
|-------------------------|---------|----------|---------------------------------------|
| Estimation MAE          | >30%    | >50%     | Increase calibration α, review priors |
| CI Width                | >50%    | >75%     | Increase sample size, check variance  |
| KPI Pass Rate           | <90%    | <80%     | Tighten thresholds, review validators |
| Invariants Failures     | 1/week  | 2/week   | Root cause analysis, quarantine data  |
| Budget Runway (minutes) | <15     | <5       | Slack alert, PagerDuty (critical)     |

---

## ✅ "Done Means Done" Acceptance Criteria

Before declaring Tier 1 shipped, verify ALL of the following:

### Functional Requirements
- [ ] All 3 demos pass on prod (idempotent start, canary, KPI-gated)
- [ ] Two concurrent real projects complete successfully
- [ ] KPI gates enforced (no "green echo, red tests" slips)
- [ ] No lane starvation (all lanes >20% utilization)

### Quality Requirements
- [ ] Estimation MAE ≤30% (average across all lanes)
- [ ] Credible intervals contain actuals (≥90% of runs)
- [ ] KPI pass rate ≥95% (baseline/hotfix runs only)

### Operational Requirements
- [ ] Invariants pass 3 consecutive nights (no violations)
- [ ] Grafana dashboard operational (8 panels rendering)
- [ ] Alerts working (Slack + email tested, PagerDuty optional)
- [ ] systemd timer running (next run scheduled at 02:00 ET)

### Documentation Requirements
- [ ] Rollout report published with metrics & sign-off
- [ ] Tuning changes documented (lane caps, α, thresholds)
- [ ] Lessons learned captured in retro notes

### Sign-Off
- [ ] Engineering: ________________ (Date: ________)
- [ ] Operations: ________________ (Date: ________)
- [ ] PM: ________________ (Date: ________)

---

## 📚 Reference Documents

### Core Documentation
- **PLMS PRD**: `docs/PRDs/PRD_Project_Lifecycle_Management_System_PLMS.md`
- **HMI Contracts**: `docs/HMI_JSON_CONTRACTS_PLMS.md`
- **Systemd Setup**: `deployment/SYSTEMD_SETUP.md`
- **Grafana Setup**: `grafana/README.md`

### Test Vectors
- **Idempotent Start**: `tests/demos/demo_1_idempotent_start.sh`
- **Representative Canary**: `tests/demos/demo_2_representative_canary.sh`
- **KPI-Gated Completion**: `tests/demos/demo_3_kpi_gated_completion.sh`

### Scripts
- **Invariants Checker**: `scripts/check_plms_invariants.py`
- **Invariants Runner**: `scripts/run_invariants.sh`
- **Service Management**: `scripts/start_all_fastapi_services.sh`

---

## 🎉 Post-Rollout

### Immediate (Day 8, Nov 15)
- Send rollout report to stakeholders
- Archive rollout logs to `/backups/plms_rollout_logs/`
- Close rollout Slack channel (keep #plms-ops active)

### 1 Week Later (Day 14, Nov 21)
- Review 7-day metrics (MAE, CI width, KPI pass rate)
- Tune any outlier lanes (MAE >40% or CI width >60%)
- Publish updated thresholds in repo

### 1 Month Later (Dec 14)
- Full Tier 1 retrospective (What worked? What didn't?)
- Plan Tier 2 features (adaptive scheduling, Q-tower ranker)
- Update PLMS roadmap based on production learnings

---

## 🔍 Questions & Support

- **Deployment issues**: See `deployment/SYSTEMD_SETUP.md`
- **API questions**: See `docs/HMI_JSON_CONTRACTS_PLMS.md`
- **Invariants failures**: Run `./scripts/check_plms_invariants.py --help`
- **Emergency contact**: #plms-ops (Slack) or plms-ops@company.com

---

**End of Rollout Plan**
