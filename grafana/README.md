# PLMS Grafana Dashboard

**Status**: Ready to deploy
**Version**: 1.0
**Data Source**: PLMS Ops API + Prometheus (optional)

---

## Quick Setup

### Option 1: Use PLMS Ops API (JSON data source)

```bash
# 1. Import dashboard
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d @grafana/plms_invariants_dashboard.json

# 2. Configure JSON data source
# In Grafana UI:
# - Go to Configuration → Data Sources → Add data source
# - Select "JSON"
# - URL: http://plms-api:6100/api/ops/invariants/history
# - Method: GET
# - Click "Save & Test"

# 3. Open dashboard
# http://grafana:3000/d/plms-invariants
```

### Option 2: Use Prometheus (recommended for production)

Requires exposing PLMS metrics in Prometheus format.

**Example Prometheus exporter** (`services/plms/prometheus_exporter.py`):

```python
from prometheus_client import Gauge, Counter, generate_latest
from fastapi import Response
import glob
import os

# Metrics
plms_status = Gauge('plms_invariants_status', 'Latest check status (1=OK, 0=FAIL)')
plms_violations = Gauge('plms_invariants_violations_total', 'Total violations')
plms_checks_passed = Counter('plms_invariants_checks_passed', 'Total passed checks')
plms_checks_failed = Counter('plms_invariants_checks_failed', 'Total failed checks')

@app.get("/metrics")
def metrics():
    # Update metrics from latest log
    latest = get_latest_invariants()
    plms_status.set(1 if latest["status"] == "ok" else 0)
    plms_violations.set(latest.get("violations", 0))

    return Response(generate_latest(), media_type="text/plain")
```

Then configure Prometheus scrape:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'plms'
    static_configs:
      - targets: ['plms-api:6100']
    metrics_path: '/metrics'
    scrape_interval: 60s
```

---

## Dashboard Panels

### Row 1: Current Status (4 panels)

1. **Current Status** (Stat)
   - Shows: ✓ PASS / ✗ FAIL / ? UNKNOWN
   - Color: Green (pass) / Red (fail) / Yellow (unknown)
   - Updates: Every 1 minute

2. **Total Violations** (Stat)
   - Shows: Count of violations in latest check
   - Color: Green (0) / Orange (1-9) / Red (10+)

3. **Check Success Rate** (Gauge)
   - Shows: % of passed checks in last 30 days
   - Thresholds: <90% (red), 90-95% (yellow), >95% (green)

4. **Last Check Time** (Stat)
   - Shows: Time since last check ran
   - Format: Hours/days ago

### Row 2: Time Series (2 panels)

5. **Check Status Over Time** (Time Series)
   - Shows: Pass/fail status history
   - Visualization: Step line (green = pass, red = fail)
   - Time range: Last 30 days

6. **Violations by Check Type** (Time Series)
   - Shows: Violations grouped by check (passport, KPI, calibration, etc.)
   - Visualization: Stacked bars
   - Legend: Right side table

### Row 3: Analysis (2 panels)

7. **Mean Time to Fix** (Stat)
   - Shows: Average days from FAIL to PASS
   - Thresholds: <1 day (green), 1-3 days (yellow), >3 days (red)

8. **Critical Checks (STRICT Mode)** (Table)
   - Shows: Critical violations (calibration pollution, missing snapshots)
   - Columns: Check name, severity, violation count, last seen
   - Sorting: By severity (critical first)

---

## Data Source Configuration

### JSON API Data Source

**URL**: `http://plms-api:6100/api/ops/invariants/history?limit=30`

**Method**: GET

**Response Format**:
```json
{
  "history": [
    {
      "timestamp": "2025-11-06T07:00:03Z",
      "status": "ok",
      "violations": 0,
      "log_path": "/logs/plms/invariants_2025-11-06T07:00:03Z.log"
    }
  ],
  "count": 30
}
```

**JSONPath Mappings**:
- `$.history[*].timestamp` → Time field
- `$.history[*].status` → Status field
- `$.history[*].violations` → Violations field

---

## Prometheus Metrics

If using Prometheus exporter:

| Metric | Type | Description |
|--------|------|-------------|
| `plms_invariants_status` | Gauge | Latest check status (1=OK, 0=FAIL) |
| `plms_invariants_violations_total` | Gauge | Total violations in latest check |
| `plms_invariants_checks_passed` | Counter | Cumulative passed checks |
| `plms_invariants_checks_failed` | Counter | Cumulative failed checks |
| `plms_invariants_last_check_timestamp` | Gauge | Unix timestamp of last check |
| `plms_invariants_violations_by_check{check_name}` | Gauge | Violations per check type |
| `plms_invariants_time_to_fix_seconds` | Histogram | Time to fix violations (FAIL→PASS) |

---

## Alerts

Configure Grafana alerts on these conditions:

### Critical Alert: Check Failed

```yaml
# Alert condition
expr: plms_invariants_status == 0
for: 5m
severity: critical
message: "🔴 PLMS invariants check FAILED - {{ $value }} violations detected"
```

### Warning Alert: Check Stale

```yaml
# Alert condition
expr: (time() - plms_invariants_last_check_timestamp) > 90000
for: 1h
severity: warning
message: "⚠️ PLMS invariants check stale (last run >25h ago)"
```

### Warning Alert: Success Rate Low

```yaml
# Alert condition
expr: (sum(plms_invariants_checks_passed) / sum(plms_invariants_checks_total)) < 0.90
for: 1d
severity: warning
message: "⚠️ PLMS invariants success rate below 90% (7-day avg)"
```

---

## Customization

### Change Time Range

Edit dashboard JSON:
```json
"time": {
  "from": "now-7d",  // Change to "now-7d", "now-90d", etc.
  "to": "now"
}
```

### Add Custom Panels

1. Open dashboard in Grafana UI
2. Click "+ Add panel"
3. Configure query and visualization
4. Save dashboard
5. Export JSON: Settings → JSON Model → Copy

### Modify Thresholds

Edit `fieldConfig.defaults.thresholds` in panel JSON:
```json
"thresholds": {
  "steps": [
    {"value": 0, "color": "green"},
    {"value": 10, "color": "red"}  // Adjust threshold
  ]
}
```

---

## Troubleshooting

### "No data" in panels

1. Check PLMS API is running: `curl http://plms-api:6100/api/ops/invariants/history`
2. Check logs exist: `ls -la logs/plms/invariants_*.log`
3. Verify data source configuration in Grafana
4. Check data source test: Configuration → Data Sources → Test

### Metrics not updating

1. Check refresh interval: Dashboard settings → Time range
2. Verify scrape interval in Prometheus config
3. Check PLMS API `/metrics` endpoint: `curl http://plms-api:6100/metrics`

### Authentication errors

1. Add auth to data source: Configuration → Data Sources → Auth section
2. Use Bearer token or Basic auth as needed
3. Verify API allows cross-origin requests (CORS)

---

## Production Recommendations

1. **Use Prometheus** over JSON API for better performance
2. **Set up alerting** to Slack/PagerDuty (see above)
3. **Snapshot daily** to track dashboard evolution
4. **Archive old logs** (keep 30 days, compress older)
5. **Monitor Grafana itself** (disk space, memory)

---

**Last Updated**: 2025-11-06
**Maintainer**: Platform Team
