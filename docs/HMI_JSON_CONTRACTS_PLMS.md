# HMI JSON Contracts for PLMS - Tier 1

**Date**: 2025-11-06
**Purpose**: Define JSON payloads for HMI overlays (Budget Runway + Risk Heatmap)

---

## 1. Budget Runway Gauge

**Endpoint**: `GET /api/projects/{id}/budget-runway`

**Purpose**: Show time-to-budget-depletion + projected overrun

**Response Schema**:
```json
{
  "budget": {
    "usd_max": 2.00,        // Maximum budget (from approval)
    "usd_spent": 1.10,      // Current spend
    "burn_per_min": 0.083   // Current burn rate ($/min)
  },
  "runway": {
    "minutes_to_depletion": 12.0,  // Time until budget exhausted
    "projected_overrun_usd": 0.32   // Expected overrun at completion
  },
  "status": "warning"  // "ok" | "warning" | "critical"
}
```

**Status Levels**:
- `"ok"`: Runway > 30 minutes, no projected overrun
- `"warning"`: Runway 10-30 minutes OR projected overrun < 20%
- `"critical"`: Runway < 10 minutes OR projected overrun ≥ 20%

**HMI Rendering**:
```html
<div class="budget-runway">
  <h4>Budget Runway</h4>
  <div class="gauge">
    <div class="gauge-bar" style="width: 55%"></div>
    <span class="gauge-label">$1.10 / $2.00 spent</span>
  </div>

  <div class="projection">
    <strong>Projected Depletion:</strong> 12 minutes
    <span class="warning">⚠️ High burn rate</span>
  </div>

  <div class="breakdown">
    <p>Current rate: $0.083/min</p>
    <p>Remaining budget: $0.90</p>
    <p>Estimated completion: $1.42 (⚠️ 29% over budget)</p>
  </div>
</div>
```

**CSS Classes**:
```css
.budget-runway .status.ok       { color: green; }
.budget-runway .status.warning  { color: orange; }
.budget-runway .status.critical { color: red; font-weight: bold; }
```

---

## 2. Risk Heatmap (Lane × Phase)

**Endpoint**: `GET /api/projects/{id}/risk-heatmap`

**Purpose**: Show risk scores per lane/phase (estimation, execution, validation)

**Response Schema**:
```json
{
  "lanes": [
    {
      "lane_id": 4200,
      "name": "Code-API",
      "estimation_risk": "low",
      "execution_risk": "medium",
      "validation_risk": "low",
      "signals": {
        "mae_pct": 0.12,        // Mean absolute error (%)
        "ci_width_pct": 0.28    // CI width (% of mean)
      }
    },
    {
      "lane_id": 5100,
      "name": "Data-Schema",
      "estimation_risk": "high",
      "execution_risk": "medium",
      "validation_risk": "high",
      "signals": {
        "mae_pct": 0.37,
        "ci_width_pct": 0.55
      }
    }
  ],
  "legend": {
    "low":    {"mae_pct_lt": 0.15, "ci_width_lt": 0.30},
    "medium": {"mae_pct_le": 0.30, "ci_width_le": 0.50},
    "high":   {"else": true}
  }
}
```

**Risk Level Calculation**:
```python
def compute_risk_level(mae_pct: float, ci_width_pct: float) -> str:
    """Compute risk level from MAE and CI width."""
    if mae_pct < 0.15 and ci_width_pct < 0.30:
        return "low"
    elif mae_pct <= 0.30 and ci_width_pct <= 0.50:
        return "medium"
    else:
        return "high"
```

**HMI Rendering**:
```html
<div class="risk-heatmap">
  <h4>Risk Heatmap (Lane × Phase)</h4>
  <table>
    <thead>
      <tr>
        <th>Lane</th>
        <th>Estimation</th>
        <th>Execution</th>
        <th>Validation</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Code-API (4200)</td>
        <td class="risk-low">✓ Low</td>
        <td class="risk-medium">⚠️ Medium</td>
        <td class="risk-low">✓ Low</td>
      </tr>
      <tr>
        <td>Data-Schema (5100)</td>
        <td class="risk-high">⚠️ High</td>
        <td class="risk-medium">⚠️ Medium</td>
        <td class="risk-high">⚠️ High</td>
      </tr>
    </tbody>
  </table>

  <div class="legend">
    <span class="risk-low">✓ Low: MAE &lt; 15%, CI width &lt; 30%</span>
    <span class="risk-medium">⚠️ Medium: MAE 15-30%, CI width 30-50%</span>
    <span class="risk-high">⚠️ High: MAE &gt; 30%, CI width &gt; 50%</span>
  </div>
</div>
```

**CSS Classes**:
```css
.risk-low    { background-color: #d4edda; color: #155724; }
.risk-medium { background-color: #fff3cd; color: #856404; }
.risk-high   { background-color: #f8d7da; color: #721c24; font-weight: bold; }
```

---

## 3. Estimation Drift Sparkline

**Endpoint**: `GET /api/projects/{id}/estimation-drift?lane_id={lane}`

**Purpose**: Show MAE trend over last N projects for a lane

**Response Schema**:
```json
{
  "lane_id": 4200,
  "lane_name": "Code-API",
  "history": [
    {"project_id": 1, "mae_pct": 0.22},
    {"project_id": 2, "mae_pct": 0.18},
    {"project_id": 3, "mae_pct": 0.15},
    {"project_id": 4, "mae_pct": 0.19},
    {"project_id": 5, "mae_pct": 0.14},
    {"project_id": 6, "mae_pct": 0.12},
    {"project_id": 7, "mae_pct": 0.11},
    {"project_id": 8, "mae_pct": 0.13},
    {"project_id": 9, "mae_pct": 0.10},
    {"project_id": 10, "mae_pct": 0.12}
  ],
  "trend": "improving"  // "improving" | "stable" | "degrading"
}
```

**Trend Calculation**:
```python
def compute_trend(history: List[Dict]) -> str:
    """Compute trend from MAE history."""
    if len(history) < 3:
        return "stable"

    # Linear regression slope
    slope = compute_slope([h["mae_pct"] for h in history])

    if slope < -0.01:  # Decreasing MAE (improving)
        return "improving"
    elif slope > 0.01:  # Increasing MAE (degrading)
        return "degrading"
    else:
        return "stable"
```

**HMI Rendering**:
```html
<div class="estimation-drift">
  <h4>Estimation Drift (Code-API Lane)</h4>
  <canvas id="drift-sparkline-4200"></canvas>
  <p>Last 10 projects: MAE 22% → 18% → 15% → 12% (✓ improving)</p>
</div>

<script>
// Chart.js sparkline
new Chart(document.getElementById('drift-sparkline-4200'), {
  type: 'line',
  data: {
    labels: ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'],
    datasets: [{
      label: 'MAE %',
      data: [22, 18, 15, 19, 14, 12, 11, 13, 10, 12],
      borderColor: 'green',
      tension: 0.4
    }]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { display: false }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: { display: true, text: 'MAE %' }
      }
    }
  }
});
</script>
```

---

## 4. Integration Guide

### Polling vs WebSocket

**Recommended**: WebSocket (SSE) for live updates

```javascript
// HMI client (JavaScript)
const eventSource = new EventSource(`/api/projects/${projectId}/budget-runway/stream`);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  updateBudgetGauge(data);

  if (data.status === "critical") {
    showNotification("⚠️ Budget depleting in " + data.runway.minutes_to_depletion + " minutes!");
  }
};
```

### Refresh Intervals

- **Budget Runway**: Every 10 seconds (fast burn detection)
- **Risk Heatmap**: Every 60 seconds (slower changes)
- **Estimation Drift**: On page load only (historical data)

### Error Handling

```javascript
fetch(`/api/projects/${projectId}/budget-runway`)
  .then(res => res.json())
  .then(data => updateBudgetGauge(data))
  .catch(err => {
    console.error("Budget runway fetch failed:", err);
    showFallbackUI("Budget data unavailable");
  });
```

---

## 5. Backend Implementation Notes

### Caching Strategy

```python
from functools import lru_cache
from datetime import datetime, timedelta

@lru_cache(maxsize=128)
def get_budget_runway_cached(project_id: int, cache_key: str):
    """Cache budget runway for 10 seconds."""
    return compute_budget_runway(project_id)

def get_budget_runway(project_id: int):
    # Cache key includes timestamp rounded to 10s
    cache_key = datetime.now().timestamp() // 10
    return get_budget_runway_cached(project_id, str(cache_key))
```

### Database Queries

**Budget Runway**:
```sql
SELECT
    SUM(tokens_used) AS total_tokens,
    SUM(cost_usd) AS total_cost,
    MAX(updated_at) - MIN(started_at) AS duration_seconds
FROM action_logs
WHERE project_id = ? AND status = 'completed';
```

**Risk Heatmap**:
```sql
SELECT
    ev.lane_id,
    AVG(ev.mean_absolute_error_tokens / ev.tokens_mean) AS mae_pct,
    AVG(ev.tokens_stddev / ev.tokens_mean) AS ci_width_pct
FROM estimate_versions ev
WHERE ev.lane_id IS NOT NULL
  AND ev.n_observations >= 3
GROUP BY ev.lane_id;
```

---

## 6. Testing Checklist

- [ ] Budget runway updates every 10s during execution
- [ ] Status transitions: ok → warning → critical
- [ ] Risk heatmap shows correct color coding
- [ ] Sparkline renders with Chart.js
- [ ] WebSocket reconnects on disconnect
- [ ] Graceful fallback if API unavailable
- [ ] Mobile responsive (gauges scale down)
- [ ] Accessible (ARIA labels on status badges)

---

## 7. Example Workflows

### Workflow 1: Budget Overrun Warning

```
1. Project starts with $2.00 budget
2. After 5 minutes: $0.80 spent, burn rate $0.16/min
3. HMI polls /budget-runway → status: "ok"
4. After 10 minutes: $1.60 spent, burn rate $0.16/min
5. Projected completion: $1.92 (within budget)
6. HMI status: "warning" (low runway)
7. User sees notification: "Budget depleting in 2.5 minutes"
8. User increases budget to $2.50 via PATCH /projects/{id}/budget
9. HMI status returns to "ok"
```

### Workflow 2: High-Risk Lane Alert

```
1. User views risk heatmap for new project
2. Data-Schema lane shows: estimation_risk = "high" (MAE 37%)
3. HMI highlights row in red
4. User hovers → tooltip: "High variance lane, add 50% buffer"
5. User adjusts budget from $1.00 → $1.50
6. User proceeds with project
```

---

**Version**: 1.0
**Status**: Production Ready
**Last Updated**: 2025-11-06
