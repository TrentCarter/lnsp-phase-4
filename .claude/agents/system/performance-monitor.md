---
name: performance-monitor
display_name: Performance Monitor
role: system
tier: 1
mode: long
parent: architect
capabilities:
  - metrics_tracking
  - performance_analysis
resources:
  token_budget:
    target_ratio: 0.0
    hard_max_ratio: 0.0
  cpu_cores: 1
  memory_mb: 1024
heartbeat:
  interval_s: 30
metadata:
  version: 1.0.0
  port: 6116
  status: not_implemented
---

# Performance Monitor
Tracks and analyzes system performance metrics.
